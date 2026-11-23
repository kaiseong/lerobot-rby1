"""Low-overhead debug recorder for async Pi0.5 inference runs.

Enabled with ``--debug_record=true`` on ``robot_client_demo`` (key 'r' turns it
on/off at runtime; every ON period is its own session). Every hook
called from the control loop, the receiver thread or the SDK state thread only
builds a small tuple and does ``put_nowait`` on a bounded queue; a full queue
counts a drop instead of blocking. One writer thread does the JPEG encoding,
file I/O and the periodic part-file flush. It keeps the process priority on
purpose: a lower-niced thread that is preempted while holding the GIL would
stall the control thread when the cores are saturated (priority inversion).

Session layout (``<debug_record_dir>/<YYYYmmdd_HHMMSS>/``)::

    meta.json            config, clock mapping, feature names, tasks, git state
    events.jsonl         start/stop, pause/resume, task switch, failures, drops
    ticks/part_*.npz     one row per control-loop iteration
    obs/part_*.npz       one row per observation captured and queued
    requests/part_*.npz  one row per server request (raw + converted chunk)
    state/part_*.npz     SDK state stream (optional)
    images/<cam>/r<req>_t<timestep>.jpg   images actually sent to the server
    stats.json           counts, drops, writer CPU time (written by close())

Hooks may still be called on a recorder that is being closed or is closed
(the client swaps recorders while its threads run): they are no-ops then.
A request begun but not ended before close() is stored with status
``recording_stopped``, so every image set has its request row.

All times are ``time.monotonic()`` seconds; ``meta.json`` maps them to wall
clock. Part files are written to a temporary name and renamed, so a crash
leaves only complete parts and loses at most one flush interval.

Images: lerobot cameras rebind ``latest_frame`` to a new array for every frame
and never write into a returned frame, so arrays that own their memory
(``cv2.cvtColor``/``cv2.rotate`` output of the OpenCV cameras) are queued by
reference. Arrays that do not own their memory are copied once in the receiver
thread: the RealSense camera without color conversion or rotation returns a
view of the librealsense frame buffer, and holding many of those would pin the
SDK frame pool (frame drops / read-loop timeouts). Encoding uses
``cv2.imencode`` on the RGB->BGR view, which runs on the writer thread only
(no OpenCV thread pool) with the GIL released.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

logger = logging.getLogger("infer_recorder")

STATE_FIELDS = ("position", "velocity", "current", "torque", "target_position", "temperature")
MODEL_INDEX_FIELDS = ("torso_idx", "right_arm_idx", "left_arm_idx", "head_idx", "mobility_idx")
# Remote observation image key -> config attribute holding the robot camera key.
REMOTE_IMAGE_KEYS = {
    "observation/front_image": "front_camera_key",
    "observation/left_image": "left_wrist_camera_key",
    "observation/right_image": "right_wrist_camera_key",
}

STREAM_DIRS = {"tick": "ticks", "obs": "obs", "request": "requests", "state": "state"}
TICK_EXECUTED, TICK_EMPTY, TICK_BLOCKED = 0, 1, 2
WARN_INTERVAL_S = 5.0
FLUSH_INTERVAL_S = 2.0
# The writer blocks on the image queue at most this long between record drains.
WRITER_IDLE_S = 0.05
# On close, images still queued after this budget are dropped (not encoded).
CLOSE_IMAGE_BUDGET_S = 1.0
CLOSE_TIMEOUT_S = 3.0
GIT_TIMEOUT_S = 2.0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _config_dict(config: Any) -> Any:
    if is_dataclass(config) and not isinstance(config, type):
        try:
            return _jsonable(asdict(config))
        except Exception:  # noqa: BLE001 - fall back to the shallow view
            pass
    return _jsonable(dict(getattr(config, "__dict__", {})))


def _git_state(path: Path) -> dict[str, Any]:
    """Commit and dirty flag of the checkout containing ``path`` (one subprocess)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v2", "--branch", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - provenance is best effort
        return {"error": repr(exc)}
    lines = out.splitlines()
    commit = next((ln.split()[-1] for ln in lines if ln.startswith("# branch.oid")), None)
    branch = next((ln.split()[-1] for ln in lines if ln.startswith("# branch.head")), None)
    changed = [ln for ln in lines if not ln.startswith("#")]
    return {"commit": commit, "branch": branch, "dirty": bool(changed), "changed_files": len(changed)}


class DiskLowError(RuntimeError):
    """Free disk is below the level at which a running recorder stops itself."""


def _require_free_disk(root: Path, min_bytes: float) -> None:
    """Refuse to start a session that the disk guard would stop right away."""
    existing = next((p for p in (root, *root.parents) if p.exists()), None)
    if existing is None or min_bytes <= 0:
        return
    free = shutil.disk_usage(existing).free
    if free < min_bytes:
        raise DiskLowError(f"free disk {free / 1e9:.2f} GB below {min_bytes / 1e9:.2f} GB at {existing}")


def _to_numpy(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


class RequestRecord:
    """Mutable record of one server request, filled by the receiver thread."""

    __slots__ = (
        "req_id", "generation", "obs_timestep", "obs_ts", "prompt", "state_sent", "t_send", "t_recv",
        "t_fail", "action_dict", "status", "error", "queue_before", "queue_after", "latest_action",
        "timed_actions",
    )

    def __init__(self, req_id: int, generation: int, observation: Any, remote_observation: dict[str, Any]):
        self.req_id = req_id
        self.generation = generation
        self.obs_timestep = observation.get_timestep()
        self.obs_ts = observation.get_timestamp()
        self.prompt = remote_observation.get("prompt", "")
        self.state_sent = remote_observation.get("observation/state")
        self.t_send = time.monotonic()
        self.t_recv = math.nan
        self.t_fail = math.nan
        self.action_dict = None
        self.status = "pending"
        self.error = ""
        self.queue_before = -1
        self.queue_after = -1
        self.latest_action = -1
        self.timed_actions = None

    def received(self, action_dict: dict[str, Any]) -> None:
        self.t_recv = time.monotonic()
        self.action_dict = action_dict

    def discard(self, reason: str) -> None:
        self.status = reason

    def before_aggregate(self, queue_size: int, latest_action: int) -> None:
        self.queue_before = queue_size
        self.latest_action = latest_action

    def accept(self, timed_actions: list[Any], queue_size: int) -> None:
        self.timed_actions = timed_actions
        self.queue_after = queue_size
        self.status = "accepted"

    def fail(self, error: Any) -> None:
        self.t_fail = time.monotonic()
        self.status = "interrupted" if error == "interrupted" else "error"
        self.error = str(error)[:500]


class InferenceRecorder:
    def __init__(
        self,
        root_dir: str | os.PathLike,
        *,
        action_names: list[str],
        config: Any = None,
        camera_keys: dict[str, str] | None = None,
        tasks: dict[str, Any] | None = None,
        note: str = "",
        record_images: bool = True,
        jpeg_quality: int = 90,
        min_free_gb: float = 2.0,
        environment_dt: float = 1 / 30,
        extra_meta: dict[str, Any] | None = None,
        queue_size: int = 8192,
        image_queue_size: int = 8,
        flush_interval_s: float = FLUSH_INTERVAL_S,
    ):
        cpu_start = time.process_time()
        root = Path(root_dir).expanduser()
        _require_free_disk(root, float(min_free_gb) * 1e9 / 2)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = root / stamp
        suffix = 1
        while session.exists():
            session = root / f"{stamp}_{suffix}"
            suffix += 1
        session.mkdir(parents=True)
        self.session_dir = session

        self.action_names = list(action_names)
        self.camera_keys = dict(camera_keys or {})
        self.record_images = record_images
        self.jpeg_quality = int(jpeg_quality)
        self.min_free_bytes = float(min_free_gb) * 1e9
        self.flush_interval_s = flush_interval_s

        self._queue: Queue = Queue(maxsize=queue_size)
        self._image_queue: Queue = Queue(maxsize=image_queue_size)
        self.active = True
        self.images_active = record_images
        self.drops = {"tick": 0, "obs": 0, "request": 0, "state": 0, "event": 0, "image": 0}
        self.counts = {"tick": 0, "obs": 0, "request": 0, "state": 0, "event": 0, "image": 0}
        self.hook_errors = 0
        self.writer_errors = 0
        self.bytes_written = 0
        self.writer_cpu_s = 0.0
        self.close_image_drops = 0
        self._process_cpu_start = cpu_start
        self._image_deadline = math.inf
        self._last_hook_warn = 0.0
        self._pending_action: tuple | None = None
        self._req_counter = 0
        self._req_lock = threading.Lock()
        # Requests begun and not yet ended; close() stores them (guarded by _req_lock).
        self._open_requests: set[RequestRecord] = set()
        self._close_lock = threading.Lock()
        self._state_robot = None
        self._stop = threading.Event()
        self._closed = False

        self._task_keys = {
            str(v.get("task")): str(k) for k, v in (tasks or {}).items() if isinstance(v, dict)
        }
        self._buffers: dict[str, list[tuple]] = {"tick": [], "obs": [], "request": [], "state": []}
        self._part_index = {name: 0 for name in self._buffers}
        self._events_file = open(session / "events.jsonl", "a", encoding="utf-8")  # noqa: SIM115

        # Same instant on every clock; all records use time.monotonic().
        t_wall, t_mono, t_perf = time.time(), time.monotonic(), time.perf_counter()
        self.perf_to_mono = t_mono - t_perf
        self.t0 = t_mono
        meta = {
            "format": "infer_record/1",
            "created": datetime.now().isoformat(timespec="milliseconds"),
            "host": platform.node(),
            "clock": {
                "wall_time": t_wall, "monotonic": t_mono, "perf_counter": t_perf,
                "perf_to_monotonic": self.perf_to_mono,
            },
            "action_names": self.action_names,
            "camera_keys": self.camera_keys,
            "image_channel_order": "rgb",
            "environment_dt": environment_dt,
            "tasks": _jsonable(tasks or {}),
            "note": note,
            "record_images": record_images,
            "jpeg_quality": self.jpeg_quality,
            "min_free_gb": min_free_gb,
            "config": _config_dict(config) if config is not None else {},
            "git": _git_state(Path(__file__).resolve().parent),
            "tick_status": {"executed": TICK_EXECUTED, "empty": TICK_EMPTY, "blocked": TICK_BLOCKED},
            **_jsonable(extra_meta or {}),
        }
        self._meta = meta
        self._write_meta()

        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="InferRecorderWriter")
        self._writer.start()
        self.event("start", session=str(session))
        logger.info("Debug recording to %s", session)

    # ------------------------------------------------------------------ hot path
    def _put(self, stream: str, item: tuple) -> None:
        try:
            self._queue.put_nowait((stream, item))
        except Full:
            self.drops[stream] += 1

    def _hook_error(self, where: str, exc: BaseException) -> None:
        self.hook_errors += 1
        now = time.monotonic()
        if now - self._last_hook_warn > WARN_INTERVAL_S:
            self._last_hook_warn = now
            logger.warning("Recorder hook %s failed (%s errors so far): %r", where, self.hook_errors, exc)

    def action_sent(self, timed_action: Any, queue_size_before: int) -> None:
        """Control thread, right after robot.send_action() returned."""
        if not self.active:
            return
        try:
            self._pending_action = (
                time.monotonic(),
                timed_action.get_timestep(),
                timed_action.get_timestamp(),
                queue_size_before,
                timed_action.get_action(),
            )
        except Exception as exc:  # noqa: BLE001
            self._hook_error("action_sent", exc)

    def tick(self, loop_start_perf: float, blocked: bool) -> None:
        """Control thread, once per loop iteration after the action and observation steps."""
        if not self.active:
            return
        try:
            pending, self._pending_action = self._pending_action, None
            if pending is None:
                status = TICK_BLOCKED if blocked else TICK_EMPTY
                pending = (math.nan, -1, math.nan, -1, None)
            else:
                status = TICK_EXECUTED
            self._put("tick", (loop_start_perf + self.perf_to_mono, time.monotonic(), status, *pending))
        except Exception as exc:  # noqa: BLE001
            self._hook_error("tick", exc)

    def observation(
        self,
        observation: Any,
        raw_observation: dict[str, Any],
        capture_start_perf: float,
        capture_s: float,
        generation: int,
        queue_size: int,
        replaced: Any,
        task: str,
    ) -> None:
        """Control thread, after the observation was queued for the server.

        ``replaced`` is the not-yet-sent observation this one evicted (or None);
        ``task`` is the instruction text sent with this observation.
        """
        if not self.active:
            return
        try:
            # Copy only the joint floats; images are recorded when actually sent.
            joints = [raw_observation.get(k, math.nan) for k in self.action_names]
            t_start = capture_start_perf + self.perf_to_mono
            replaced_ts = math.nan if replaced is None else replaced.get_timestamp()
            self._put(
                "obs",
                (
                    t_start, t_start + capture_s, time.monotonic(), observation.get_timestamp(),
                    observation.get_timestep(), generation, bool(getattr(observation, "must_go", False)),
                    queue_size, replaced_ts, self._task_keys.get(task, "?"), joints,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._hook_error("observation", exc)

    def begin_request(
        self, generation: int, observation: Any, remote_observation: dict[str, Any]
    ) -> RequestRecord | None:
        """Receiver thread, right before the request is sent. Queues the sent images."""
        if not self.active:
            return None
        try:
            with self._req_lock:
                if not self.active:
                    return None
                req_id = self._req_counter
                self._req_counter += 1
                record = RequestRecord(req_id, generation, observation, remote_observation)
                self._open_requests.add(record)
            if self.images_active:
                images = {}
                for key in REMOTE_IMAGE_KEYS:
                    image = remote_observation.get(key)
                    if isinstance(image, np.ndarray):
                        # Views of SDK frame buffers are copied so the pool is not pinned.
                        images[self.camera_keys.get(key, key)] = image if image.flags.owndata else image.copy()
                try:
                    self._image_queue.put_nowait((req_id, record.obs_timestep, images))
                except Full:
                    self.drops["image"] += 1
            return record
        except Exception as exc:  # noqa: BLE001
            self._hook_error("begin_request", exc)
            return None

    def end_request(self, record: RequestRecord | None) -> None:
        """Receiver thread, after the response was aggregated, discarded or failed."""
        if record is None or not self.active:
            return
        try:
            with self._req_lock:
                if not self.active:
                    return  # closed meanwhile: close() stored it if it was in flight
                self._open_requests.discard(record)
                if record.status == "pending":
                    record.status = "unknown"
                self._put("request", (time.monotonic(), record))
            if record.status in ("error", "interrupted"):
                # Stamped at the failure, not after the reconnect delay.
                self._put("event", (record.t_fail, "request_" + record.status,
                                    {"req_id": record.req_id, "error": record.error}))
        except Exception as exc:  # noqa: BLE001
            self._hook_error("end_request", exc)

    def event(self, name: str, **fields: Any) -> None:
        """Any thread. Small structured event appended to events.jsonl."""
        if not self.active and name not in ("stop", "disk_low_stop"):
            return
        try:
            self._put("event", (time.monotonic(), name, fields))
        except Exception as exc:  # noqa: BLE001
            self._hook_error("event", exc)

    def _state_callback(self, state: Any, *_: Any) -> None:
        """SDK thread. Must never raise."""
        if not self.active:
            return
        try:
            t = time.monotonic()
            values = []
            for field in STATE_FIELDS:
                value = getattr(state, field, None)
                values.append(None if value is None else np.asarray(value, dtype=np.float32).copy())
            self._put("state", (t, *values))
        except Exception as exc:  # noqa: BLE001
            self._hook_error("state", exc)

    # ------------------------------------------------------------ state stream
    def start_state_stream(self, robot: Any, hz: float) -> bool:
        sdk = getattr(robot, "_robot", None)
        start = getattr(sdk, "start_state_update", None)
        if hz <= 0 or not callable(start):
            if hz > 0:
                self.event("state_stream_unavailable")
            return False
        model = getattr(robot, "_model", None)
        indices = {}
        for name in MODEL_INDEX_FIELDS:
            try:
                value = getattr(model, name, None)
                if value is not None:
                    indices[name] = [int(i) for i in np.asarray(value).reshape(-1)]
            except Exception:  # noqa: BLE001
                pass
        names = getattr(model, "robot_joint_names", None)
        self._meta["state_stream"] = {
            "hz": hz,
            "fields": list(STATE_FIELDS),
            "model_indices": indices,
            "joint_names": [str(n) for n in names] if names is not None else None,
        }
        try:
            self._write_meta()
        except Exception as exc:  # noqa: BLE001 - meta update is best effort
            logger.warning("Could not update meta.json with the state stream: %r", exc)
        try:
            start(self._state_callback, hz)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not start SDK state stream: %r", exc)
            self.event("state_stream_failed", error=repr(exc))
            return False
        self._state_robot = sdk
        self.event("state_stream_start", hz=hz)
        return True

    def stop_state_stream(self) -> None:
        sdk, self._state_robot = self._state_robot, None
        if sdk is None:
            return
        try:
            sdk.stop_state_update()
            self.event("state_stream_stop")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not stop SDK state stream: %r", exc)

    # ------------------------------------------------------------ lifecycle
    def close(self, timeout_s: float = CLOSE_TIMEOUT_S) -> None:
        """Stop the state stream, flush and join the writer. Thread safe; only the first call runs."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.stop_state_stream()
        # Hooks become no-ops; requests still in flight and the final "stop" event are queued.
        with self._req_lock:
            was_active, self.active = self.active, False
            in_flight, self._open_requests = self._open_requests, set()
        t_close = time.monotonic()
        for record in sorted(in_flight, key=lambda r: r.req_id) if was_active else ():
            # Snapshot: the receiver may still be filling the original.
            snapshot = copy.copy(record)
            snapshot.status = "recording_stopped"
            self._put("request", (t_close, snapshot))
        self.event("stop")
        # Bound the time spent on images still queued; numeric parts are always flushed.
        self._image_deadline = time.monotonic() + min(CLOSE_IMAGE_BUDGET_S, timeout_s / 2)
        self._stop.set()
        self._writer.join(timeout=timeout_s)
        if self._writer.is_alive():
            logger.warning("Recorder writer did not finish within %.1fs; some data may be missing", timeout_s)
        stats = {
            "counts": self.counts,
            "drops": self.drops,
            "hook_errors": self.hook_errors,
            "writer_errors": self.writer_errors,
            "bytes_written": self.bytes_written,
            "writer_cpu_s": self.writer_cpu_s,
            # Whole process (control, receiver, SDK and writer threads) since the recorder started.
            "process_cpu_s": time.process_time() - self._process_cpu_start,
            "images_dropped_on_close": self.close_image_drops,
            "duration_s": time.monotonic() - self.t0,
            "writer_finished": not self._writer.is_alive(),
        }
        try:
            (self.session_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write recorder stats: %r", exc)
        logger.info(
            "Debug recording closed: %s ticks, %s requests, %s images, drops %s",
            self.counts["tick"], self.counts["request"], self.counts["image"],
            {k: v for k, v in self.drops.items() if v},
        )
        if self.close_image_drops:
            logger.warning("Recorder skipped %s queued image sets at close", self.close_image_drops)

    # ------------------------------------------------------------ writer thread
    def _write_meta(self) -> None:
        tmp = self.session_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self._meta, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp, self.session_dir / "meta.json")

    def _writer_loop(self) -> None:
        # Same priority as the client on purpose (see module docstring).
        cpu_start = time.thread_time()
        next_flush = time.monotonic() + self.flush_interval_s
        reported_drops = dict(self.drops)
        self._check_disk()
        while True:
            stopping = self._stop.is_set()
            busy = False
            try:
                busy = self._drain_records()
                # Idle: sleep on the image queue so a new image is encoded right away.
                busy |= self._encode_one_image(timeout=None if busy or stopping else WRITER_IDLE_S)
                now = time.monotonic()
                if now >= next_flush:
                    next_flush = now + self.flush_interval_s
                    reported_drops = self._report_drops(reported_drops)
                    self._check_disk()
                    self._flush()
            except Exception as exc:  # noqa: BLE001 - keep the writer alive
                self._writer_error("writer_loop", exc)
            self.writer_cpu_s = time.thread_time() - cpu_start
            if stopping and not busy and self._queue.empty() and self._image_queue.empty():
                break
        self._report_drops(reported_drops)
        self._flush()
        self.writer_cpu_s = time.thread_time() - cpu_start
        try:
            self._events_file.close()
        except Exception:  # noqa: BLE001
            pass

    def _drain_records(self, limit: int = 2000) -> bool:
        busy = False
        for _ in range(limit):
            try:
                stream, item = self._queue.get_nowait()
            except Empty:
                break
            busy = True
            self.counts[stream] += 1
            if stream == "event":
                self._write_event(*item)
            else:
                self._buffers[stream].append(item)
        return busy

    def _write_event(self, t: float, name: str, fields: dict[str, Any]) -> None:
        try:
            line = json.dumps({"t": t, "event": name, **_jsonable(fields)}, ensure_ascii=False)
            self._events_file.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            self._writer_error("event", exc)

    def _encode_one_image(self, timeout: float | None = None) -> bool:
        try:
            if timeout is None:
                req_id, timestep, images = self._image_queue.get_nowait()
            else:
                req_id, timestep, images = self._image_queue.get(timeout=timeout)
        except Empty:
            return False
        if not self.images_active:
            return True
        if time.monotonic() > self._image_deadline:
            self.close_image_drops += 1
            return True
        import cv2

        for cam, image in images.items():
            if time.monotonic() > self._image_deadline:
                break  # closing: the remaining cameras of this set are skipped
            try:
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[-1] == 3:
                    # RGB -> BGR view; imencode copies it on this thread (cvtColor would use the OpenCV pool).
                    arr = arr[..., ::-1]
                ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if not ok:
                    raise RuntimeError("cv2.imencode failed")
                cam_dir = self.session_dir / "images" / cam
                cam_dir.mkdir(parents=True, exist_ok=True)
                (cam_dir / f"r{req_id:05d}_t{timestep:06d}.jpg").write_bytes(buf.tobytes())
                self.bytes_written += buf.nbytes
                self.counts["image"] += 1
            except Exception as exc:  # noqa: BLE001
                self._writer_error("image", exc)
        return True

    def _report_drops(self, reported: dict[str, int]) -> dict[str, int]:
        current = dict(self.drops)
        new = {k: current[k] - reported.get(k, 0) for k in current if current[k] != reported.get(k, 0)}
        if new:
            logger.warning("Recorder dropped records (queue full): new %s, total %s", new, current)
            self._write_event(time.monotonic(), "recorder_drops", {"new": new, "total": current})
        return current

    def _check_disk(self) -> None:
        if not self.active:
            return
        try:
            free = shutil.disk_usage(self.session_dir).free
        except Exception as exc:  # noqa: BLE001
            self._writer_error("disk_usage", exc)
            return
        if free < self.min_free_bytes / 2:
            self.active = False
            self.images_active = False
            logger.error(
                "Free disk %.2f GB below %.2f GB: debug recording stopped", free / 1e9, self.min_free_bytes / 2e9
            )
            # The SDK callback is a no-op now; close() stops the stream itself.
            self._write_event(time.monotonic(), "disk_low_stop", {"free_gb": free / 1e9})
        elif free < self.min_free_bytes and self.images_active:
            self.images_active = False
            logger.warning(
                "Free disk %.2f GB below %.2f GB: image recording stopped", free / 1e9, self.min_free_bytes / 1e9
            )
            self._write_event(time.monotonic(), "disk_low_images_off", {"free_gb": free / 1e9})

    def _writer_error(self, where: str, exc: BaseException) -> None:
        self.writer_errors += 1
        if self.writer_errors <= 5 or self.writer_errors % 100 == 0:
            logger.warning("Recorder writer error in %s (%s so far): %r", where, self.writer_errors, exc)

    def _flush(self) -> None:
        try:
            self._events_file.flush()
        except Exception as exc:  # noqa: BLE001
            self._writer_error("events_flush", exc)
        for stream, rows in self._buffers.items():
            if not rows:
                continue
            self._buffers[stream] = []
            try:
                arrays = getattr(self, f"_pack_{stream}")(rows)
                self._save_part(stream, arrays)
            except Exception as exc:  # noqa: BLE001
                self._writer_error(f"flush_{stream}", exc)

    def _save_part(self, stream: str, arrays: dict[str, np.ndarray]) -> None:
        part_dir = self.session_dir / STREAM_DIRS[stream]
        part_dir.mkdir(exist_ok=True)
        index = self._part_index[stream]
        self._part_index[stream] += 1
        final = part_dir / f"part_{index:05d}.npz"
        tmp = part_dir / f".part_{index:05d}.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, final)
        self.bytes_written += final.stat().st_size

    def _vectors(self, values: list[Any]) -> np.ndarray:
        dim = len(self.action_names)
        out = np.full((len(values), dim), np.nan, dtype=np.float32)
        for i, value in enumerate(values):
            if value is None:
                continue
            try:
                vec = _to_numpy(value)
                out[i, : min(dim, vec.size)] = vec[:dim]
            except Exception as exc:  # noqa: BLE001 - one bad row stays NaN
                self._writer_error("vector", exc)
        return out

    def _pack_tick(self, rows: list[tuple]) -> dict[str, np.ndarray]:
        t_loop, t_end, status, t_action, timestep, action_ts, queue_before, actions = zip(*rows, strict=True)
        return {
            "t_loop": np.asarray(t_loop, dtype=np.float64),
            "t_end": np.asarray(t_end, dtype=np.float64),
            "status": np.asarray(status, dtype=np.int8),
            "t_action": np.asarray(t_action, dtype=np.float64),
            "timestep": np.asarray(timestep, dtype=np.int64),
            "action_ts": np.asarray(action_ts, dtype=np.float64),
            "queue_before": np.asarray(queue_before, dtype=np.int32),
            "action": self._vectors(list(actions)),
        }

    def _pack_obs(self, rows: list[tuple]) -> dict[str, np.ndarray]:
        cols = list(zip(*rows, strict=True))
        names = ["t_capture_start", "t_capture_end", "t_queued", "obs_ts", "timestep", "generation", "must_go",
                 "queue_size", "replaced_ts", "task_key", "joints"]
        out = dict(zip(names, cols, strict=True))
        replaced_ts = np.asarray(out["replaced_ts"], dtype=np.float64)
        return {
            **{k: np.asarray(out[k], dtype=np.float64) for k in names[:4]},
            "timestep": np.asarray(out["timestep"], dtype=np.int64),
            "generation": np.asarray(out["generation"], dtype=np.int64),
            "must_go": np.asarray(out["must_go"], dtype=bool),
            "queue_size": np.asarray(out["queue_size"], dtype=np.int32),
            # This observation evicted a not-yet-sent one (whose obs_ts is replaced_ts).
            "replaced_prev": np.isfinite(replaced_ts),
            "replaced_ts": replaced_ts,
            "task_key": np.asarray(out["task_key"], dtype=str),
            "joints": np.asarray(out["joints"], dtype=np.float32).reshape(len(rows), len(self.action_names)),
        }

    def _pack_request(self, rows: list[tuple]) -> dict[str, np.ndarray]:
        scalars: dict[str, list] = {k: [] for k in (
            "req_id", "generation", "obs_timestep", "obs_ts", "t_send", "t_recv", "t_fail", "t_end", "request_ms",
            "infer_ms", "status", "error", "queue_before", "queue_after", "latest_action", "prompt")}
        states, raw_flat, raw_shapes, conv_ts, conv_actions = [], [], [], [], []
        conv_offsets = [0]
        for t_end, rec in rows:
            status, error = rec.status, rec.error
            infer_ms = math.nan
            raw = np.zeros((0,), dtype=np.float32)
            timed = rec.timed_actions or []
            try:
                if isinstance(rec.action_dict, dict):
                    timing = rec.action_dict.get("policy_timing")
                    if isinstance(timing, dict) and timing.get("infer_ms") is not None:
                        infer_ms = float(timing["infer_ms"])
                    if "actions" in rec.action_dict:
                        raw = np.asarray(rec.action_dict["actions"], dtype=np.float32)
                row_ts = [int(a.get_timestep()) for a in timed]
                row_actions = [a.get_action() for a in timed]
            except Exception as exc:  # noqa: BLE001 - keep the row, drop only what cannot be packed
                self._writer_error("pack_request_row", exc)
                raw, row_ts, row_actions = np.zeros((0,), dtype=np.float32), [], []
                error = (error + " | " if error else "") + f"recorder could not pack response: {exc!r}"[:300]
            raw_shapes.append(list(raw.shape[-2:]) if raw.ndim >= 2 else [raw.size, 1])
            raw_flat.append(raw.reshape(-1))
            conv_ts.extend(row_ts)
            conv_actions.extend(row_actions)
            conv_offsets.append(len(conv_ts))
            states.append(rec.state_sent)
            t_done = rec.t_recv if np.isfinite(rec.t_recv) else rec.t_fail
            for key, value in (
                ("req_id", rec.req_id), ("generation", rec.generation), ("obs_timestep", rec.obs_timestep),
                ("obs_ts", rec.obs_ts), ("t_send", rec.t_send), ("t_recv", rec.t_recv), ("t_fail", rec.t_fail),
                ("t_end", t_end), ("request_ms", (t_done - rec.t_send) * 1000), ("infer_ms", infer_ms),
                ("status", status), ("error", error), ("queue_before", rec.queue_before),
                ("queue_after", rec.queue_after), ("latest_action", rec.latest_action), ("prompt", rec.prompt),
            ):
                scalars[key].append(value)
        raw_offsets = np.cumsum([0] + [r.size for r in raw_flat])
        floats = ("obs_ts", "t_send", "t_recv", "t_fail", "t_end", "request_ms", "infer_ms")
        strings = ("status", "error", "prompt")
        out = {
            k: np.asarray(v, dtype=np.float64 if k in floats else (str if k in strings else np.int64))
            for k, v in scalars.items()
        }
        out.update(
            state_sent=self._vectors(states),
            raw_offsets=raw_offsets.astype(np.int64),
            raw_shapes=np.asarray(raw_shapes, dtype=np.int64).reshape(-1, 2),
            raw_actions=np.concatenate(raw_flat) if raw_flat else np.zeros((0,), np.float32),
            conv_offsets=np.asarray(conv_offsets, dtype=np.int64),
            conv_timesteps=np.asarray(conv_ts, dtype=np.int64),
            conv_actions=self._vectors(conv_actions),
        )
        return out

    def _pack_state(self, rows: list[tuple]) -> dict[str, np.ndarray]:
        out = {"t": np.asarray([r[0] for r in rows], dtype=np.float64)}
        for i, field in enumerate(STATE_FIELDS, start=1):
            first = next((r[i] for r in rows if r[i] is not None), None)
            if first is None:
                continue
            dim = first.size
            arr = np.full((len(rows), dim), np.nan, dtype=np.float32)
            for j, r in enumerate(rows):
                if r[i] is not None and r[i].size == dim:
                    arr[j] = r[i].reshape(-1)
            out[field] = arr
        return out


def create_recorder_for_client(client: Any) -> InferenceRecorder:
    """Build a recorder from a DemoPi05RobotClient's config and start the state stream."""
    cfg = client.config
    action_names = list(client.robot.action_features)
    camera_keys = {key: str(getattr(cfg, attr, key)) for key, attr in REMOTE_IMAGE_KEYS.items()}
    tasks = {key: {"label": label, "task": task} for key, (label, task) in getattr(client, "task_options", {}).items()}
    cameras = getattr(client.robot, "cameras", None)
    recorder = InferenceRecorder(
        cfg.debug_record_dir,
        action_names=action_names,
        config=cfg,
        camera_keys=camera_keys,
        tasks=tasks,
        note=getattr(cfg, "debug_record_note", ""),
        record_images=getattr(cfg, "debug_record_images", True),
        jpeg_quality=getattr(cfg, "debug_record_jpeg_quality", 90),
        min_free_gb=getattr(cfg, "debug_record_min_free_gb", 2.0),
        environment_dt=cfg.environment_dt,
        extra_meta={
            "robot_type": getattr(getattr(cfg, "robot", None), "type", None),
            "robot_camera_keys": list(cameras) if isinstance(cameras, dict) else None,
            "aggregate_fn_name": getattr(cfg, "aggregate_fn_name", None),
            "actions_per_chunk": getattr(cfg, "actions_per_chunk", None),
            "chunk_size_threshold": getattr(cfg, "chunk_size_threshold", None),
            "fps": getattr(cfg, "fps", None),
            "server_address": getattr(client, "server_address", None),
        },
    )
    try:
        recorder.start_state_stream(client.robot, float(getattr(cfg, "debug_record_state_hz", 0.0)))
    except BaseException:
        recorder.close()
        raise
    return recorder
