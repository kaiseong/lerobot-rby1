"""Offline inspector for sessions written by ``infer_recorder.InferenceRecorder``.

    python -m lerobot.async_inference.inspect_infer_record <session_dir> [--video]

Writes ``summary.md`` and PNG plots (matplotlib, Agg) into the session folder.
``--video`` also stitches the sent JPEGs into one mp4 per camera.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .infer_recorder import TICK_BLOCKED, TICK_EMPTY, TICK_EXECUTED

RAGGED_KEYS = {"raw_offsets", "raw_shapes", "raw_actions", "conv_offsets", "conv_timesteps", "conv_actions"}
# Events after which queued actions were dropped (chunks are not drawn past them).
QUEUE_CLEAR_EVENTS = {"pause", "resume", "queue_cleared", "server_connected"}


def _load_parts(folder: Path) -> list[dict[str, np.ndarray]]:
    parts = []
    for path in sorted(folder.glob("part_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                parts.append({k: data[k] for k in data.files})
        except Exception as exc:  # noqa: BLE001 - e.g. a 0-byte part after a power loss
            print(f"warning: skipping unreadable {path}: {exc!r}")
    return parts


def _concat(parts: list[dict[str, np.ndarray]], skip: set[str] = frozenset()) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    keys = [k for k in parts[0] if k not in skip]
    return {k: np.concatenate([p[k] for p in parts if k in p]) for k in keys}


def _load_requests(folder: Path) -> dict[str, Any]:
    parts = _load_parts(folder)
    out: dict[str, Any] = _concat(parts, skip=RAGGED_KEYS)
    raw, chunks = [], []
    for p in parts:
        for i in range(len(p["req_id"])):
            a, b = p["raw_offsets"][i], p["raw_offsets"][i + 1]
            flat = p["raw_actions"][a:b]
            shape = tuple(p["raw_shapes"][i])
            raw.append(flat.reshape(shape) if flat.size == shape[0] * shape[1] else flat)
            a, b = p["conv_offsets"][i], p["conv_offsets"][i + 1]
            chunks.append((p["conv_timesteps"][a:b], p["conv_actions"][a:b]))
    out["raw"] = raw
    out["chunks"] = chunks
    return out


def load_infer_record(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    events = []
    events_path = session / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A crash can leave a truncated last line.
    # Failure events are stamped at the failure time, so file order is not time order.
    events.sort(key=lambda ev: ev.get("t", 0.0))
    stats_path = session / "stats.json"
    return {
        "meta": json.loads((session / "meta.json").read_text(encoding="utf-8")),
        "stats": json.loads(stats_path.read_text()) if stats_path.exists() else None,
        "ticks": _concat(_load_parts(session / "ticks")),
        "obs": _concat(_load_parts(session / "obs")),
        "requests": _load_requests(session / "requests"),
        "state": _concat(_load_parts(session / "state")),
        "events": events,
    }


def _pct(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else math.nan


def _fmt(value: float, digits: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _stats_row(name: str, values: np.ndarray, unit: str) -> str:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return f"| {name} | 0 | n/a | n/a | n/a | n/a | {unit} |"
    return (
        f"| {name} | {values.size} | {_fmt(values.mean())} | {_fmt(_pct(values, 50))} | "
        f"{_fmt(_pct(values, 90))} | {_fmt(values.max())} | {unit} |"
    )


def loop_metrics(ticks: dict[str, np.ndarray], dt: float) -> dict[str, float]:
    if not ticks or len(ticks["t_loop"]) < 2:
        return {}
    period = np.diff(ticks["t_loop"]) * 1000
    work = (ticks["t_end"] - ticks["t_loop"]) * 1000
    executed = ticks["status"] == TICK_EXECUTED
    # Send jitter only between back-to-back executed ticks; gaps (starved/paused) are reported separately.
    pairs = executed[1:] & executed[:-1]
    send_dt = (np.diff(ticks["t_action"]) * 1000)[pairs]
    t_exec = ticks["t_action"][executed]
    gap = np.diff(t_exec) * 1000 if t_exec.size > 1 else np.zeros(0)
    first = int(np.argmax(executed)) if executed.any() else len(executed)
    after_start = ticks["status"][first:]
    return {
        "ticks": float(len(ticks["t_loop"])),
        "executed": float(executed.sum()),
        "starved_after_first_action": float((after_start == TICK_EMPTY).sum()),
        "empty_before_first_action": float((ticks["status"][:first] == TICK_EMPTY).sum()),
        "blocked": float((ticks["status"] == TICK_BLOCKED).sum()),
        "period_mean_ms": float(period.mean()),
        "period_p99_ms": _pct(period, 99),
        "period_max_ms": float(period.max()),
        "overruns": float((period > 1.5 * dt * 1000).sum()),
        "work_p99_ms": _pct(work, 99),
        "work_max_ms": float(np.nanmax(work)),
        "send_interval_std_ms": float(send_dt.std()) if send_dt.size else math.nan,
        "send_interval_p99_ms": _pct(send_dt, 99),
        "send_interval_max_ms": float(send_dt.max()) if send_dt.size else math.nan,
        "max_action_gap_ms": float(gap.max()) if gap.size else math.nan,
    }


def write_summary(rec: dict[str, Any], out_path: Path) -> str:
    meta, ticks, obs, req, stats = rec["meta"], rec["ticks"], rec["obs"], rec["requests"], rec["stats"]
    dt = float(meta.get("environment_dt") or 1 / 30)
    lines = [f"# Inference record {out_path.parent.name}", ""]
    lines += [
        f"- created: {meta.get('created')} on {meta.get('host')}",
        f"- note: {meta.get('note') or '-'}",
        f"- git: {meta.get('git')}",
        f"- aggregate: {meta.get('aggregate_fn_name')}, actions_per_chunk {meta.get('actions_per_chunk')}, "
        f"chunk_size_threshold {meta.get('chunk_size_threshold')}, fps {meta.get('fps')}",
    ]
    for key, task in (meta.get("tasks") or {}).items():
        lines.append(f"- task {key} ({task.get('label')}): {task.get('task')!r}")
    if ticks:
        lines.append(f"- duration: {ticks['t_loop'][-1] - ticks['t_loop'][0]:.1f} s")

    lines += ["", "## Counts", ""]
    n_req = len(req.get("req_id", []))
    status = req.get("status", np.zeros(0, dtype=str))
    status_counts = {s: int((status == s).sum()) for s in sorted(set(status.tolist()))}
    lines += [
        f"- ticks: {len(ticks.get('t_loop', []))}",
        _obs_line(obs, req),
        f"- requests: {n_req} {status_counts}",
        f"- state samples: {len(rec['state'].get('t', []))}",
        f"- images: {stats['counts']['image'] if stats else 'n/a'}",
    ]
    if stats:
        drops = {k: v for k, v in stats["drops"].items() if v}
        lines += [
            f"- recorder drops: {drops or 0}, hook errors {stats['hook_errors']}, "
            f"writer errors {stats['writer_errors']}",
            f"- recorder writer CPU: {stats['writer_cpu_s']:.2f} s over {stats['duration_s']:.1f} s, "
            f"{stats['bytes_written'] / 1e6:.1f} MB written; whole-process CPU "
            f"{_fmt(stats.get('process_cpu_s', math.nan), 2)} s; "
            f"image sets skipped at close: {stats.get('images_dropped_on_close', 0)}",
        ]

    lines += ["", "## Timing", "", "| metric | n | mean | p50 | p90 | max | unit |", "|---|---|---|---|---|---|---|"]
    if n_req:
        ok = status == "accepted"
        request_ms = req["request_ms"]
        lines.append(_stats_row("request (accepted)", request_ms[ok], "ms"))
        lines.append(_stats_row("server infer (accepted)", req["infer_ms"][ok], "ms"))
        lines.append(_stats_row("network/overhead (accepted, request - infer)",
                                (request_ms - req["infer_ms"])[ok], "ms"))
        if (~ok).any():
            lines.append(_stats_row("request (discarded/failed, until failure)", request_ms[~ok], "ms"))
        lines.append(_stats_row("obs capture->send", (req["t_send"] - _capture_time(obs, req)) * 1000, "ms"))
    if obs:
        lines.append(_stats_row("observation capture", (obs["t_capture_end"] - obs["t_capture_start"]) * 1000, "ms"))
    if ticks:
        lines.append(_stats_row("loop period", np.diff(ticks["t_loop"]) * 1000, "ms"))
        lines.append(_stats_row("loop work", (ticks["t_end"] - ticks["t_loop"]) * 1000, "ms"))
        m = loop_metrics(ticks, dt)
    else:
        m = {}
    if m:
        lines += [
            "",
            f"- loop period mean {_fmt(m['period_mean_ms'], 2)} ms, p99 {_fmt(m['period_p99_ms'], 2)} ms, "
            f"max {_fmt(m['period_max_ms'], 2)} ms; overruns (>{1.5 * dt * 1000:.1f} ms): {int(m['overruns'])}",
            f"- executed ticks {int(m['executed'])}; starved (queue empty) after first action: "
            f"{int(m['starved_after_first_action'])}; "
            f"empty before first action: {int(m['empty_before_first_action'])}; "
            f"blocked (paused/ready pose): {int(m['blocked'])}",
            f"- action send interval (back-to-back executed ticks) std {_fmt(m['send_interval_std_ms'], 2)} ms, "
            f"p99 {_fmt(m['send_interval_p99_ms'], 2)} ms, max {_fmt(m['send_interval_max_ms'], 2)} ms; "
            f"longest gap between executed actions {_fmt(m['max_action_gap_ms'], 1)} ms",
        ]
    elif ticks:
        lines += ["", "- not enough ticks for loop metrics"]
    if n_req:
        accepted = status == "accepted"
        if accepted.any():
            arrival_queue = req["queue_before"][accepted]
            lines.append(
                f"- queue size at chunk arrival: min {int(arrival_queue.min())}, "
                f"mean {_fmt(float(arrival_queue.mean()))}"
            )

    errors = [(int(r), e) for r, e, s in zip(req.get("req_id", []), req.get("error", []), status, strict=False) if e]
    if errors:
        lines += ["", "## Request errors", ""] + [f"- req {r}: {e}" for r, e in errors[:50]]

    lines += ["", "## Events", ""]
    t0 = float(ticks["t_loop"][0]) if ticks else rec["meta"]["clock"]["monotonic"]
    for ev in rec["events"]:
        fields = {k: v for k, v in ev.items() if k not in ("t", "event")}
        lines.append(f"- {ev['t'] - t0:8.2f} s  {ev['event']} {fields if fields else ''}".rstrip())
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def _obs_line(obs: dict[str, np.ndarray], req: dict[str, Any]) -> str:
    """Observations captured, sent, and why the rest were never sent."""
    if not obs:
        return "- observations queued: 0"
    sent = np.isin(obs["obs_ts"], req.get("obs_ts", np.zeros(0)))
    replaced = np.isin(obs["obs_ts"], obs["replaced_ts"][np.isfinite(obs["replaced_ts"])]) & ~sent
    other = ~sent & ~replaced
    return (
        f"- observations queued: {len(sent)}, sent: {int(sent.sum())}, never sent: {int((~sent).sum())} "
        f"(replaced by a newer one {int(replaced.sum())}, drained on pause/reconnect/stop {int(other.sum())})"
    )


def _capture_time(obs: dict[str, np.ndarray], req: dict[str, Any]) -> np.ndarray:
    """Capture start of the observation each request sent (matched on wall timestamp)."""
    if not obs:
        return np.full(len(req["req_id"]), np.nan)
    order = np.argsort(obs["obs_ts"])
    idx = np.searchsorted(obs["obs_ts"][order], req["obs_ts"])
    idx = np.clip(idx, 0, len(order) - 1)
    match = order[idx]
    t = obs["t_capture_start"][match].copy()
    t[obs["obs_ts"][match] != req["obs_ts"]] = np.nan
    return t


def _chunk_segments(rec: dict[str, Any], col: int, scale) -> list[np.ndarray]:
    """Accepted chunks as (time, value) lines, cut where the action queue was cleared.

    Timesteps run monotonically between queue clears, so inside one segment a
    chunk timestep maps to the time the matching action was executed; the
    unexecuted tail is extrapolated at dt and never drawn past the next clear.
    """
    ticks, req, meta = rec["ticks"], rec["requests"], rec["meta"]
    dt = float(meta.get("environment_dt") or 1 / 30)
    executed = ticks["status"] == TICK_EXECUTED
    t_exec, ts_exec = ticks["t_action"][executed], ticks["timestep"][executed].astype(np.float64)
    clears = np.asarray(sorted(ev["t"] for ev in rec["events"] if ev.get("event") in QUEUE_CLEAR_EVENTS))
    segments = []
    for i, (steps, actions) in enumerate(req.get("chunks", [])):
        if req["status"][i] != "accepted" or len(steps) < 2:
            continue
        t_recv = req["t_recv"][i]
        k = np.searchsorted(clears, t_recv)
        start = clears[k - 1] if k > 0 else -np.inf
        end = clears[k] if k < len(clears) else np.inf
        sel = (t_exec >= start) & (t_exec < end)
        if not sel.any():
            continue
        ts, tt = ts_exec[sel], t_exec[sel]
        steps = np.asarray(steps, dtype=np.float64)
        t = np.interp(steps, ts, tt)
        t = np.where(steps < ts[0], tt[0] + (steps - ts[0]) * dt, t)
        t = np.where(steps > ts[-1], np.maximum(tt[-1], t_recv) + (steps - ts[-1]) * dt, t)
        keep = t < end
        if keep.sum() > 1:
            segments.append(np.column_stack([t[keep], scale(actions[keep, col])]))
    return segments


def _executed_line(ticks: dict[str, np.ndarray], col: int) -> tuple[np.ndarray, np.ndarray]:
    """Executed action per tick; NaN on starved/blocked ticks so the line breaks there."""
    executed = ticks["status"] == TICK_EXECUTED
    t = np.where(executed, ticks["t_action"], ticks["t_loop"])
    return t, np.where(executed, ticks["action"][:, col], np.nan)


def plot_session(rec: dict[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    meta, ticks, req, state, obs = rec["meta"], rec["ticks"], rec["requests"], rec["state"], rec["obs"]
    names = meta["action_names"]
    dt = float(meta.get("environment_dt") or 1 / 30)
    written: list[Path] = []
    if not ticks:
        return written
    t0 = float(ticks["t_loop"][0])

    indices = (meta.get("state_stream") or {}).get("model_indices", {})
    has_state = bool(state) and "position" in state
    groups = [("right_arm", "right_arm_idx"), ("left_arm", "left_arm_idx")]
    for group, idx_key in groups:
        cols = [i for i, n in enumerate(names) if n.startswith(group)]
        if not cols:
            continue
        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.2 * len(cols)), sharex=True, squeeze=False)
        sdk_idx = indices.get(idx_key)
        for row, col in enumerate(cols):
            ax = axes[row, 0]
            segments = [seg - (t0, 0) for seg in _chunk_segments(rec, col, np.degrees)]
            if segments:
                ax.add_collection(LineCollection(segments, colors="tab:orange", linewidths=0.6, alpha=0.5))
            if has_state and sdk_idx and len(sdk_idx) == len(cols):
                ax.plot(state["t"] - t0, np.degrees(state["position"][:, sdk_idx[row]]), color="tab:blue", lw=0.8)
            elif obs:
                ax.plot(obs["t_capture_end"] - t0, np.degrees(obs["joints"][:, col]), color="tab:blue", lw=0.8)
            t_exec, a_exec = _executed_line(ticks, col)
            ax.plot(t_exec - t0, np.degrees(a_exec), color="black", lw=0.9)
            ax.set_ylabel(f"{names[col]}\n[deg]", fontsize=8)
            ax.grid(alpha=0.3)
            ax.autoscale_view()
        axes[0, 0].set_title(
            f"{group}: executed action (black), measured {'SDK state' if has_state and sdk_idx else 'observation'} "
            "(blue), received chunks (orange)",
            fontsize=10,
        )
        axes[-1, 0].set_xlabel("time [s]")
        fig.tight_layout()
        path = out_dir / f"plot_{group}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)

    grip_cols = [i for i, n in enumerate(names) if "gripper" in n]
    if grip_cols:
        fig, axes = plt.subplots(len(grip_cols), 1, figsize=(14, 2.2 * len(grip_cols)), sharex=True, squeeze=False)
        for row, col in enumerate(grip_cols):
            ax = axes[row, 0]
            segments = [seg - (t0, 0) for seg in _chunk_segments(rec, col, lambda v: v)]
            if segments:
                ax.add_collection(LineCollection(segments, colors="tab:orange", linewidths=0.6, alpha=0.5))
            if obs:
                ax.plot(obs["t_capture_end"] - t0, obs["joints"][:, col], color="tab:blue", lw=0.8)
            t_exec, a_exec = _executed_line(ticks, col)
            ax.plot(t_exec - t0, a_exec, color="black", lw=0.9)
            ax.set_ylabel(names[col], fontsize=8)
            ax.grid(alpha=0.3)
            ax.autoscale_view()
        axes[0, 0].set_title("grippers: executed (black), observation (blue), chunks (orange)", fontsize=10)
        axes[-1, 0].set_xlabel("time [s]")
        fig.tight_layout()
        path = out_dir / "plot_grippers.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    if len(req.get("req_id", [])):
        ok = req["status"] == "accepted"
        axes[0].plot(req["t_recv"][ok] - t0, req["request_ms"][ok], ".-", lw=0.6, ms=3, label="request")
        axes[0].plot(req["t_recv"] - t0, req["infer_ms"], ".", ms=3, label="server infer")
        bad = ~ok
        if bad.any():
            t_fail = req["t_fail"][bad] if "t_fail" in req else req["t_end"][bad]
            t_bad = np.where(np.isfinite(req["t_recv"][bad]), req["t_recv"][bad], t_fail)
            axes[0].plot(t_bad - t0, np.zeros(bad.sum()), "rx", label="discarded/failed")
        axes[0].legend(fontsize=8)
    axes[0].set_ylabel("latency [ms]")
    # queue_before is -1 on ticks without a pop; leave a gap there (starved/blocked marks below).
    queue = np.where(ticks["queue_before"] >= 0, ticks["queue_before"], np.nan)
    axes[1].step(ticks["t_loop"] - t0, queue, where="post", lw=0.7, label="queue before pop")
    starved = ticks["status"] == TICK_EMPTY
    axes[1].plot(ticks["t_loop"][starved] - t0, np.zeros(starved.sum()), "r|", ms=8, label="starved tick")
    blocked = ticks["status"] == TICK_BLOCKED
    if blocked.any():
        axes[1].plot(ticks["t_loop"][blocked] - t0, np.zeros(blocked.sum()), "|", color="gray", ms=8, label="blocked")
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("action queue")
    period = np.diff(ticks["t_loop"]) * 1000
    axes[2].plot(ticks["t_loop"][1:] - t0, period, lw=0.6, label="loop period")
    axes[2].plot(ticks["t_loop"] - t0, (ticks["t_end"] - ticks["t_loop"]) * 1000, lw=0.6, label="loop work")
    axes[2].axhline(1.5 * dt * 1000, color="r", ls="--", lw=0.6, label="overrun")
    axes[2].set_ylabel("[ms]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "plot_timeline.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    written.append(path)
    return written


def write_videos(session: Path, fps: float = 5.0) -> list[Path]:
    import cv2

    written = []
    for cam_dir in sorted((session / "images").glob("*")):
        frames = sorted(cam_dir.glob("r*_t*.jpg"))
        first = next((img for img in (cv2.imread(str(f)) for f in frames) if img is not None), None)
        if first is None:
            continue
        h, w = first.shape[:2]
        path = session / f"video_{cam_dir.name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.shape[:2] != (h, w):
                continue
            req_id, timestep = frame_path.stem[1:].split("_t")
            cv2.putText(frame, f"req {int(req_id)} t {int(timestep)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--video", action="store_true", help="Stitch sent JPEGs into one mp4 per camera")
    parser.add_argument("--video_fps", type=float, default=5.0)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args(argv)

    rec = load_infer_record(args.session_dir)
    print(write_summary(rec, args.session_dir / "summary.md"))
    if not args.no_plots:
        try:
            for path in plot_session(rec, args.session_dir):
                print(f"wrote {path}")
        except ImportError as exc:
            print(f"plots skipped: {exc}")
    if args.video:
        for path in write_videos(args.session_dir, args.video_fps):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
