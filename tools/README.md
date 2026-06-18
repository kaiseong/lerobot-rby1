# Standalone LeRobot Dataset Tools

These tools live outside `src/lerobot/**` so they can be copied beside a fresh
LeRobot clone without patching upstream source files.

## Logical Stationary Trim

```bash
./dataset_tools.sh trim \
  --repo-id rainbowrobotics/example \
  --new-repo-id rainbowrobotics/example_trimmed \
  --workers auto \
  --dry-run
```

The trim tool is logical-only: it rewrites dataset rows and episode metadata, but
copies whole video files. It does not physically trim, decode, or re-encode MP4s.
If `--root` is omitted, LeRobot resolves the input from its cache or the Hub. If
`--new-root` is omitted, the output defaults to
`$HF_LEROBOT_HOME/<new-repo-id>`.

## Merge

```bash
./dataset_tools.sh merge \
  --source rainbowrobotics/a \
  --source rainbowrobotics/b \
  --source rainbowrobotics/c \
  --new-repo-id rainbowrobotics/merged \
  --dry-run
```

The merge tool validates compatibility first and defaults to `--remux-policy
never`, so videos are copied to new files instead of concatenated or re-encoded.
Use `--source repo_id=/local/path` only when a source dataset is outside the
default LeRobot cache.

Hub upload is disabled unless `--push-to-hub` is explicitly provided.
