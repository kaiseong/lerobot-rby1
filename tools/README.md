# Standalone LeRobot Dataset Tools

These tools live outside `src/lerobot/**` so they can be copied beside a fresh
LeRobot clone without patching upstream source files.

## Logical Stationary Trim

```bash
./dataset_tools.sh trim \
  --repo-id rainbowrobotics/example \
  --root /path/to/example \
  --new-repo-id rainbowrobotics/example_trimmed \
  --new-root /path/to/example_trimmed \
  --workers auto \
  --dry-run
```

The trim tool is logical-only: it rewrites dataset rows and episode metadata, but
copies whole video files. It does not physically trim, decode, or re-encode MP4s.

## Merge

```bash
./dataset_tools.sh merge \
  --source rainbowrobotics/a=/path/to/a \
  --source rainbowrobotics/b=/path/to/b \
  --source rainbowrobotics/c=/path/to/c \
  --new-repo-id rainbowrobotics/merged \
  --new-root /path/to/merged \
  --dry-run
```

The merge tool validates compatibility first and defaults to `--remux-policy
never`, so videos are copied to new files instead of concatenated or re-encoded.

Hub upload is disabled unless `--push-to-hub` is explicitly provided.
