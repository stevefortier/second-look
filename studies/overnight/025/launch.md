# 025 launch receipt — November 22, 2026 UTC

11:31:33: strict project SSH identity/known-host pin to ubuntu@68.209.107.253; fresh 025 paths, remote STOP absent, project flock available, NVIDIA RTX A6000 with no compute PID. Local STOP absent immediately before submission. Prior exit confirmed by caller; no 024 status, outcomes or key opened. Only six new 025 inputs uploaded without replacement to `/home/ubuntu/second-look/work/q38-code`.

All remote artifacts passed `sha256sum --strict -c`; manifest `9c54b8315450bd9674685cc89044e0c08a53bd2f5ec88cdca33378118b2866cb`. Local and remote schedule `dd704c3d34090d5905c57a3946e19e1a8f83f80dcb9a71130f7e39d774d6404e`, 80 steps, eight groups of ten. Local pytest excluding 024: 80 tests and 19 subtests passed; worker shell syntax passed.

11:31:55: one detached submission. 11:32:07: fresh claim present, remote STOP absent, `work/overnight-025/status` = `train`. Kernel FLOCK WRITE held by PID/PGID/SID 103149, PPID 1. Worker 103150, foreground 45-minute timeout 103174, trainer 103175 in that group. Sole GPU compute client 103175 used 17022 MiB. This verifies launch, not completion. No retry. Monitor status; checksum-back up and blind-code after exit 0 per README. No historical-final access, commit, push or STATUS edit.
