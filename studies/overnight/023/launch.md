# 023 launch receipt — 2026-11-22 UTC

Launched once at 10:50:59 on `ubuntu@68.209.107.253`, `/home/ubuntu/second-look/work/q38-code`, using the project identity and strict project known-hosts file. Local and remote `work/STOP-OVERNIGHT` absent; previous status `exit 0 complete`; output absent, project lock available and RTX A6000 idle before launch. Only new 023 inputs transferred. Fresh run directory and worker `.claimed` guard. No historical-final data used.

Local suite: 79 tests and 19 subtests passed. Remote verified all 43 SHA-256 entries and the manifest digest `51d679bb03bdcd97b3f4573c0b7c8fb79bc41f06bc78e37c7d485bc10a44e96a`. Local and remote schedule checks agreed: `dcbe8745414c1c7c79c8497dfb9c3468e62d0703c29061e8d11eaa373c629435`, 80 steps, seed 38127, eight groups × ten steps. For reproduction, trainer bytes are from `591fd2a50c9aae1330830ca4f982135fb7e059fd`, not the mutable working trainer.

Readback at 10:51:15 UTC: `work/overnight-023/status` = `train`; kernel FLOCK WRITE on `work/overnight.lock` owned by detached PID/PGID/SID 96708. Worker 96710, 45-minute timeout 96759, trainer 96760 all in that group/session. PID 96760 was the sole GPU compute client (17068 MiB). This verifies launch, not completion or a behavioral result.

Next: check local STOP and remote status; do not relaunch or remove the claim. On `exit 0 complete`, checksum-back up and freeze blind judgments before joining or scoring, per README.md. Remote log: `work/overnight-023/runner.log`.
