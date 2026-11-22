# 024 launch receipt — November 22, 2026 (UTC)

Launched once at 11:11:57 on `ubuntu@68.209.107.253`, root `/home/ubuntu/second-look/work/q38-code`, using the designated project key and strict project known-hosts file. Local and remote STOP guards clear on entry and immediately before launch. Prior exit marker `exit 0 complete`; project flock available, fresh output, A6000 idle. Only 024 inputs transferred; retained pinned trainer (source `591fd2a50c9aae1330830ca4f982135fb7e059fd`) and existing remote environment. No 023 outcomes or keys inspected.

Local suite: 80 passed, 19 subtests passed. Local/remote schedule checks agreed: `0a0b82ccb54dd7e6c7db8c15f4aa358af419e353329d2265dce66ae9b5dfa0d5`, 80 steps, seed 38127, eight groups × ten. Eight explicit targets, one per image and axis; eight matched captions restored. Remote verified all 48 manifest entries; manifest SHA-256 `0070d6f0b6c8b9ed48c7cabb8022f2ea33bc4c6102d674df312baa66ffe632fa`.

At 11:12:08, remote `work/overnight-024/status` read `train`. Detached flock PID/PGID/SID `100031`, PPID 1; kernel fdinfo confirmed exclusive FLOCK WRITE on `work/overnight.lock`. Worker `100032`, 45-minute stage timeout `100057`, trainer `100058` in same group and session. Trainer was sole GPU compute client on RTX A6000 `GPU-f40b0708-a493-ded5-fba6-c520d2813fcc` (17000 MiB). Launch verified, not completion.

Monitor `work/overnight-024/status` and `work/overnight-024/runner.log` after checking STOP. Never relaunch or remove `.claimed`. After successful exit, checksum-back up and freeze blind judgments before joining.
