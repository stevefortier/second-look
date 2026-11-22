# Launch receipt — 010

Launched once at **2026-11-22 04:40:54 UTC** on the authorized `ubuntu@68.209.107.253` RTX A6000. Both hosts passed preflight. Local and remote STOP absent, previous worker reported `exit 0 complete`, lock free, no compute clients, output root freshly created. No 009 outcome artifacts accessed. Strict host checking with project known-hosts and project key only.

From `/home/ubuntu/second-look/work/q38-code`: exclusive `mkdir work/overnight-010`; `setsid flock -n -E 75 work/overnight.lock bash scripts/q38-010.sh </dev/null >work/overnight-010/log 2>&1 &`, leader recorded in `work/overnight-010/leader.pid`. Same session/parent lock proof, STOP checks, exclusive claim, GPU gates, bounded stage timeouts and no retries as 009.

Read-back **04:41:05 UTC**: `before`; log and manifest present. Leader/SID/PGID **43087** (PPID 1), worker **43093**, timeout **43116**, pinned `/home/ubuntu/second-look/.venv-q38/bin/python` **43117** sole GPU client (17006 MiB). `/proc/43087/fdinfo` confirmed advisory write flock; independent nonblocking acquisition returned 75. No behavioral result is claimed. Do not relaunch; next check remote STOP and status. After completion checksum-back up before blind review.

Local test failed before implementation, passed after. Local/remote bash syntax and pinned bank/ordered-schedule checks passed. Base `Qwen/Qwen3.8-27B` rev `37561816e66b5b9153b1b64c10968c18c1e6f4a0`. See README for fit-only proxy provenance, recipe and endpoint. Verified SHA-256:

```
e87c8eb796a3fa4dd0091d13600d5b21344784d27ff11d8acdb7005282b5725c  fixed 008 bank (validator digest)
c07c9dc7f97150e3ef76d1cbba9a201e7a897b86bc32913a3e2d6660ffac56f3  ordered 010 schedule (manifest data_sha256)
d302c4fd4993cb14e490ee3d32321a7676a9b6d807b111036d8a99a1050b43bb  remote work/overnight-010/manifest.json
5afa66d19741c799e1f80d70b0bb91178f90ccd092eb1c1c4dc507f4b0b23a5a  studies/overnight/010/focused.json
3a48d2056b0b15eabeae4e3a28fff42ff886f7e3334cc0ca7e4d8b5bd92067dd  studies/overnight/010/README.md
8f435cdd158bc259143cb649f476903f7fb0474a006c08d71283af8092b68b42  studies/overnight/010/schedule.sha256
2186c365e82e7870717d8d5e14d52059091e07b12b6d42e50569e6059478a9b2  scripts/q38_quality010.py
66bf9bc9ac2ecf2d4c78b363d7a891bac0627f4404c6357d3e7f8f12b81870ff  scripts/q38-010.sh
0db6450abb91ad41b1aa91bae153eb5059f7eeb4b72cacf22231a50fa1699791  scripts/q38_quality.py
0a1177467a0f7860ac4cd65a739029ffd45156fac48b3f9f984c08a6fe655c96  scripts/q38_quality007.py
5d36865421ff5e25474eb0c7378549b477727c45bca2324c87937321825d438b  studies/overnight/008/protocol.json
e05d590531a617c56fa86c4ee549f9aa4df7427958a2a1051a5dfd12e6f759d0  studies/overnight/008/targets.json
ecf99e15c17bf5b4852e361abdccfe92d1140c949ab22e3067c657884864b9d8  remote scripts/train_q38.py (unchanged from 009 launch receipt)
```
