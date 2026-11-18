# Syscall Markov HIDS

[![CI](https://img.shields.io/github/actions/workflow/status/laithfarhat-cell/Syscall-Tracer-Markov-Chain-Anomaly-Detector/ci.yml?label=CI)](https://github.com/laithfarhat-cell/Syscall-Tracer-Markov-Chain-Anomaly-Detector/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Linux live / cross-platform replay](https://img.shields.io/badge/live-Linux%20%7C%20replay%20anywhere-246A8D)

A host-based intrusion detection sensor that learns how one application normally uses Linux syscalls. It scores each thread's recent behavior in **bits per transition** and produces auditable alerts when the sequence drifts or a security-sensitive transition is rare. An alert is evidence for investigation, not proof of an exploit. The sensor does not block, remediate, or execute attacks.

| Signal | Detection rule |
| --- | --- |
| **Sequence drift** | Mean negative log-likelihood over the last `W` calls exceeds a percentile learned from held-out benign windows. |
| **Critical transition** | The newest transition involving a configured sensitive token, or completing a nearby `connect` + `dup2`/`dup3` pair, is unseen or at/below the configured probability cutoff. Works during window warm-up. |

```text
  Linux strace (ptrace)                 Saved JSONL / strace / auditd
          |                                            |
          +--------------> normalize + demultiplex <---+
                                       |
                +----------------------+----------------------+
                |                                             |
      benign training + validation                       observed workload
                |                                             |
      sparse Markov counts + smoothing                        |
                |                                             |
      percentile calibration -> profile.json -> per-stream rolling scorer
                                                              |
                                +-----------------------------+-----------+
                                |                             |           |
                         Rich telemetry                 JSON alerts   PNG + DOT
```

## Quick start (no privileges needed)

Python 3.11+ is required. Replay, calibration, testing, and plots work without Linux or ptrace. Run from the checkout:

```bash
python -m venv .venv
source .venv/bin/activate                # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
syscall-hids simulate --output-dir artifacts/demo
```

Open `artifacts/demo/anomaly_timeline.png` for the surprise spike, `markov_heatmap.png` for the learned probabilities, and `syscall_alerts.json` for the exact evidence. `scenario.json` marks the synthetic segment. The lab inserts **only normalized records** into a seeded, file-serving worker trace. It does not run a daemon, shell, network connection, or payload. To generate just the input traces, run `syscall-hids-generate --output-dir artifacts/inputs` or `python -m scripts.simulate_behavior`.

`python main.py` can replace `syscall-hids` in the following commands. `requirements.txt` is the runtime dependency list; `.[dev]` also installs pytest, Ruff, and the build frontend.

## CLI

### 1. Capture what you own

Live collection uses **Linux `strace`**, which uses ptrace and resolves syscall names itself on x86-64 and arm64. Spawned targets require strace 6.6+ (`--kill-on-exit`); attaching does not use that option. Use a noninteractive command or attach to an existing process. A private FIFO keeps trace data separate from target stdout/stderr; no shell interprets the command passed to the CLI.

```bash
# Spawn an ordinary application capture. Repeat for representative benign sessions.
syscall-hids trace -o artifacts/captures/run-01.jsonl -- python -c 'print("benign capture")'

# Or attach to an application you are authorized to trace:
syscall-hids trace --pid 12345 --syscalls openat,read,write,close \
  -o artifacts/captures/service.jsonl
```

`--pid` and `-- PROGRAM [ARGS...]` are mutually exclusive. `--syscalls` defaults to `all`; use the **same filter** for training and monitoring. Threads and children are followed by default. `--no-follow` limits capture to the selected task. Ctrl-C detaches from an attached target; if the sensor spawned the workload, it stops that tracing session and its tracees. Ctrl-C or SIGTERM retains a nonempty partial capture. A nonzero target or tracer exit is reported as an error rather than publishing a potentially incomplete new trace. A call is emitted once strace has produced its complete record; blocked calls may appear only on return.

Live permissions are controlled by Linux UID/dumpability, ptrace policy (e.g. Yama), and container isolation. Prefer tracing your own spawned workload. If attach is denied, use an authorized recording or an approved deployment policy; the CLI never changes host security settings.

### 2. Train and calibrate

```bash
syscall-hids train -i 'artifacts/demo/benign_*.jsonl' \
  --validation 'artifacts/demo/validation_*.jsonl' \
  --order 1 --alpha 0.01 --window 20 --model artifacts/model/profile.json
```

- Supply **independent benign captures** for `--validation` when possible. Without it, the last fifth of input files (rounded up) is held out; with one file, its final fifth of events is held out. File boundaries and the split never produce transitions. Literal glob patterns expand in sorted order; explicit filenames retain argument order. At least one validation stream must contain `W` calls.
- `--order` selects a first- or higher-order model (1–16); use `order < window ≤ 8192`. Use representative workload phases and enough independent sessions before increasing the order. `--percentile` defaults to 99; the saved calibration contains the threshold, sample count, median, and median absolute deviation.
- Increment trusted counts with `train --update previous-profile.json -i NEW_BENIGN ... --validation HELD_OUT ... --model profile.json`. Updates cannot change order or smoothing, and always recalibrate; never feed unreviewed monitor traffic back into the baseline.

### 3. Monitor

```bash
# Replay a recording offline:
syscall-hids monitor --model artifacts/model/profile.json \
  --input artifacts/demo/simulated_anomaly.jsonl --plot \
  --alerts artifacts/replay/syscall_alerts.json

# Live, same configuration and scope as training:
syscall-hids monitor --model artifacts/model/profile.json --input live --pid 12345 \
  --alerts artifacts/live/syscall_alerts.jsonl
# Or replace --pid 12345 with: -- ./your-approved-application
```

`--plot` writes `markov_heatmap.png`, `anomaly_timeline.png`, and `markov_graph.dot` next to the alerts unless `--plot-dir` is set. If Graphviz is installed, render the graph with `dot -Tsvg artifacts/replay/markov_graph.dot -o artifacts/replay/markov_graph.svg`. Blue edges are observed baseline transitions, red edges are evidence (the newest critical edge or the most surprising edge in a drift window). For higher orders, a graph node is a syscall context. Visuals cap displayed contexts/edges/markers and keep peak scores in timeline buckets for long traces; this does not truncate SIEM evidence.

The monitor uses the saved window and threshold. Changing `--window` requires a corresponding calibration or an explicit `--threshold` in bits. In a terminal it shows a live transition, the score vs. threshold, and recent alerts; `--quiet` suppresses the dashboard, not the summary or JSON. Exit codes: `0` completed (even if alerts were found), `1` runtime/input failure, `2` invalid CLI syntax, `130` interrupted (Ctrl-C), `143` terminated (SIGTERM on POSIX).

## Input and output contracts

| Input | Interpretation |
| --- | --- |
| `.jsonl` / `.ndjson` | UTF-8, one object per line: `{"token":"openat:READ","pid":2345,"timestamp":1750000000.01,"source":"worker-a"}`. Only `token` is required. Optional positive `stream_id` distinguishes reused task IDs. Use `syscall` instead of `token` and optional `args` (strings or integers in argument order) to request normalization. Input order wins; timestamps are optional Unix seconds. |
| `.audit`, `.auditd`, `audit.log` | Raw Linux audit `type=SYSCALL` records; auxiliary records are ignored. Native x86-64 and arm64 numbers are resolved using bundled Linux ABI facts, not the analysis host. Unknown ABI/number pairs remain distinct numeric tokens. Audit argument registers are hex; a pointer such as audit `openat2`'s `open_how` cannot be decoded offline. |
| Other extensions | `strace` text with optional PID/TID and timestamps (Unix `-ttt` recommended). Unfinished/resumed calls are matched per TID, and signal/exit banners are not syscalls. Clock-of-day prefixes are accepted without assigning a date. |

Override detection with `--format strace`, `--format auditd` (e.g. rotated `audit.log.1`), or `--format jsonl`. Normalization drops file names, addresses, descriptors, buffer contents, and other raw arguments from stored event records. It preserves read-like vs. write-intent opens; executable protection on `mprotect`, `pkey_mprotect`, and `mmap`; and `socket` families (including IPv4 and IPv6 under `INET`). If context is missing, the default is a bare syscall name, not a fabricated argument classification.

Each `(source, pid, stream_id)` is scored independently within a recording. `pid` is the ID reported by the collector (usually a Linux TID in a followed strace log). Strace starts a new `stream_id` after a task exit. Other producers can supply distinct stream IDs or separate files when PID reuse occurs. An anonymous stream must not contain multiplexed tasks. Use separate capture files for unrelated runs; don't concatenate unrelated anonymous streams. The configurable `max_streams` limit fails explicitly instead of silently merging or evicting streams.

**Alerts.** `syscall_alerts.json` is a JSON array finalized on controlled shutdown (including Ctrl-C or SIGTERM). Choose `.jsonl` or `.ndjson` to ingest individual alerts as they arrive. Each record has ECS-style `event`, `rule`, optional `process.pid` and `@timestamp` (from the observation, omitted if unavailable), plus a `syscall_hids` object with trigger names, heuristic severity (0–100), threshold, and evidence. Indices are 1-based within the input and within the stream. The evidence includes the normalized calls and the probabilities, observed counts, and surprises of all evaluated transitions within the indicated window. A critical warm-up alert has a shorter window and `score_bits: null`. Overlapping windows can give several alerts for one incident; group them in the downstream system. Prefer dedicated, access-controlled directories for profiles, traces, and alerts.

## Configuration and containers

```bash
syscall-hids --config config.example.yaml monitor \
  --model artifacts/model/profile.json --input artifacts/demo/simulated_anomaly.jsonl
```

The example YAML shows all supported keys. CLI overrides take precedence where a CLI option exists. The default critical cutoff is `0.001`; an unseen transition counts as rare regardless of its smoothed probability. The `connect`/descriptor-duplication check considers either ordering in the same stream within `critical_pair_span` calls, capped by `W-1`; set the span to `0` to turn it off. Use a workload-specific critical catalog where needed.

```bash
docker build -t syscall-hids .
mkdir -p artifacts
# Offline: no ptrace capability or host PID namespace required.
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/artifacts:/work" syscall-hids \
  simulate --output-dir /work/demo
# Live: only grant extra tracing capability if approved and required by the runtime.
docker run --rm --cap-add=SYS_PTRACE --user "$(id -u):$(id -g)" \
  -v "$PWD/artifacts:/work" syscall-hids trace -o /work/benign.jsonl -- /bin/true
```

The image defaults to a non-root user; the example maps file ownership to the invoking host user. Capabilities granted to a container may not be effective for a non-root UID; use the runtime's approved UID/capability and seccomp policy if attachment is needed. The target must be visible in the container's PID namespace; prefer a dedicated workload container rather than exposing the host PID namespace. Do not use `--privileged` for offline use, and do not disable host-wide tracing protections to run the demo.

## Validation and deployment limits

```bash
python -m pytest                      # offline suite plus live tests when supported
python -m pytest -m 'not live'        # omit tests gated on Linux strace / ptrace
python -m ruff check .
python -m ruff format --check .
python -m build
```

CI exercises Linux and Windows replay, checks live capture where ptrace is available, and uploads a generated inert lab as a GitHub Actions artifact. The simulator checks behavior on a controlled trace, not real-attack precision or recall. Use separate profiles by workload, release, architecture, and capture scope. Hold out enough real benign sessions to measure false alarms before rollout. Strace perturbs scheduling and is best scoped to selected processes; audit replay is only as complete as the audit policy. Lost or selectively captured syscalls change transition meaning. Exact percentile calibration retains one numeric score per complete validation window; keep calibration sets appropriately bounded. Runtime history is bounded per stream; monitoring without `--plot` does not retain a timeline. See [THEORY.md](THEORY.md) for the likelihood derivation, calibration caveats, and limitations against mimicry.
