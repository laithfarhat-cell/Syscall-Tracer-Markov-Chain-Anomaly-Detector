"""Linux strace capture and streaming normalization of recorded syscalls."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from .common import DEFAULT_STREAM_LIMIT, StreamKey, bounded_float, bounded_int, stream_limit

MAX_LINE = 1 << 20
MAX_PID = (1 << 31) - 1
MAX_STREAM_ID = (1 << 63) - 1
MAX_TIMESTAMP = 253402300799.0  # last whole UTC second representable by datetime
_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,95}")
_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{0,95}(?::[a-zA-Z0-9_]{1,31})?")
_SOURCE = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.:@-]{0,127}")


class TraceError(ValueError):
    """An invalid recording or a failed live capture."""


def valid_token(value: object) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class SyscallEvent:
    """A normalized observation, with no raw syscall arguments or return values."""

    token: str
    pid: int | None = None
    timestamp: float | None = None
    source: str | None = None
    stream_id: int | None = None

    def __post_init__(self) -> None:
        if not valid_token(self.token):
            raise TraceError("invalid syscall token")
        if self.pid is not None:
            bounded_int(self.pid, "pid", 1, MAX_PID)
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", bounded_float(self.timestamp, "timestamp", 0, MAX_TIMESTAMP))
        if self.source is not None and (not isinstance(self.source, str) or not _SOURCE.fullmatch(self.source)):
            raise TraceError("invalid source identifier")
        if self.stream_id is not None:
            bounded_int(self.stream_id, "stream_id", 1, MAX_STREAM_ID)

    @property
    def stream(self) -> StreamKey:
        return self.source, self.pid, self.stream_id

    def as_dict(self) -> dict[str, str | int | float]:
        record: dict[str, str | int | float] = {"token": self.token}
        if self.pid is not None:
            record["pid"] = self.pid
        if self.timestamp is not None:
            record["timestamp"] = self.timestamp
        if self.source is not None:
            record["source"] = self.source
        if self.stream_id is not None:
            record["stream_id"] = self.stream_id
        return record

    def json_line(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), allow_nan=False) + "\n"


_OPEN_INDEX = {"open": 1, "openat": 2, "openat2": 2}
_PROT_INDEX = {"mprotect": 2, "pkey_mprotect": 2, "mmap": 2, "mmap2": 2}
_WRITE_FLAGS = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE", "__O_TMPFILE"}
_FAMILIES = {
    "AF_UNIX": "UNIX",
    "AF_LOCAL": "UNIX",
    "PF_UNIX": "UNIX",
    "PF_LOCAL": "UNIX",
    "AF_INET": "INET",
    "AF_INET6": "INET",
    "PF_INET": "INET",
    "PF_INET6": "INET",
    "AF_NETLINK": "NETLINK",
    "PF_NETLINK": "NETLINK",
}


# On success, return the static tag only; never keep paths, pointers or fd numbers.
def _number(value: str) -> int | None:
    value = value.strip()
    if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]{1,16}|[0-9]{1,20})", value):
        return None
    if value.lower().startswith("0x"):
        return int(value, 16)
    if len(value) > 1 and value.startswith("0") and all(ch in "01234567" for ch in value):
        return int(value, 8)
    return int(value, 10)


def _flags(value: str) -> tuple[set[str], int | None]:
    names: set[str] = set()
    bits: int | None = None
    for part in value.split("|"):
        part = part.strip()
        number = _number(part)
        if number is not None:
            bits = (bits or 0) | number
        else:
            names.add(part)
    return names, bits


def normalize_syscall(name: str, args: Sequence[str] = ()) -> str:
    """Abstract only the arguments with meaningful, bounded security categories."""
    if not valid_token(name):
        raise TraceError("invalid syscall name")
    if ":" in name:
        return name  # already normalized
    if name == "creat":
        return "creat:WRITE"
    if name in _OPEN_INDEX and len(args) > (idx := _OPEN_INDEX[name]):
        value = args[idx]
        if name == "openat2":
            flags = re.search(r"(?:^|[,{])\s*flags\s*=\s*([^,}]+)", value)
            if not flags:
                return name  # e.g. an audit record has just a pointer to struct open_how
            value = flags[1]
        names, bits = _flags(value)
        # O_ACCMODE, O_CREAT, O_TRUNC, O_APPEND, __O_TMPFILE (native Linux ABIs).
        if names & _WRITE_FLAGS or (bits is not None and (bits & 3 in (1, 2) or bits & 0x400640)):
            return f"{name}:WRITE"
        if "O_RDONLY" in names or (bits is not None and (bits & 3) == 0):
            return f"{name}:READ"
    if name in _PROT_INDEX and len(args) > (idx := _PROT_INDEX[name]):
        names, bits = _flags(args[idx])
        if "PROT_EXEC" in names or (bits is not None and bits & 0x4):
            return f"{name}:EXEC"
    if name == "socket" and args:
        family = args[0].strip()
        if family in _FAMILIES:
            return "socket:" + _FAMILIES[family]
        number = _number(family)
        if number is not None:
            return "socket:" + {1: "UNIX", 2: "INET", 10: "INET", 16: "NETLINK"}.get(number, "OTHER")
        if family.startswith(("AF_", "PF_")):
            return "socket:OTHER"
    return name


def _arguments(text: str) -> list[str]:
    """Split strace arguments at top-level commas, respecting quotes and nesting."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    quote = ""
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch
        elif ch in pairs:
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif not stack:
            if ch == ")":
                break  # end of the call, not a ')' in a pathname or buffer
            if ch == ",":
                parts.append(text[start:i].strip())
                start = i + 1
    else:
        i = len(text)
    parts.append(text[start:i].strip())
    return parts


_PID_PREFIX = re.compile(r"(?:\[pid\s+(\d{1,10})\]\s*|(\d{1,10})\s+)")
_TIME_PREFIX = re.compile(r"(?:(\d{1,12}\.\d{1,9})|\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?)\s+")
_CALL = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]{0,95})\(")
_RESUMED = re.compile(r"<\.\.\. ([a-zA-Z_][a-zA-Z0-9_]{0,95}) resumed>\s*")
_RESULT = re.compile(r"\)\s+=\s+")
_UNFINISHED = "<unfinished ...>"


def parse_strace(lines: Iterable[str], *, max_streams: int = DEFAULT_STREAM_LIMIT) -> Iterator[SyscallEvent]:
    """Emit calls in record/completion order; restore split records per Linux TID."""
    limit = stream_limit(max_streams)
    pending: dict[int | None, tuple[str, str, float | None]] = {}
    active: dict[int | None, int] = {}
    last_stream_id = 0
    for raw in lines:
        line = raw.strip()
        pid: int | None = None
        timestamp: float | None = None
        for _ in range(2):  # strace prefixes PID then time; accept the inverse too
            if pid is None and (match := _PID_PREFIX.match(line)):
                pid = int(match[1] or match[2])
                line = line[match.end() :]
            elif match := _TIME_PREFIX.match(line):
                if match[1] and timestamp is None:
                    timestamp = float(match[1])
                line = line[match.end() :]
            else:
                break
        if line.startswith("+++"):
            pending.pop(pid, None)
            active.pop(pid, None)
            continue
        if match := _RESUMED.match(line):
            previous = pending.pop(pid, None)
            if previous is None or previous[0] != match[1]:
                continue  # truncated trace or a mismatched resumption
            name, prefix, entered_at = previous
            args = prefix + line[match.end() :]
            if timestamp is None:
                timestamp = entered_at
        elif match := _CALL.match(line):
            name, args = match[1], line[match.end() :]
            pending.pop(pid, None)
        else:
            continue  # signals, tracing diagnostics, blank lines, etc.
        if args.endswith(_UNFINISHED):
            if len(pending) >= limit:
                raise TraceError(f"too many pending strace records (max_streams={limit})")
            # A resumed call can be interrupted again. Limit aggregate record size too.
            if len(args) > MAX_LINE:
                raise TraceError("unfinished strace record exceeded the line limit")
            pending[pid] = (name, args[: -len(_UNFINISHED)], timestamp)
        elif _RESULT.search(args):
            token = normalize_syscall(name, _arguments(args) if name in _ARG_CALLS else ())
            if pid not in active:
                if len(active) >= limit:
                    raise TraceError(f"too many active strace tasks (max_streams={limit})")
                last_stream_id += 1
                active[pid] = last_stream_id
            yield SyscallEvent(token, pid, timestamp, stream_id=active[pid])


_ARG_CALLS = _OPEN_INDEX.keys() | _PROT_INDEX.keys() | {"socket"}
_AUDIT_FIELDS = re.compile(r'(?:^|\s)([a-zA-Z_][a-zA-Z0-9_]*)=(?:"((?:[^"\\]|\\.)*)"|(\S+))')
_AUDIT_TIME = re.compile(r"audit\((\d{1,12}(?:\.\d{1,9})?):\d+\):?")
_ARCH_ALIAS = {"x86_64": "c000003e", "aarch64": "c00000b7"}


@cache
def _audit_syscalls() -> dict[str, dict[str, str]]:
    with files("core.data").joinpath("linux_syscalls_v6_12.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _audit_arg(value: str) -> str:
    # Raw audit a0..a3 are hex even if all digits happen to be decimal digits.
    if re.fullmatch(r"[a-fA-F0-9]{1,16}", value):
        return "0x" + value
    return value


def parse_auditd(lines: Iterable[str]) -> Iterator[SyscallEvent]:
    """Read SYSCALL records only, ignoring the auxiliary PATH/EXECVE/etc. records."""
    for line in lines:
        fields = {m[1]: m[2] if m[2] is not None else m[3] for m in _AUDIT_FIELDS.finditer(line)}
        if fields.get("type") != "SYSCALL":
            continue
        syscall = fields.get("syscall", "")
        if re.fullmatch(r"(?:-?\d{1,10}|0[xX][0-9a-fA-F]{1,8})", syscall):
            arch = fields.get("arch", "unknown").lower().removeprefix("0x")
            arch = _ARCH_ALIAS.get(arch, arch)
            if not re.fullmatch(r"[0-9a-f]{8}", arch):
                arch = "unknown"
            number = int(syscall, 16 if syscall.lower().startswith("0x") else 10)
            # Unmapped ABI or future syscall: retain a stable, ABI-qualified token.
            suffix = str(number).replace("-", "neg")
            syscall = _audit_syscalls().get(arch, {}).get(str(number), f"syscall_{arch}_{suffix}")
        if not _NAME.fullmatch(syscall):
            raise TraceError("missing or invalid syscall field in audit SYSCALL record")
        try:
            pid = int(fields["pid"], 10)
        except (KeyError, ValueError):
            raise TraceError("missing or invalid pid in audit SYSCALL record") from None
        stamp = _AUDIT_TIME.fullmatch(fields.get("msg", ""))
        args = [_audit_arg(fields.get(f"a{i}", "")) for i in range(4)]
        yield SyscallEvent(
            normalize_syscall(syscall, args), pid, float(stamp[1]) if stamp else None, fields.get("node")
        )


def parse_jsonl(lines: Iterable[str]) -> Iterator[SyscallEvent]:
    """Canonical input accepts `token`; `syscall` is an alias for external producers."""
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict) or ("syscall" in obj) == ("token" in obj):
                raise TraceError("expected an object with either token or syscall")
            token = obj.get("token", obj.get("syscall"))
            if not valid_token(token):
                raise TraceError("invalid syscall token")
            args = obj.get("args", [])
            if not isinstance(args, list) or not all(isinstance(arg, str) or type(arg) is int for arg in args):
                raise TraceError("args must be an array of strings or integers")
            if "syscall" in obj:
                token = normalize_syscall(token, [str(arg) for arg in args])
            yield SyscallEvent(token, obj.get("pid"), obj.get("timestamp"), obj.get("source"), obj.get("stream_id"))
        except (ValueError, RecursionError) as exc:
            raise TraceError(f"line {lineno}: {exc}") from exc


def _bounded_lines(handle) -> Iterator[str]:
    while line := handle.readline(MAX_LINE + 1):
        if len(line) > MAX_LINE:
            raise TraceError(f"trace line exceeds {MAX_LINE} characters")
        yield line


def read_trace(
    path: str | Path, format: str = "auto", *, max_streams: int = DEFAULT_STREAM_LIMIT
) -> Iterator[SyscallEvent]:
    """Normalize one file. File order is preserved; the caller separates capture sessions."""
    path = Path(path)
    if format == "auto":
        format = {".jsonl": "jsonl", ".ndjson": "jsonl", ".audit": "auditd", ".auditd": "auditd"}.get(
            path.suffix.lower(), "auditd" if path.name == "audit.log" else "strace"
        )
    if format not in {"jsonl", "auditd", "strace"}:
        raise TraceError(f"unsupported input format: {format}")
    try:
        with path.open(encoding="utf-8-sig") as handle:
            lines = _bounded_lines(handle)
            if format == "strace":
                yield from parse_strace(lines, max_streams=max_streams)
            elif format == "auditd":
                yield from parse_auditd(lines)
            else:
                yield from parse_jsonl(lines)
    except (UnicodeError, ValueError) as exc:
        raise TraceError(f"{path}: {exc}") from exc


def _fifo_lines(fd: int, path: Path, process: subprocess.Popen) -> Iterator[str]:
    """Use a temporary writer to avoid startup EOF; drain remaining bytes on exit."""
    guard = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    buffer = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                if guard is not None and process.poll() is not None:
                    os.close(guard)
                    guard = None
                if not selector.select(timeout=0.2):
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)
                cursor = 0
                while (end := buffer.find(b"\n", cursor)) >= 0:
                    if end - cursor >= MAX_LINE:
                        raise TraceError("live strace line exceeded the line limit")
                    yield buffer[cursor:end].decode("utf-8", errors="replace")
                    cursor = end + 1
                del buffer[:cursor]
                if len(buffer) >= MAX_LINE:
                    raise TraceError("live strace line exceeded the line limit")
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
    finally:
        if guard is not None:
            os.close(guard)


def _stop_tracer(process: subprocess.Popen, *, spawned: bool) -> None:
    if process.poll() is not None:
        return
    try:
        if spawned:
            # This session was created for the new target. EXITKILL also covers
            # tracees that left its process group. Never do this in attach mode.
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def live_strace(
    *,
    command: Sequence[str] = (),
    pid: int | None = None,
    syscalls: str = "all",
    follow: bool = True,
    max_streams: int = DEFAULT_STREAM_LIMIT,
) -> Iterator[SyscallEvent]:
    """Stream ptrace observations via strace; close the iterator to stop capture."""
    if isinstance(command, (str, bytes)):
        raise TraceError("command must be an argument vector, not a shell string")
    if bool(command) == (pid is not None):
        raise TraceError("choose either --pid PID or a command after --")
    if pid is not None:
        bounded_int(pid, "pid", 1, MAX_PID)
    if not all(_NAME.fullmatch(name) for name in syscalls.split(",")) or (
        "all" in syscalls.split(",") and syscalls != "all"
    ):
        raise TraceError("--syscalls must be 'all' or a comma-separated list of syscall names")
    stream_limit(max_streams)
    if sys.platform != "linux":
        raise TraceError("live capture requires Linux; offline capture files work on any platform")
    if not (binary := shutil.which("strace")):
        raise TraceError("strace not found; install strace or select a recording")
    with TemporaryDirectory(prefix="syscall-hids-") as directory:
        fifo = Path(directory) / "trace.pipe"
        os.mkfifo(fifo, mode=0o600)
        # The target never inherits the trace descriptor. Its stdout/stderr
        # cannot be mistaken for tracing records, even when it prints strace-like text.
        reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        try:
            argv = [binary, "-ttt", "-s", "128", "-e", f"trace={syscalls}", "-o", str(fifo)]
            if follow:
                argv.append("-f")
            argv.extend(["-p", str(pid)] if pid is not None else ["--kill-on-exit", "--", *command])
            process = subprocess.Popen(argv, start_new_session=True)
            try:
                with closing(_fifo_lines(reader, fifo, process)) as lines:
                    for event in parse_strace(lines, max_streams=max_streams):
                        yield replace(event, pid=pid) if pid is not None and event.pid is None and not follow else event
                status = process.wait()
                if status:
                    raise TraceError(
                        f"strace or target exited with status {status}; check strace diagnostics and ptrace policy"
                    )
            finally:
                _stop_tracer(process, spawned=pid is None)
        finally:
            os.close(reader)
