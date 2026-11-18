from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from contextlib import closing

import pytest

from core.tracer import (
    SyscallEvent,
    TraceError,
    live_strace,
    normalize_syscall,
    parse_auditd,
    parse_jsonl,
    parse_strace,
    read_trace,
)


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("open", ['"a,b"', "O_RDONLY|O_CLOEXEC"], "open:READ"),
        ("openat", ["AT_FDCWD", '"file"', "O_RDONLY|O_CREAT"], "openat:WRITE"),
        ("openat", ["AT_FDCWD", '"file"', "0x80000"], "openat:READ"),
        ("openat", ["AT_FDCWD", '"file"', "0x400000"], "openat:WRITE"),
        ("openat2", ["AT_FDCWD", '"file"', "{flags=O_RDWR, mode=0644}"], "openat2:WRITE"),
        ("openat2", ["AT_FDCWD", '"file"', "0xff014030"], "openat2"),
        ("creat", ['"file"', "0644"], "creat:WRITE"),
        ("mprotect", ["0x1234", "4096", "PROT_READ|0x4"], "mprotect:EXEC"),
        ("pkey_mprotect", ["0x1234", "4096", "PROT_READ|PROT_EXEC", "0"], "pkey_mprotect:EXEC"),
        ("mmap", ["NULL", "4096", "PROT_READ|PROT_WRITE", "MAP_PRIVATE"], "mmap"),
        ("mprotect", ["0x1234", "4096"], "mprotect"),
        ("socket", ["AF_INET6", "SOCK_STREAM"], "socket:INET"),
        ("socket", ["AF_UNIX", "SOCK_STREAM"], "socket:UNIX"),
        ("socket", ["AF_NETLINK"], "socket:NETLINK"),
        ("socket", ["AF_ALG"], "socket:OTHER"),
        ("execve", ['"/not-retained"'], "execve"),
    ],
)
def test_security_argument_categories(name, args, expected):
    assert normalize_syscall(name, args) == expected


def test_strace_formats_and_unfinished_calls():
    text = r"""
1234  1750000000.000001 read(4,  <unfinished ...>
[pid 66] 1750000000.000002 openat(AT_FDCWD, "name-with-\",)}", O_RDONLY|O_CLOEXEC) = 9
[pid 66] --- SIGUSR1 {si_signo=SIGUSR1} ---
1234  1750000000.000003 <... read resumed>"secret", 5) = 5
1234  1750000000.000004 mprotect(0x7ffa1000, 4096, PROT_READ|PROT_EXEC) = 0
1234  1750000000.000005 +++ exited with 0 +++
1750000000.000006 [pid 9] socket(AF_INET, SOCK_STREAM|SOCK_CLOEXEC, 0) = 4
11:12:13.000456 close(3) = 0
openat2(AT_FDCWD, "name,{", {flags=O_RDWR, mode=0644}, 24) = 3
18 write(5,  <unfinished ...>
18 +++ killed by SIGTERM +++
18 <... write resumed>"text", 4) = 4
strace: Process 2 attached
"""
    parsed = list(parse_strace(text.splitlines()))
    assert [(e.token, e.pid) for e in parsed] == [
        ("openat:READ", 66),
        ("read", 1234),
        ("mprotect:EXEC", 1234),
        ("socket:INET", 9),
        ("close", None),
        ("openat2:WRITE", None),
    ]
    assert parsed[1].timestamp == 1750000000.000003
    assert parsed[-2].timestamp is None
    assert "secret" not in "".join(e.json_line() for e in parsed)


def test_strace_exit_distinguishes_reused_pids():
    parsed = list(
        parse_strace(
            [
                "71 read(0, NULL, 0) = 0",
                "71 close(0) = 0",
                "71 +++ exited with 0 +++",
                '71 execve("/bin/true", [], []) = 0',
            ]
        )
    )
    assert parsed[0].stream == parsed[1].stream != parsed[2].stream
    assert parsed[0].pid == parsed[1].pid == parsed[2].pid == 71
    assert list(parse_jsonl(e.json_line() for e in parsed)) == parsed


def test_audit_abi_resolution_hex_registers_and_record_filter():
    data = [
        'node=n1 type=SYSCALL msg=audit(1750000000.125:1): arch=c000003e syscall=257 pid=15 a2=80000 comm="pid=99"',
        'node=n1 type=PATH msg=audit(1750000000.125:1): name="a private file"',
        "type=SYSCALL msg=audit(1750000001:2): arch=c00000b7 syscall=56 pid=8 a2=241",
        "type=SYSCALL msg=audit(1750000002.9:3): arch=aarch64 syscall=226 pid=8 a2=5",
        "type=SYSCALL msg=audit(1750000003.9:4): arch=0xc00000b7 syscall=198 pid=8 a0=a",
        "type=SYSCALL msg=audit(1750000004.9:5): arch=c000003e syscall=41 pid=8 a0=10",
        "type=SYSCALL msg=audit(1750000004.9:6): arch=c000003e syscall=999 pid=9",
        "type=SYSCALL msg=audit(1750000005.9:7): arch=40000003 syscall=11 pid=9",
        "type=SYSCALL msg=audit(1750000006.9:8): arch=c00000b7 syscall=437 pid=9 a2=123",
        "type=EXECVE msg=audit(1750000007.9:9): argc=1",
    ]
    items = list(parse_auditd(data))
    assert [e.token for e in items] == [
        "openat:READ",
        "openat:WRITE",
        "mprotect:EXEC",
        "socket:INET",
        "socket:NETLINK",
        "syscall_c000003e_999",
        "syscall_40000003_11",
        "openat2",
    ]
    assert items[0] == SyscallEvent("openat:READ", pid=15, timestamp=1750000000.125, source="n1")


def test_jsonl_roundtrip_and_recorded_arguments(tmp_path):
    path = tmp_path / "trace.ndjson"
    data = [SyscallEvent("mprotect:EXEC", 321, 1750000000.25, "node-1"), SyscallEvent("close")]
    path.write_text("".join(event.json_line() for event in data), encoding="utf-8")
    assert list(read_trace(path)) == data
    assert list(parse_jsonl(['{"syscall":"socket","args":[2,"SOCK_STREAM"],"pid":1}'])) == [
        SyscallEvent("socket:INET", 1)
    ]


@pytest.mark.parametrize(
    "line",
    [
        "{not a JSON object}",
        "[]",
        "{}",
        '{"token":"<UNK>"}',
        '{"token":"x","timestamp":NaN}',
        '{"token":"x","pid":true}',
        '{"token":"x","source":"weird\u001b"}',
        '{"token":"x","syscall":"x"}',
        '{"syscall":"openat","args":"invalid"}',
    ],
)
def test_jsonl_input_error_has_file_and_line(tmp_path, line):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"token":"read"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(TraceError, match="trace.jsonl: line 2:"):
        list(read_trace(path))


def test_explicit_format_override_and_line_limit(tmp_path):
    path = tmp_path / "capture.log"
    path.write_text("type=SYSCALL msg=audit(1750000000.0:1): arch=c000003e syscall=59 pid=17\n", encoding="utf-8")
    assert list(read_trace(path)) == []
    assert list(read_trace(path, "auditd"))[0].token == "execve"
    path.write_text("x" * ((1 << 20) + 1), encoding="utf-8")
    with pytest.raises(TraceError, match="trace line exceeds"):
        list(read_trace(path, "strace"))


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX FIFO and strace process semantics")
def test_live_pipe_and_stdout_isolation_with_fake_tracer(tmp_path, monkeypatch, capfd):
    fake = tmp_path / "strace"
    fake.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        assert args[:5] == ['-ttt', '-s', '128', '-e', 'trace=read,write']
        assert args[-3:] == ['--', 'benign-app', 'argument;as-data']
        assert '--kill-on-exit' in args
        print('77 1750000000.1 execve("not-a-trace-record", [], []) = 0', file=sys.stderr)
        with Path(args[args.index('-o') + 1]).open('w') as f:
            f.write('17 1750000000.2 read(0,  <unfinished ...>\\n')
            f.flush()
            f.write('18 1750000000.3 write(1, "ok", 2) = 2\\n')
            f.write('17 1750000000.4 <... read resumed>"x", 1) = 1\\n')
        """),
        encoding="utf-8",
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    result = list(live_strace(command=["benign-app", "argument;as-data"], syscalls="read,write"))
    assert [(e.token, e.pid) for e in result] == [("write", 18), ("read", 17)]
    assert "not-a-trace-record" in capfd.readouterr().err


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX FIFO and strace process semantics")
def test_startup_error_before_fifo_open_does_not_hang(tmp_path, monkeypatch):
    fake = tmp_path / "strace"
    fake.write_text(f"#!{sys.executable}\nraise SystemExit(14)\n", encoding="utf-8")
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    target = tmp_path / "capture.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "main", "trace", "-o", str(target), "--", "unused"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1 and "status 14" in result.stderr
    assert not target.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX FIFO and strace process semantics")
def test_closing_attach_terminates_only_tracer(tmp_path, monkeypatch):
    marker = tmp_path / "detached"
    fake = tmp_path / "strace"
    fake.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""\
        import os
        import signal
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        def detach(*_):
            Path(os.environ['DETACH_MARKER']).touch()
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, detach)
        assert args[-2:] == ['-p', '321'] and '--kill-on-exit' not in args
        with Path(args[args.index('-o') + 1]).open('w') as f:
            print('321 1750000000.0 read(1, "x", 1) = 1', file=f, flush=True)
            signal.pause()
        """),
        encoding="utf-8",
    )
    fake.chmod(0o700)
    monkeypatch.setenv("DETACH_MARKER", str(marker))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with closing(live_strace(pid=321)) as capture:
        assert next(capture).pid == 321
    assert marker.exists()


@pytest.mark.live
@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux strace")
def test_native_strace_spawn_own_benign_process(capfd):
    binary = shutil.which("strace")
    if not binary:
        pytest.skip("strace is not installed")
    help_text = subprocess.check_output([binary, "--help"], text=True)
    if "--kill-on-exit" not in help_text:
        pytest.skip("installed strace lacks --kill-on-exit")
    try:
        result = list(
            live_strace(
                command=[sys.executable, "-c", "print('benign workload')"], syscalls="execve,openat,write,close"
            )
        )
    except TraceError:
        if "Operation not permitted" in capfd.readouterr().err:
            pytest.skip("sandbox ptrace policy denied tracing")
        raise
    assert result[0].token == "execve"
    assert any(e.token == "openat:READ" for e in result)
    assert any(e.token == "write" for e in result)
