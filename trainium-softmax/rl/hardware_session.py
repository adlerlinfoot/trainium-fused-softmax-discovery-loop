"""Run a TP4 command while restoring the seat's Qwen service on exit.

Use only on the dedicated seat, after pushing a source checkpoint.
The existing server command is captured from /proc rather than reconstructed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('a hardware command is required')
    proc = Path('/proc') / str(args.server_pid)
    server = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
    if 'serve' not in server or 'Qwen/Qwen3-8B' not in server:
        raise RuntimeError('PID is not the expected Qwen server')
    server_env = dict(part.decode().split('=', 1) for part in
                      (proc / 'environ').read_bytes().split(b'\0') if part)
    group = os.getpgid(args.server_pid)
    if group != args.server_pid:
        raise RuntimeError('Server does not own its process group')
    print(json.dumps({'phase': 'pausing_qwen', 'command': server}), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        os.killpg(group, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            members = [p for p in Path('/proc').iterdir() if p.name.isdigit()
                       and p.name != str(os.getpid()) and _in_group(p, group)]
            if not members:
                break
            time.sleep(1)
        else:
            os.killpg(group, signal.SIGKILL)
        child = subprocess.Popen(command, start_new_session=True)
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        with Path('/tmp/vllm.log').open('a') as log:
            restored = subprocess.Popen(server, env=server_env, stdout=log, stderr=log,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'phase': 'restoring_qwen', 'pid': restored.pid}), flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if restored.poll() is not None:
                raise RuntimeError('Restored Qwen server exited; inspect /tmp/vllm.log')
            try:
                with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
                    if response.status == 200:
                        print(json.dumps({'phase': 'qwen_restored'}), flush=True)
                        break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('Qwen restoration did not become healthy within 600 seconds')


def _in_group(proc, group):
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[0] != 'Z' and int(stat[2]) == group
    except (OSError, ValueError, IndexError):
        return False


if __name__ == '__main__':
    raise SystemExit(main())
