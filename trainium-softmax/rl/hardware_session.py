"""Run a TP4 command while restoring the seat's Qwen service on exit.

Use only on the dedicated seat, after pushing a source checkpoint.
The existing server command is captured from /proc rather than reconstructed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--stop-at', type=float, help='Absolute UTC epoch at which experiments stop')
    parser.add_argument('--timeout-seconds', type=float, default=2700)
    parser.add_argument('--evidence-source', type=Path)
    parser.add_argument('--evidence-target', type=Path)
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
    deadline_timer = None
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
        stop_at = min(time.time() + args.timeout_seconds, args.stop_at or float('inf'))
        deadline_timer = threading.Timer(max(0, stop_at - time.time()),
            lambda: os.kill(os.getpid(), signal.SIGTERM))
        deadline_timer.daemon = True
        deadline_timer.start()
        last_push = time.monotonic()
        while child.poll() is None:
            remaining = stop_at - time.time()
            if remaining <= 0:
                print(json.dumps({'phase': 'experiment_deadline', 'stop_at': stop_at}), flush=True)
                return 124
            try:
                child.wait(timeout=min(20, remaining))
            except subprocess.TimeoutExpired:
                pass
            if args.evidence_source and time.monotonic() - last_push >= 240:
                checkpoint(args.evidence_source, args.evidence_target)
                last_push = time.monotonic()
        return child.returncode
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        if child is not None and child.poll() is None:
            groups = descendant_groups(child.pid)
            for group_id in groups:
                try:
                    os.killpg(group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                for group_id in groups:
                    try:
                        os.killpg(group_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
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


def descendant_groups(pid):
    parents = {}
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                parents[int(proc.name)] = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    descendants = {pid}
    while True:
        expanded = descendants | {child for child, parent in parents.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    groups = set()
    for descendant in descendants:
        try:
            groups.add(os.getpgid(descendant))
        except ProcessLookupError:
            pass
    groups.discard(os.getpgrp())
    return groups


def checkpoint(source, target):
    if target is None:
        raise ValueError('Evidence target required for periodic checkpoints')
    target.mkdir(parents=True, exist_ok=True)
    for file in source.glob('rank-*/*.json'):
        destination = target / file.parent.name / file.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(file, destination)
    (target / 'progress.json').write_text(json.dumps(
        {'scope': 'partial-model verification', 'checkpoint_utc_epoch': time.time()}, indent=2))
    subprocess.run(['git', 'add', str(target)], check=True, timeout=15)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], timeout=15).returncode
    if changed:
        subprocess.run(['git', 'commit', '-m', 'Checkpoint partial TP4 hardware progress'], check=True, timeout=15)
    subprocess.run(['git', 'push', 'origin', 'HEAD'], check=True, timeout=45)
    print(json.dumps({'phase': 'github_checkpoint_pushed', 'time': time.time()}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
