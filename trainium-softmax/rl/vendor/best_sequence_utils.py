from __future__ import annotations
import json, os, time
from contextlib import contextmanager
from typing import Any
from .state import to_json_serializable
@contextmanager
def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}\n{time.time()}\n".encode("utf-8"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            # If stale, delete and try again
            try:
                st = os.stat(lock_path)
                if (time.time() - st.st_mtime) > stale_s:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(obj), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json_or_default(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Corrupt/partial file; treat as empty
        return default

