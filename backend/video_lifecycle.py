
"""
VideoMoment video lifecycle coordination.

Coordinates indexing, searching, streaming, and cleanup
within a single Python process.

Requires a single API worker.
"""

import threading
from collections import defaultdict
from contextlib import contextmanager


_registry_lock = threading.Lock()
_video_locks = {}
_active_streams = defaultdict(int)


def get_video_lock(video_id: str) -> threading.RLock:
    with _registry_lock:
        lock = _video_locks.get(video_id)

        if lock is None:
            lock = threading.RLock()
            _video_locks[video_id] = lock

        return lock


@contextmanager
def video_operation(video_id: str):
    """
    Hold the lifecycle lock for indexing or searching.
    """
    lock = get_video_lock(video_id)

    with lock:
        yield


def try_acquire_video_lock(video_id: str):
    """
    Attempt to acquire the video lifecycle lock.

    Cleanup receives None if indexing/searching is active
    or a stream has been registered.
    """
    lock = get_video_lock(video_id)

    if not lock.acquire(blocking=False):
        return None

    with _registry_lock:
        if _active_streams.get(video_id, 0) > 0:
            lock.release()
            return None

    return lock


def release_video_lock(lock):
    lock.release()


def begin_video_stream(video_id: str):
    """
    Register an active video stream.

    Must be called before returning the streaming response.
    """
    with video_operation(video_id):
        with _registry_lock:
            _active_streams[video_id] += 1


def end_video_stream(video_id: str):
    """
    Release a registered video stream.

    Must run when the response finishes or is cancelled.
    """
    with _registry_lock:
        count = _active_streams.get(video_id, 0)

        if count <= 1:
            _active_streams.pop(video_id, None)
        else:
            _active_streams[video_id] = count - 1


def active_stream_count(video_id: str) -> int:
    with _registry_lock:
        return _active_streams.get(video_id, 0)
