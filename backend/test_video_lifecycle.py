
import threading
import unittest

from video_lifecycle import (
    video_operation,
    try_acquire_video_lock,
    release_video_lock,
    begin_video_stream,
    end_video_stream,
    active_stream_count,
)


class TestVideoLifecycle(unittest.TestCase):

    def test_active_stream_blocks_cleanup(self):
        video_id = "test-stream"

        begin_video_stream(video_id)
        try:
            self.assertEqual(active_stream_count(video_id), 1)
            self.assertIsNone(try_acquire_video_lock(video_id))
        finally:
            end_video_stream(video_id)

        lock = try_acquire_video_lock(video_id)
        self.assertIsNotNone(lock)
        release_video_lock(lock)

    def test_multiple_streams_block_cleanup(self):
        video_id = "test-multiple-streams"

        begin_video_stream(video_id)
        begin_video_stream(video_id)

        try:
            self.assertEqual(active_stream_count(video_id), 2)

            end_video_stream(video_id)
            self.assertEqual(active_stream_count(video_id), 1)

            self.assertIsNone(try_acquire_video_lock(video_id))
        finally:
            end_video_stream(video_id)

        self.assertEqual(active_stream_count(video_id), 0)

    def test_indexing_blocks_cleanup(self):
        video_id = "test-indexing"
        started = threading.Event()
        release = threading.Event()

        def worker():
            with video_operation(video_id):
                started.set()
                release.wait(timeout=5)

        thread = threading.Thread(target=worker)
        thread.start()

        try:
            self.assertTrue(started.wait(timeout=5))
            self.assertIsNone(try_acquire_video_lock(video_id))
        finally:
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())

        lock = try_acquire_video_lock(video_id)
        self.assertIsNotNone(lock)
        release_video_lock(lock)


if __name__ == "__main__":
    unittest.main(verbosity=2)
