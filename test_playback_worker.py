"""Unit tests for PlaybackWorker.

Verifies:
1. Chunks enqueued are written by the background thread.
2. interrupt() drains all queued chunks and calls stream.abort() + stream.start().
3. interrupt() while writing discards remaining chunks.
4. stop() is clean and idempotent.
"""

import time
import unittest
from unittest.mock import MagicMock
from playback_worker import PlaybackWorker


class TestPlaybackWorker(unittest.TestCase):
    def setUp(self):
        self.mock_stream = MagicMock()
        self.mock_stream.active = True
        self.worker = PlaybackWorker(out_stream=self.mock_stream)

    def tearDown(self):
        self.worker.stop()

    def test_enqueue_and_write(self):
        chunk1 = b"\x00\x01" * 100
        chunk2 = b"\x02\x03" * 100
        self.worker.enqueue(chunk1)
        self.worker.enqueue(chunk2)

        # Wait briefly for worker thread to process
        time.sleep(0.1)
        self.assertGreaterEqual(self.mock_stream.write.call_count, 2)
        self.mock_stream.write.assert_any_call(chunk1)
        self.mock_stream.write.assert_any_call(chunk2)

    def test_interrupt_aborts_and_drains(self):
        # Simulate slow writes
        def slow_write(chunk):
            time.sleep(0.05)
        self.mock_stream.write.side_effect = slow_write

        # Queue 20 chunks
        for _ in range(20):
            self.worker.enqueue(b"\x00" * 200)

        # Immediately interrupt
        self.worker.interrupt()

        # Check that stream.abort was called
        self.mock_stream.abort.assert_called()
        self.mock_stream.start.assert_called()

        # The queue should now be empty
        self.assertTrue(self.worker._queue.empty())

    def test_stop_idempotent(self):
        self.worker.stop()
        self.assertFalse(self.worker._running)
        # Second stop must not raise
        self.worker.stop()
        self.worker.close()


if __name__ == "__main__":
    unittest.main()
