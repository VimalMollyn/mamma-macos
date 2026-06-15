"""In-memory video frame reader for MP4 files.

No frames are written to disk. SAM receives the MP4 path directly;
YOLO and CLIP receive PIL Images from this reader.

Usage::

    from capture.video_reader import VideoFrameReader, cam_data_from_video

    reader = VideoFrameReader("/path/to/IOI_09.mp4", start=10, end=50)
    pil_img = reader.read_pil(0)   # frame 0 within the range (global frame 10)
    print(len(reader))             # 40 frames

    cam_data = cam_data_from_video("/path/to/IOI_09.mp4", start=10, end=50)
"""
import logging
import os
import threading
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class VideoFrameReader:
    """Frame reader for MP4 videos. No disk I/O for frames.

    Supports reading any frame index in any order, but is optimized for the
    common in-order (forward) access pattern. Keeps a single persistent
    ``cv2.VideoCapture`` and decodes **forward**: when the requested frame is at
    or after the current position it reads sequentially (no seek), and only
    seeks when asked for an earlier frame. For 4K H.264/H.265 with a large GOP
    this avoids the per-frame seek-from-keyframe re-decode that made sequential
    access quadratic. Out-of-order / backward reads still work, paying a seek.

    Frames returned are byte-identical to the previous open/seek/read/close
    implementation: same decoder, same color order (BGR), same dtype.

    Thread-safe: the persistent capture and its position cursor are guarded by
    an internal lock, so concurrent callers (e.g. the SAM3 mask-export thread
    pool) are serialized at the decode boundary while their per-frame work still
    runs in parallel. The lock is uncontended for the common single-threaded
    forward scan, so that path keeps the full sequential-decode speedup.
    """

    def __init__(self, video_path: str, start: int = None, end: int = None):
        import cv2

        self.video_path = os.path.abspath(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.start = max(0, min(start or 0, total))
        self.end = max(self.start, min(end or total, total))
        self.n_frames = self.end - self.start
        # Persistent capture for sequential forward decode. Lazily opened on the
        # first read; ``_cap_pos`` is the global index of the NEXT frame the
        # capture will return (one past the last decoded frame). ``_lock``
        # serializes access to both so the reader stays thread-safe.
        self._cap = None
        self._cap_pos = -1
        self._lock = threading.Lock()
        logger.info(
            "VideoFrameReader: '%s' frames [%d:%d] (%d of %d frames, %dx%d, %.1f fps)",
            os.path.basename(video_path), self.start, self.end,
            self.n_frames, total, self.width, self.height, self.fps,
        )

    def __len__(self) -> int:
        return self.n_frames

    def _ensure_cap(self):
        """Open the persistent capture if needed. Caller must hold ``_lock``."""
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self._cap_pos = 0
        return self._cap

    def read_bgr(self, local_idx: int) -> np.ndarray:
        """Read frame as BGR numpy array (cv2 convention).

        Decodes forward from the persistent capture; seeks only when the
        requested global index is behind the current cursor. Thread-safe.
        """
        import cv2

        if local_idx < 0 or local_idx >= self.n_frames:
            raise IndexError(f"Frame index {local_idx} out of range [0, {self.n_frames})")
        global_idx = self.start + local_idx

        with self._lock:
            cap = self._ensure_cap()

            # Seek only when going backward (or cursor unknown). Forward access
            # reads sequentially, skipping intervening frames without a seek.
            if global_idx < self._cap_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
                self._cap_pos = global_idx
            elif global_idx > self._cap_pos:
                # Skip-decode forward to the target (grab() decodes without copy).
                while self._cap_pos < global_idx:
                    if not cap.grab():
                        raise RuntimeError(
                            f"Failed to advance to frame {global_idx} in '{self.video_path}'")
                    self._cap_pos += 1

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {global_idx} from '{self.video_path}'")
            self._cap_pos += 1
            return frame

    def close(self):
        """Release the persistent capture, if open."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                self._cap_pos = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read_rgb(self, local_idx: int) -> np.ndarray:
        """Read frame as RGB numpy array."""
        import cv2
        return cv2.cvtColor(self.read_bgr(local_idx), cv2.COLOR_BGR2RGB)

    def read_pil(self, local_idx: int):
        """Read frame as PIL Image (RGB)."""
        from PIL import Image
        return Image.fromarray(self.read_rgb(local_idx))


def cam_data_from_video(video_path: str, start: int = None, end: int = None) -> Dict[str, Any]:
    """Build a cam_data dict from an MP4 video -- no disk I/O.

    Args:
        video_path: Path to MP4 video file.
        start: First frame index (0-based, inclusive). None = 0.
        end: Last frame index (0-based, exclusive). None = all frames.

    Returns:
        Dict compatible with ``process_multi_video_auto`` / ``FrameSource`` factory.
    """
    cam_name = os.path.splitext(os.path.basename(video_path))[0]
    reader = VideoFrameReader(video_path, start=start, end=end)

    return {
        'cam_name': np.array(cam_name),
        'video_path': np.array(os.path.abspath(video_path)),
        'frame_reader': reader,
    }
