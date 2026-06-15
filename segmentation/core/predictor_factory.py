"""Factory for building SAM2 or SAM3 video predictors.

Both backends are wrapped to expose a unified API matching SAM2's interface:
    predictor.init_state(video_path=...) -> inference_state
    predictor.add_new_points_or_box(inference_state, frame_idx, obj_id, box=...)
    predictor.propagate_in_video(inference_state, reverse=False)
        -> yields (frame_idx, obj_ids, mask_logits)
    predictor.reset_state(inference_state)
"""

import os
import logging
import warnings

# Suppress pkg_resources deprecation warning from SAM3
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)

logger = logging.getLogger(__name__)

# Frame filter: when set, the patched SAM frame loaders only load filenames
# in this set. Set to None to load all frames (default behavior).
# Used by _apply_frame_range to avoid copying/symlinking frames.
_frame_name_filter = None


def set_frame_filter(frame_names):
    """Set the frame filter. Only these filenames will be loaded by SAM.

    Args:
        frame_names: Set/list of filenames (basenames, e.g. {"0000000200.png", ...}),
                     or None to clear the filter.
    """
    global _frame_name_filter
    _frame_name_filter = set(frame_names) if frame_names is not None else None


def clear_frame_filter():
    """Clear the frame filter so SAM loads all frames."""
    global _frame_name_filter
    _frame_name_filter = None


def _patch_sam2_png_support():
    """
    Monkey-patch sam2.utils.misc to support PNG images and robust frame sorting.
    Replaces the build-time .def patch with a runtime equivalent.
    """
    try:
        import sam2.utils.misc as sam2_misc
    except ImportError:
        logger.warning("sam2.utils.misc not found; skipping PNG patch.")
        return

    _original_load_video_frames_from_jpg_images = getattr(
        sam2_misc, "load_video_frames_from_jpg_images", None
    )
    if _original_load_video_frames_from_jpg_images is None:
        return

    # Check if already patched
    if getattr(sam2_misc, "_png_patch_applied", False):
        return

    import os
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    def _load_img_as_tensor(img_path, image_size):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size
        return img, video_height, video_width

    def patched_load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only image frames in a folder are supported. "
                "Use ffmpeg to extract frames from video files."
            )

        # PNG + JPEG support
        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        # Robust sort: try integer-based, fall back to lexicographic
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set (for --start/--end frame range)
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        if async_loading_frames:
            lazy_images = sam2_misc.AsyncVideoFrameLoader(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG/PNG)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    sam2_misc.load_video_frames_from_jpg_images = patched_load_video_frames_from_jpg_images
    sam2_misc._png_patch_applied = True
    logger.info("Applied SAM2 PNG/sorting monkey-patch to sam2.utils.misc.")


def _patch_sam2_sdpa_dtype():
    """Make scaled_dot_product_attention tolerant of mixed dtypes (MPS/CPU).

    SAM2 stores memory-bank features as bfloat16 (a CUDA memory optimization
    that pairs with CUDA's bf16 autocast — see SAM2VideoPredictor where
    ``maskmem_features.to(torch.bfloat16)`` is unconditional). On MPS/CPU there
    is no autocast, so the float32 current-frame query/key meet the bf16 memory
    ``value`` in memory attention and torch raises:
        "Expected query, key, and value to have the same dtype".
    Upcast the offending inputs to the query dtype. This is a no-op when all
    three already match (the CUDA path), so behavior there is unchanged.
    """
    import torch.nn.functional as F

    if getattr(F, "_sam2_sdpa_dtype_patch", False):
        return
    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa(query, key, value, *args, **kwargs):
        if not (query.dtype == key.dtype == value.dtype):
            key = key.to(query.dtype)
            value = value.to(query.dtype)
        return _orig_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = _sdpa
    F._sam2_sdpa_dtype_patch = True
    logger.info("Applied SAM2 scaled_dot_product_attention mixed-dtype monkey-patch (MPS/CPU).")


def _patch_sam3_png_support():
    """
    Monkey-patch sam3.model.utils.sam2_utils to support PNG images and robust
    frame sorting — same logic as _patch_sam2_png_support but targeting SAM3's
    copy of the loader.
    """
    try:
        import sam3.model.utils.sam2_utils as sam3_misc
    except ImportError:
        logger.warning("sam3.model.utils.sam2_utils not found; skipping SAM3 PNG patch.")
        return

    if getattr(sam3_misc, "_png_patch_applied", False):
        return

    if not hasattr(sam3_misc, "load_video_frames_from_jpg_images"):
        return

    import os
    import torch
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    def _load_img_as_tensor(img_path, image_size):
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size
        return img, video_height, video_width

    def patched_load_video_frames(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only image frames in a folder are supported. "
                "Use ffmpeg to extract frames from video files."
            )

        frame_names = [
            p for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set (for --start/--end frame range)
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        if async_loading_frames:
            lazy_images = sam3_misc.AsyncVideoFrameLoader(
                img_paths, image_size, offload_video_to_cpu,
                img_mean, img_std, compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG/PNG)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)
        images -= img_mean
        images /= img_std
        return images, video_height, video_width

    sam3_misc.load_video_frames_from_jpg_images = patched_load_video_frames
    sam3_misc._png_patch_applied = True
    logger.info("Applied SAM3 PNG/sorting monkey-patch to sam3.model.utils.sam2_utils.")

    # Also patch sam3.model.io_utils (used by Sam3VideoPredictor / sam3_prompt mode)
    try:
        import sam3.model.io_utils as sam3_io
    except ImportError:
        return

    if getattr(sam3_io, "_frame_filter_patch_applied", False):
        return

    _original_load_from_folder = getattr(sam3_io, "load_video_frames_from_image_folder", None)
    if _original_load_from_folder is None:
        return

    def patched_load_from_image_folder(
        image_folder, image_size, offload_video_to_cpu, img_mean, img_std, async_loading_frames,
    ):
        """Patched loader that respects _frame_name_filter."""
        import os
        frame_names = [
            p for p in os.listdir(image_folder)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()

        # Apply frame filter if set
        if _frame_name_filter is not None:
            frame_names = [f for f in frame_names if f in _frame_name_filter]

        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {image_folder}")
        img_paths = [os.path.join(image_folder, fn) for fn in frame_names]
        img_mean_t = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        img_std_t = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

        if async_loading_frames:
            lazy_images = sam3_io.AsyncImageFrameLoader(
                img_paths, image_size, offload_video_to_cpu, img_mean_t, img_std_t
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float16)
        video_height, video_width = None, None
        for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (image folder, patched)")):
            images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            # Respect the available accelerator instead of hardcoding CUDA, so
            # this loader also works on MPS (Apple Silicon) / CPU-only hosts.
            dev = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            images = images.to(dev)
            img_mean_t = img_mean_t.to(dev)
            img_std_t = img_std_t.to(dev)
        images -= img_mean_t
        images /= img_std_t
        return images, video_height, video_width

    sam3_io.load_video_frames_from_image_folder = patched_load_from_image_folder
    sam3_io._frame_filter_patch_applied = True
    logger.info("Applied frame-filter monkey-patch to sam3.model.io_utils.")


def _to_relative_box(box, width, height):
    """Convert absolute pixel box [x1, y1, x2, y2] to relative [0-1] coords."""
    import numpy as np
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 1:
        box = box.reshape(1, 4)
    rel = box.copy()
    rel[:, [0, 2]] /= width
    rel[:, [1, 3]] /= height
    return rel


def _to_relative_points(points, width, height):
    """Convert absolute pixel points [[x, y], ...] to relative [0-1] coords."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    rel = points.copy()
    rel[..., 0] /= width
    rel[..., 1] /= height
    return rel


class _Sam3VideoAdapter:
    """
    Adapter wrapping SAM3's tracker to match SAM2's video predictor API.

    Key differences handled:
    - Build: use build_sam3_video_model().tracker with shared detector backbone
    - Coordinates: SAM3 expects relative [0-1], our pipeline passes absolute pixels
    - Reset: SAM3 uses clear_all_points_in_video() instead of reset_state()
    - Propagate: SAM3 yields 5-tuple, SAM2 yields 3-tuple
    """

    def __init__(self, sam3_model):
        # Per official sam3_for_sam2_video_task_example.ipynb:
        # use the tracker directly, share backbone from detector
        self._tracker = sam3_model.tracker
        self._tracker.backbone = sam3_model.detector.backbone
        self._video_width = None
        self._video_height = None

    def init_state(self, video_path, **kwargs):
        state = self._tracker.init_state(video_path=video_path, **kwargs)
        # Cache video dims for coordinate conversion
        self._video_height = state["video_height"]
        self._video_width = state["video_width"]
        return state

    def reset_state(self, inference_state):
        self._tracker.clear_all_points_in_video(inference_state)

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        box=None,
        **kwargs,
    ):
        # Convert absolute pixel coords -> relative [0-1] for SAM3
        if self._video_width is None or self._video_height is None:
            raise RuntimeError("Must call init_state() before add_new_points_or_box()")
        if box is not None:
            box = _to_relative_box(box, self._video_width, self._video_height)
        if points is not None:
            points = _to_relative_points(points, self._video_width, self._video_height)

        result = self._tracker.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            box=box,
            **kwargs,
        )
        # SAM3 tracker returns 5-tuple: (frame_idx, obj_ids, low_res, video_res, scores)
        # SAM2 returns 3-tuple: (frame_idx, obj_ids, mask_logits)
        # Return video_res as mask_logits (same resolution as SAM2's output)
        _, out_obj_ids, _low_res, video_res_masks, *_ = result
        return _, out_obj_ids, video_res_masks

    def propagate_in_video(self, inference_state, reverse=False, **kwargs):
        """
        Adapt SAM3 tracker's 5-tuple yield to SAM2's 3-tuple.

        start_frame_idx=None lets SAM3 auto-detect the first annotation frame,
        which is critical for reverse propagation (backward from the anchor frame).
        propagate_preflight=True is only effective on the first call (consolidates
        temp outputs); subsequent calls (e.g. reverse pass) are a no-op.
        """
        for result in self._tracker.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=None,
            max_frame_num_to_track=None,
            reverse=reverse,
            propagate_preflight=True,
        ):
            frame_idx, obj_ids, _low_res, video_res_masks, _scores = result
            yield frame_idx, obj_ids, video_res_masks


class _Sam3PromptVideoAdapter:
    """
    Adapter wrapping SAM3's video predictor (session-based API with text prompts).

    This uses the new SAM3 API (build_sam3_video_predictor) which supports:
    - Text prompts: "person" auto-detects and tracks all people with instance IDs
    - Point/box prompts: for manual/matched prompts on non-init cameras
    - Bidirectional propagation: forward + backward in one call

    The session-based API is fundamentally different from the SAM2-compatible
    tracker API (_Sam3VideoAdapter), so this adapter exposes its own interface.
    """

    def __init__(self, checkpoint_path=None, tracking_overrides=None):
        from sam3.model_builder import build_sam3_video_predictor  # type: ignore
        import torch
        gpus = list(range(torch.cuda.device_count())) or [0]
        kwargs = {"gpus_to_use": gpus}
        # Only pass checkpoint_path if it's a real file path (not a HuggingFace ID).
        # When None or a HF ID like "facebook/sam3", let SAM3 handle the download.
        if checkpoint_path and os.path.isfile(checkpoint_path):
            kwargs["checkpoint_path"] = checkpoint_path
        # As used in https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb
        self._predictor = build_sam3_video_predictor(**kwargs)
        self._apply_tracking_overrides(tracking_overrides)
        self._session_id = None
        self._video_width = None
        self._video_height = None

    def _apply_tracking_overrides(self, overrides):
        """Override SAM3 tracking config on the live model after construction.

        Fields are instance attributes on predictor.model (e.g. new_det_thresh,
        trk_assoc_iou_thresh) and are read fresh on every frame during inference,
        so overriding them here takes effect immediately.
        """
        if not overrides:
            return
        model = getattr(self._predictor, "model", None)
        if model is None:
            logger.warning("SAM3 tracking overrides: cannot find predictor.model, skipping.")
            return
        applied = []
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(model, key):
                old = getattr(model, key)
                setattr(model, key, value)
                applied.append(f"{key}: {old} -> {value}")
            else:
                logger.warning(f"SAM3 tracking override ignored: '{key}' not found on model.")
        if applied:
            logger.info("SAM3 tracking overrides applied: " + ", ".join(applied))

    def start_session(self, video_path):
        """Start a session on a video (directory of frames or MP4)."""
        response = self._predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        self._session_id = response["session_id"]
        # Cache video dimensions from the inference state
        session = self._predictor._get_session(self._session_id)
        if session:
            state = session["state"]
            self._video_height = state["orig_height"]
            self._video_width = state["orig_width"]
        return self._session_id

    def add_text_prompt(self, frame_idx, text="person"):
        """Add a text prompt (e.g. 'person') to detect all instances on a frame.

        Returns:
            Dict with 'obj_ids' (list of int), 'masks' (dict obj_id -> bool tensor),
            'boxes' (dict obj_id -> [x1,y1,x2,y2] absolute pixels).
        """
        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_idx,
                text=text,
            )
        )
        return self._parse_outputs(response.get("outputs"))

    def add_box_prompt(self, frame_idx, obj_id, box_xyxy):
        """Add a box prompt for a specific object on a frame.

        Args:
            box_xyxy: [x1, y1, x2, y2] in absolute pixel coordinates.
        """
        import numpy as np
        box = np.asarray(box_xyxy, dtype=np.float32)
        # Convert xyxy to xywh normalized [0-1]
        w_video = self._video_width or 1
        h_video = self._video_height or 1
        cx = (box[0] + box[2]) / 2.0 / w_video
        cy = (box[1] + box[3]) / 2.0 / h_video
        w = (box[2] - box[0]) / w_video
        h = (box[3] - box[1]) / h_video
        import torch
        boxes_xywh = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
        box_labels = torch.tensor([True])

        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=frame_idx,
                bounding_boxes=boxes_xywh,
                bounding_box_labels=box_labels,
                obj_id=obj_id,
            )
        )
        return self._parse_outputs(response.get("outputs"))

    def propagate(self):
        """Propagate all prompts bidirectionally. Returns {frame_idx: {obj_id: mask}}."""
        import torch
        import numpy as np
        video_segments = {}
        # As used in https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb
        for response in self._predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self._session_id,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response.get("outputs")
            if outputs is None:
                continue
            parsed = self._parse_outputs(outputs)
            video_segments[frame_idx] = {
                oid: (mask.cpu().numpy() if hasattr(mask, 'cpu') else np.asarray(mask))
                for oid, mask in parsed["masks"].items()
            }
        return dict(sorted(video_segments.items()))

    def reset_session(self):
        """Reset the current session."""
        if self._session_id:
            self._predictor.handle_request(
                request=dict(type="reset_session", session_id=self._session_id)
            )

    def close_session(self):
        """Close the current session and free GPU resources."""
        if self._session_id:
            self._predictor.handle_request(
                request=dict(type="close_session", session_id=self._session_id)
            )
            self._session_id = None

    def shutdown(self):
        """Shutdown the predictor entirely."""
        self.close_session()
        self._predictor.shutdown()

    def _parse_outputs(self, outputs):
        """Parse SAM3 video predictor outputs into a usable format."""
        import torch
        import numpy as np
        result = {"obj_ids": [], "masks": {}, "boxes": {}}
        if outputs is None:
            return result

        obj_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", torch.zeros(0, dtype=torch.int64)))
        masks = outputs.get("out_binary_masks", outputs.get("binary_masks", torch.zeros(0)))
        boxes_xywh = outputs.get("out_boxes_xywh", outputs.get("boxes_xywh", torch.zeros(0, 4)))

        # Ensure obj_ids is iterable as a list of ints
        if hasattr(obj_ids, 'tolist'):
            obj_ids_list = obj_ids.tolist()
        elif isinstance(obj_ids, (list, tuple)):
            obj_ids_list = list(obj_ids)
        else:
            obj_ids_list = [int(obj_ids)]

        w_video = self._video_width or 1
        h_video = self._video_height or 1

        n_masks = masks.shape[0] if hasattr(masks, 'shape') else 0
        n_boxes = boxes_xywh.shape[0] if hasattr(boxes_xywh, 'shape') else 0

        for i, oid in enumerate(obj_ids_list):
            result["obj_ids"].append(oid)
            if i < n_masks:
                result["masks"][oid] = masks[i]  # bool array or tensor [H, W]
            if i < n_boxes:
                # Convert normalized xywh to absolute xyxy
                bx = boxes_xywh[i]
                cx = float(bx[0])
                cy = float(bx[1])
                bw = float(bx[2])
                bh = float(bx[3])
                x1 = (cx - bw / 2) * w_video
                y1 = (cy - bh / 2) * h_video
                x2 = (cx + bw / 2) * w_video
                y2 = (cy + bh / 2) * h_video
                result["boxes"][oid] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return result

    @property
    def video_width(self):
        return self._video_width

    @property
    def video_height(self):
        return self._video_height


def build_video_predictor(sam_version: str, config: str | None, checkpoint: str | None, device,
                          tracking_overrides: dict | None = None) -> object:
    """
    Build a SAM video predictor for the given backend version.

    Args:
        sam_version: "sam2" or "sam3"
        config: Model config path (required for sam2, ignored for sam3)
        checkpoint: Model checkpoint path or HuggingFace model ID
        device: torch device
        tracking_overrides: Optional dict of SAM3 tracking config overrides
            (e.g. new_det_thresh, trk_assoc_iou_thresh). Only used for sam3_prompt.

    Returns:
        A video predictor object with init_state / add_new_points_or_box /
        propagate_in_video / reset_state API (SAM2-compatible interface).
    """
    if sam_version == "sam2":
        _patch_sam2_png_support()
        # On non-CUDA devices (MPS/CPU) there is no bf16 autocast, so SAM2's
        # bf16 memory-bank features mismatch float32 attention inputs. Keep the
        # CUDA path byte-for-byte unchanged.
        if getattr(device, "type", str(device)) != "cuda":
            _patch_sam2_sdpa_dtype()
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        return build_sam2_video_predictor(config, checkpoint, device=device)

    elif sam_version == "sam3":
        _patch_sam3_png_support()
        from sam3.model_builder import build_sam3_video_model  # type: ignore
        sam3_model = build_sam3_video_model()
        return _Sam3VideoAdapter(sam3_model)

    elif sam_version == "sam3_prompt":
        _patch_sam3_png_support()
        return _Sam3PromptVideoAdapter(checkpoint_path=checkpoint, tracking_overrides=tracking_overrides)

    else:
        raise ValueError(f"Unknown sam_version: {sam_version!r}. Expected 'sam2', 'sam3', or 'sam3_prompt'.")
