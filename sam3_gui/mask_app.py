import os
from threading import RLock
from typing import Callable, Generic, TypeVar

import gradio as gr
import torch

from sam3_gui.config import (
    DEFAULT_IMG_NAME,
    DEFAULT_MASK_NAME,
    DEFAULT_VID_NAME,
)
from sam3_gui.image_handler import ImageModeHandler
from sam3_gui.image_ui import build_image_tab
from sam3_gui.utils import (
    first_or_none,
    image_file_path,
    list_image_files,
    list_image_folders,
    list_video_files,
)
from sam3_gui.video_handler import VideoModeHandler
from sam3_gui.video_ui import build_video_tab


HandlerT = TypeVar("HandlerT")


class SessionHandlerRegistry(Generic[HandlerT]):
    def __init__(self, factory: Callable[[], HandlerT]):
        self._factory = factory
        self._handlers: dict[str, HandlerT] = {}
        self._lock = RLock()

    @staticmethod
    def _session_hash(request: gr.Request) -> str:
        session_hash = request.session_hash if request is not None else None
        if not session_hash:
            raise ValueError("Gradio request is missing session_hash")
        return session_hash

    def get(self, request: gr.Request) -> HandlerT:
        session_hash = self._session_hash(request)
        with self._lock:
            handler = self._handlers.get(session_hash)
            if handler is None:
                handler = self._factory()
                self._handlers[session_hash] = handler
            return handler

    def close(self, request: gr.Request) -> None:
        session_hash = self._session_hash(request)
        with self._lock:
            handler = self._handlers.pop(session_hash, None)
        if handler is not None:
            handler.reset()

    def __len__(self) -> int:
        with self._lock:
            return len(self._handlers)


# Enable tf32 for Ampere GPUs if any CUDA device supports it
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            break


def make_demo(
    root_dir,
    checkpoint_path=None,
    gpus_to_use=None,
    vid_name: str = DEFAULT_VID_NAME,
    img_name: str = DEFAULT_IMG_NAME,
    mask_name: str = DEFAULT_MASK_NAME,
    backend=None,
):
    video_handlers = SessionHandlerRegistry(
        lambda: VideoModeHandler(
            checkpoint_path=checkpoint_path,
            gpus_to_use=gpus_to_use,
            backend=backend,
        )
    )
    image_handlers = SessionHandlerRegistry(
        lambda: ImageModeHandler(checkpoint_path=checkpoint_path, backend=backend)
    )

    vid_root = os.path.join(root_dir, vid_name)
    img_root = os.path.join(root_dir, img_name)
    initial_videos = list_video_files(vid_root)
    initial_video = first_or_none(initial_videos)
    initial_frame_dirs = list_image_folders(img_root)
    initial_frame_dir = first_or_none(initial_frame_dirs)
    initial_image_folders = initial_frame_dirs
    initial_image_folder = first_or_none(initial_image_folders)
    initial_image_files = (
        list_image_files(os.path.join(img_root, initial_image_folder))
        if initial_image_folder
        else []
    )
    initial_image_file = first_or_none(initial_image_files)
    initial_image_path = image_file_path(
        root_dir, img_name, initial_image_folder, initial_image_file
    )

    with gr.Blocks(title="SAM3.1 Segmentation") as demo:
        gr.Markdown("# SAM3.1 Segmentation Tool")
        instruction = gr.Textbox(
            "Select a mode (Video or Image) to get started.",
            label="Status",
            interactive=False,
        )

        with gr.Tabs():
            build_video_tab(
                root_dir=root_dir,
                vid_name=vid_name,
                img_name=img_name,
                mask_name=mask_name,
                initial_videos=initial_videos,
                initial_video=initial_video,
                initial_frame_dirs=initial_frame_dirs,
                initial_frame_dir=initial_frame_dir,
                video_handler_provider=video_handlers.get,
                instruction=instruction,
            )
            build_image_tab(
                root_dir=root_dir,
                img_name=img_name,
                initial_image_folders=initial_image_folders,
                initial_image_folder=initial_image_folder,
                initial_image_files=initial_image_files,
                initial_image_file=initial_image_file,
                initial_image_path=initial_image_path,
                image_handler_provider=image_handlers.get,
                instruction=instruction,
            )

        def initialize_session(request: gr.Request):
            video_handlers.get(request)
            image_handlers.get(request)

        def close_session(request: gr.Request):
            try:
                video_handlers.close(request)
            finally:
                image_handlers.close(request)

        demo.load(initialize_session, api_name=False)
        demo.unload(close_session)

    demo.session_handler_registries = {
        "video": video_handlers,
        "image": image_handlers,
    }

    return demo


if __name__ == "__main__":
    raise SystemExit("Use `python cli.py` to launch the app.")
