import inspect

import gradio as gr

from sam3_gui.mask_app import make_demo


class RecordingBackend:
    def __init__(self):
        self.closed_sessions = []

    def close_session(self, session_id):
        self.closed_sessions.append(session_id)


def get_lifecycle_callback(demo, event_name):
    for dependency in demo.fns.values():
        if any(target[1] == event_name for target in dependency.targets):
            return dependency.fn
    raise AssertionError(f"Missing {event_name} callback")


def get_handler_callback(demo, callback_name, provider_name):
    for dependency in demo.fns.values():
        callback = dependency.fn
        if callback is None or getattr(callback, "__name__", None) != callback_name:
            continue
        nonlocals = inspect.getclosurevars(callback).nonlocals
        if provider_name in nonlocals:
            return callback
    raise AssertionError(f"Missing {provider_name} callback {callback_name}")


def test_browser_sessions_get_isolated_handlers_with_shared_backend(tmp_path):
    backend = RecordingBackend()
    demo = make_demo(str(tmp_path), backend=backend)
    initialize_session = get_lifecycle_callback(demo, "load")
    request_a = gr.Request(session_hash="browser-a")
    request_b = gr.Request(session_hash="browser-b")

    initialize_session(request_a)
    initialize_session(request_b)

    video_registry = demo.session_handler_registries["video"]
    image_registry = demo.session_handler_registries["image"]
    video_a = video_registry.get(request_a)
    video_b = video_registry.get(request_b)
    image_a = image_registry.get(request_a)
    image_b = image_registry.get(request_b)

    assert video_a is not video_b
    assert image_a is not image_b
    assert video_a.backend is backend
    assert video_b.backend is backend
    assert image_a.backend is backend
    assert image_b.backend is backend
    assert len(video_registry) == 2
    assert len(image_registry) == 2


def test_unload_closes_and_removes_only_the_departing_browser_session(tmp_path):
    backend = RecordingBackend()
    demo = make_demo(str(tmp_path), backend=backend)
    initialize_session = get_lifecycle_callback(demo, "load")
    close_session = get_lifecycle_callback(demo, "unload")
    request_a = gr.Request(session_hash="browser-a")
    request_b = gr.Request(session_hash="browser-b")

    initialize_session(request_a)
    initialize_session(request_b)

    video_registry = demo.session_handler_registries["video"]
    image_registry = demo.session_handler_registries["image"]
    video_a = video_registry.get(request_a)
    image_a = image_registry.get(request_a)
    video_b = video_registry.get(request_b)
    image_b = image_registry.get(request_b)
    video_a.session_id = "video-a"
    image_a.session_id = "image-a"
    video_b.session_id = "video-b"
    image_b.session_id = "image-b"

    close_session(request_a)

    assert backend.closed_sessions == ["video-a", "image-a"]
    assert len(video_registry) == 1
    assert len(image_registry) == 1
    assert video_registry.get(request_b) is video_b
    assert image_registry.get(request_b) is image_b
    assert video_b.session_id == "video-b"
    assert image_b.session_id == "image-b"


def test_reloading_after_unload_creates_fresh_handlers(tmp_path):
    backend = RecordingBackend()
    demo = make_demo(str(tmp_path), backend=backend)
    initialize_session = get_lifecycle_callback(demo, "load")
    close_session = get_lifecycle_callback(demo, "unload")
    request = gr.Request(session_hash="browser-a")

    initialize_session(request)
    video_registry = demo.session_handler_registries["video"]
    image_registry = demo.session_handler_registries["image"]
    first_video = video_registry.get(request)
    first_image = image_registry.get(request)

    close_session(request)
    initialize_session(request)

    assert video_registry.get(request) is not first_video
    assert image_registry.get(request) is not first_image


def test_stateful_ui_callbacks_resolve_handlers_from_request(tmp_path):
    backend = RecordingBackend()
    demo = make_demo(str(tmp_path), backend=backend)
    request_a = gr.Request(session_hash="browser-a")
    request_b = gr.Request(session_hash="browser-b")
    video_set_negative = get_handler_callback(
        demo,
        "set_negative",
        "video_handler_provider",
    )
    image_set_threshold = get_handler_callback(
        demo,
        "set_confidence_threshold",
        "image_handler_provider",
    )

    video_set_negative(request_a)
    image_set_threshold(0.2, request_a)
    image_set_threshold(0.8, request_b)

    video_registry = demo.session_handler_registries["video"]
    image_registry = demo.session_handler_registries["image"]
    assert video_registry.get(request_a).cur_label_val == 0
    assert video_registry.get(request_b).cur_label_val == 1
    assert image_registry.get(request_a).confidence_threshold == 0.2
    assert image_registry.get(request_b).confidence_threshold == 0.8
