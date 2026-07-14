import sys
import types
from argparse import Namespace
from pathlib import Path


import cli
import sam3_gui.config as config
from tools import download_model


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = str(REPO_ROOT / "cli.py")


def test_default_checkpoint_candidates_are_sam31_multiplex_only():
    candidates = config.default_checkpoint_candidates(CLI_PATH)

    assert candidates
    assert all("sam3.1_multiplex.pt" in path for path in candidates)
    assert all("sam3.pt" not in path for path in candidates)
    assert all("sam3.1.pt" not in path for path in candidates)


def test_resolve_checkpoint_path_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "custom-sam3.1_multiplex.pt"

    assert config.resolve_checkpoint_path(str(explicit), []) == str(explicit)


def test_resolve_checkpoint_path_uses_existing_candidate(tmp_path):
    missing = tmp_path / "missing-sam3.1_multiplex.pt"
    existing = tmp_path / "sam3.1_multiplex.pt"
    existing.write_bytes(b"")

    assert config.resolve_checkpoint_path(None, [str(missing), str(existing)]) == str(
        existing
    )


def test_legacy_gpu_arg_becomes_hydra_override():
    args = Namespace(gpus="0, 2", use_fa3=False, reload=False, share=False)

    assert "sam.gpus=[0, 2]" in config.legacy_args_to_overrides(args)


def test_build_parser_defaults_point_to_api_launcher_data_root():
    args = cli.build_parser(CLI_PATH).parse_args([])
    cfg = config.load_app_config(CLI_PATH)

    assert args.port is None
    assert cfg.server.port == config.DEFAULT_PORT
    assert cfg.data.root_dir == str(REPO_ROOT / "data_root")
    assert args.checkpoint_path is None
    assert args.use_fa3 is False


def test_build_parser_accepts_fa3_flag():
    args = cli.build_parser(CLI_PATH).parse_args(["--use_fa3"])

    assert args.use_fa3 is True


def test_hydra_overrides_config_values():
    cfg = config.load_app_config(
        CLI_PATH,
        ["server.port=8891", "sam.use_fa3=true", "sam.gpus=[0]"],
    )

    assert cfg.server.port == 8891
    assert cfg.sam.use_fa3 is True
    assert cfg.sam.gpus == [0]


def test_download_model_uses_sam31_hf_repo(monkeypatch, tmp_path):
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        checkpoint = tmp_path / kwargs["filename"]
        checkpoint.write_bytes(b"checkpoint")
        return str(checkpoint)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    assert download_model.download_sam31_model(str(tmp_path)) is True
    assert calls == [
        {
            "repo_id": "facebook/sam3.1",
            "filename": "sam3.1_multiplex.pt",
            "local_dir": str(tmp_path),
        }
    ]


def test_download_model_reports_hf_errors(monkeypatch, tmp_path):
    def failing_hf_hub_download(**kwargs):
        raise RuntimeError("gated")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=failing_hf_hub_download),
    )

    assert download_model.download_sam31_model(str(tmp_path)) is False


def test_download_model_uses_modelscope_repo(monkeypatch, tmp_path):
    calls = []

    def fake_model_file_download(**kwargs):
        calls.append(kwargs)
        checkpoint = tmp_path / kwargs["file_path"]
        checkpoint.write_bytes(b"checkpoint")
        return str(checkpoint)

    monkeypatch.setitem(sys.modules, "modelscope", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.file_download",
        types.SimpleNamespace(model_file_download=fake_model_file_download),
    )

    assert (
        download_model.download_sam31_model(str(tmp_path), source="modelscope") is True
    )
    assert calls == [
        {
            "model_id": "facebook/sam3.1",
            "file_path": "sam3.1_multiplex.pt",
            "local_dir": str(tmp_path),
        }
    ]


def test_download_model_uses_custom_modelscope_repo(monkeypatch, tmp_path):
    calls = []

    def fake_model_file_download(**kwargs):
        calls.append(kwargs)
        checkpoint = tmp_path / kwargs["file_path"]
        checkpoint.write_bytes(b"checkpoint")
        return str(checkpoint)

    monkeypatch.setitem(sys.modules, "modelscope", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "modelscope.hub", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.file_download",
        types.SimpleNamespace(model_file_download=fake_model_file_download),
    )

    assert (
        download_model.download_sam31_model(
            str(tmp_path),
            source="modelscope",
            modelscope_model_id="mirror/sam31",
        )
        is True
    )
    assert calls[0]["model_id"] == "mirror/sam31"


def test_download_model_reports_modelscope_import_error(monkeypatch, tmp_path):
    def missing_modelscope(*args, **kwargs):
        raise ImportError("modelscope missing")

    monkeypatch.setattr(download_model, "_import_modelscope_attr", missing_modelscope)

    assert (
        download_model.download_sam31_model(str(tmp_path), source="modelscope") is False
    )


def test_download_model_rejects_unknown_source(tmp_path):
    assert download_model.download_sam31_model(str(tmp_path), source="unknown") is False


def test_download_model_parser_accepts_modelscope_source():
    args = download_model.build_parser().parse_args(
        ["--source", "modelscope", "--modelscope_model_id", "facebook/sam3.1"]
    )

    assert args.source == "modelscope"
    assert args.modelscope_model_id == "facebook/sam3.1"
