"""CPU-only contract tests for the opt-in ai-toolkit state bridge."""

from __future__ import annotations

import copy
import json
import io
import os
import random
import subprocess
import sys
import tarfile
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is not None:
    from torch.utils.data import DataLoader, TensorDataset
else:
    DataLoader = TensorDataset = None

from app.services import aitoolkit_state_bridge as activation
from app.services import training_state_bundle as bundles
from app.training_bridge import lds_aitk_bridge_contract as bridge_contract
from app.training_bridge.lds_aitk_bridge_contract import (
    BRIDGE_PROTOCOL,
    ENV_AITK_ROOT,
    ENV_ENABLE,
    ENV_IDENTITY_FILE,
    ENV_PROTOCOL,
    ENV_RESERVE_BYTES,
    IDENTITY_SCHEMA,
    atomic_json_nofollow,
    probe_aitoolkit_source,
)

# requirements-dev intentionally excludes Torch.  The production runtime imports
# it eagerly, so keep the activation/security contracts collectable in lean CI
# and skip only the tests which need that runtime module.
if torch is not None:
    from app.training_bridge import lds_aitk_bridge_runtime as runtime
else:
    runtime = None

_requires_torch_runtime = pytest.mark.skipif(
    torch is None,
    reason="the ai-toolkit bridge runtime requires torch",
)


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            pytest.skip(f"junction creation unavailable: {completed.stderr}")
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlink creation unavailable: {exc}")


def _remove_directory_link(link: Path) -> None:
    if not os.path.lexists(link):
        return
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def _fake_source_tree(root: Path, *, recognised: bool = True) -> Path:
    base = root / "jobs" / "process" / "BaseSDTrainProcess.py"
    sd = root / "extensions_built_in" / "sd_trainer" / "SDTrainer.py"
    data = root / "toolkit" / "data_loader.py"
    mixins = root / "toolkit" / "dataloader_mixins.py"
    run = root / "run.py"
    base.parent.mkdir(parents=True)
    sd.parent.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    (root / "version.py").write_text('VERSION = "0.test"\n', encoding="utf-8")
    base.write_text(
        """
class BaseSDTrainProcess:
    def save(self, step=None):
        self.ema.eval()
        self.ema.train()
    def hook_before_train_loop(self):
        self.prepare_accelerator()
    def run(self):
        self.lr_scheduler.step(self.step_num)
        dataloader_iterator = iter(dataloader)
        self.hook_train_loop(batch)
        self.save(self.step_num)
        self.step_num = step + 1
""".replace(
            "self.save(self.step_num)",
            "self.save_other(self.step_num)" if not recognised else "self.save(self.step_num)",
        ),
        encoding="utf-8",
    )
    sd.write_text(
        """
class SDTrainer:
    def hook_before_train_loop(self):
        super().hook_before_train_loop()
    def hook_train_loop(self, batch):
        self.optimizer.step()
        self.ema.update()
        self.lr_scheduler.step()
""",
        encoding="utf-8",
    )
    data.write_text(
        """
class AiToolkitDataset(object):
    def setup_epoch(self):
        self.setup_buckets()
        self.cache_latents_all_latents()
        self.epoch_num += 1
""",
        encoding="utf-8",
    )
    mixins.write_text(
        """
class BucketsMixin:
    def setup_buckets(self, quiet=False):
        self.buckets = {}
        self.shuffle_buckets()
        self.build_batch_indices()

    def cache_latents_all_latents(self):
        latent_path = file_item.get_latent_path(recalculate=True)
        if os.path.exists(latent_path):
            pass

    def cache_text_embeddings(self):
        text_embedding_path = file_item.get_text_embedding_path(recalculate=True)
        if not os.path.exists(text_embedding_path):
            pass
""",
        encoding="utf-8",
    )
    run.write_text(
        """
from dotenv import load_dotenv
load_dotenv()
os.environ["DISABLE_TELEMETRY"] = "YES"
import torch
from toolkit.accelerator import get_accelerator
accelerator = get_accelerator()
""",
        encoding="utf-8",
    )
    return root


def _identity(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": IDENTITY_SCHEMA,
                "config_hash": "config-v1",
                "dataset_hash": "dataset-v1",
                "base_hash": "base-v1",
                "network_hash": "network-v1",
                "toolkit_revision": "toolkit-v1",
                "runtime": {
                    "torch": getattr(torch, "__version__", "not-installed"),
                    "device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class _EMA:
    def __init__(self, parameter: torch.nn.Parameter):
        self.shadow = parameter.detach().clone().add_(10)

    def state_dict(self):
        return {"shadow": self.shadow.clone()}

    def load_state_dict(self, value):
        self.shadow = value["shadow"].clone()


class _Trainer:
    def __init__(self, root: Path):
        self.save_root = str(root)
        root.mkdir(parents=True, exist_ok=True)
        self.parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=0.05)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            mixed_precision="no",
            scaler=None,
            is_main_process=True,
        )
        self.ema = _EMA(self.parameter)
        self.data_loader = DataLoader(
            TensorDataset(torch.arange(12)), batch_size=2, shuffle=True, num_workers=0
        )
        self.data_loader_reg = None
        self.train_config = SimpleNamespace(start_step=None)
        self.step_num = 3
        self.start_step = 0
        self.last_save_step = 0
        self.epoch_num = 2
        self.grad_accumulation_step = 0
        self.is_grad_accumulation_step = False
        self._lds_last_optimizer_boundary_step = 3
        # ai-toolkit's loop is zero-based: checkpoint K is written after update K.
        self._lds_optimizer_updates_completed = 4
        self.checkpoint = root / "job_000000003.safetensors"
        self.checkpoint.write_bytes(b"public-checkpoint")

    def get_latest_save_path(self):
        return str(self.checkpoint)


class _Bucket:
    def __init__(self, width, height, indices):
        self.width = width
        self.height = height
        self.file_list_idx = list(indices)


class _BucketFile:
    def __init__(self, root: Path, token: str, ordinal: int):
        self.token = token
        self.path = str(root / f"export-{ordinal}-{token}.png")
        Path(self.path).write_bytes(f"image:{token}".encode())
        Path(self.path).with_suffix(".txt").write_text(
            f"caption:{token}", encoding="utf-8"
        )
        self.mask_path = str(root / f"mask-{ordinal}-{token}.png")
        Path(self.mask_path).write_bytes(f"mask:{token}".encode())
        self.control_path = None
        self.width = 12
        self.height = 10
        self.flip_x = False
        self.flip_y = False
        self.scale_to_width = 12
        self.scale_to_height = 10
        self.crop_x = ordinal % 3
        self.crop_y = ordinal % 2
        self.crop_width = 8
        self.crop_height = 8
        self.is_latent_cached = False
        self.is_text_embedding_cached = False
        self.loaded_latent_bytes = None
        self.loaded_text_bytes = None

    def get_latent_path(self, recalculate=False):
        return str(
            Path(self.path).parent
            / "_latent_cache"
            / f"{Path(self.path).stem}-{self.crop_x}-{self.crop_y}.safetensors"
        )

    def get_text_embedding_path(self, recalculate=False):
        return str(
            Path(self.path).parent
            / "_t_e_cache"
            / f"{Path(self.path).stem}.safetensors"
        )


def _dataset_config():
    return SimpleNamespace(
        type="image",
        buckets=True,
        random_crop=True,
        random_scale=False,
        resolution=8,
        scale=1.0,
        square_crop=False,
        num_repeats=1,
        flip_x=False,
        flip_y=False,
        standardize_images=False,
        caption_ext=".txt",
        default_caption=None,
        cache_latents=False,
        cache_latents_to_disk=True,
        cache_text_embeddings=True,
        mask_min_value=0.1,
        invert_mask=False,
        num_frames=1,
        auto_frame_count=False,
        augments=[],
        augmentations=None,
        shuffle_augmentations=False,
        controls=[],
        control_path=None,
        control_from_same_folder=False,
        inpaint_path=None,
        unconditional_path=None,
        clip_image_path=None,
        clip_image_from_same_folder=False,
        alpha_mask=False,
        cache_clip_vision_to_disk=False,
        load_image_when_caching_latents=False,
    )


_TorchDatasetBase = torch.utils.data.Dataset if torch is not None else object


class _BucketDataset(_TorchDatasetBase):
    def __init__(self, root: Path, tokens, *, saved_layout: bool):
        root.mkdir(parents=True, exist_ok=True)
        self.dataset_config = _dataset_config()
        self.file_list = [
            _BucketFile(root, token, index)
            for index, token in enumerate(tokens)
        ]
        self.caption_dict = None
        self.batch_size = 2
        self.epoch_num = 3 if saved_layout else 0
        self.is_video = False
        self.is_audio_model = False
        self.is_generating_controls = False
        self.is_caching_latents = True
        self.is_caching_latents_to_disk = True
        self.is_caching_text_embeddings = True
        if saved_layout:
            self.buckets = OrderedDict({
                "8x8": _Bucket(8, 8, [2, 0, 1]),
            })
            self.batch_indices = [[2, 0], [1, 1]]
            self._write_original_caches()
        else:
            self.buckets = OrderedDict()
            self.batch_indices = []

    def _write_original_caches(self):
        for item in self.file_list:
            latent = Path(item.get_latent_path(recalculate=True))
            latent.parent.mkdir(exist_ok=True)
            latent.write_bytes(f"latent:{item.token}".encode())
            item.is_latent_cached = True
            text = Path(item.get_text_embedding_path(recalculate=True))
            text.parent.mkdir(exist_ok=True)
            text.write_bytes(f"text:{item.token}".encode())
            item.is_text_embedding_cached = True

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, index):
        return torch.tensor(self.batch_indices[index], dtype=torch.int64)


class _LinearDataset(_BucketDataset):
    def __init__(self, root: Path, tokens, *, saved_layout: bool):
        super().__init__(root, tokens, saved_layout=saved_layout)
        self.dataset_config.buckets = False
        self.dataset_config.random_crop = False
        self.buckets = OrderedDict()
        self.batch_indices = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.int64)


def _fake_setup_buckets(dataset, quiet=False):
    indices = list(range(len(dataset.file_list)))
    random.shuffle(indices)
    for item in dataset.file_list:
        item.crop_x = random.randint(0, 4)
        item.crop_y = random.randint(0, 2)
        item.crop_width = 8
        item.crop_height = 8
    dataset.buckets = OrderedDict({"8x8": _Bucket(8, 8, indices)})
    first = indices[:2]
    second = indices[2:]
    second = second + second if len(second) == 1 else second
    dataset.batch_indices = [first, second]


def _fake_cache_latents(dataset):
    for item in dataset.file_list:
        path = Path(item.get_latent_path(recalculate=True))
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"new-latent:{random.random()}".encode())
        item.loaded_latent_bytes = path.read_bytes()
        item.is_latent_cached = True


def _fake_cache_text(dataset):
    for item in dataset.file_list:
        path = Path(item.get_text_embedding_path(recalculate=True))
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"new-text:{random.random()}".encode())
        item.loaded_text_bytes = path.read_bytes()
        item.is_text_embedding_cached = True


def _fake_setup_epoch(dataset):
    if dataset.epoch_num == 0:
        if dataset.dataset_config.buckets:
            runtime._patched_setup_buckets(dataset)
        runtime._patched_cache_latents(dataset)
        runtime._patched_cache_text(dataset)
    dataset.epoch_num += 1


def test_source_probe_accepts_only_the_recognised_lifecycle(tmp_path):
    supported = probe_aitoolkit_source(_fake_source_tree(tmp_path / "ok"))
    rejected = probe_aitoolkit_source(
        _fake_source_tree(tmp_path / "changed", recognised=False)
    )
    assert supported["supported"] is True
    assert supported["shape_revision"] == "aitk-base-sd-train/v2"
    assert rejected["supported"] is False
    assert any("lifecycle" in reason or "save" in reason for reason in rejected["reasons"])


def test_sitecustomize_defers_runtime_until_upstream_accelerator_returns(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    toolkit = bootstrap / "toolkit"
    toolkit.mkdir(parents=True)
    (toolkit / "__init__.py").write_text("", encoding="utf-8")
    log = tmp_path / "bootstrap-order.log"
    status = tmp_path / "bridge-status.json"

    sitecustomize_source = Path(bridge_contract.__file__).with_name(
        "sitecustomize.py"
    ).read_text(encoding="utf-8")
    (bootstrap / "sitecustomize.py").write_text(
        sitecustomize_source,
        encoding="utf-8",
    )
    (bootstrap / "lds_aitk_bridge_contract.py").write_text(
        """
import json
import os
from pathlib import Path

def atomic_json_nofollow(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding="utf-8")

def load_identity(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def verify_identity_model_artifacts(identity):
    with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
        stream.write("model-verify\\n")
    if identity.get("verified") is not True:
        raise RuntimeError("model artifacts were not verified")
""",
        encoding="utf-8",
    )
    (bootstrap / "torch.py").write_text(
        """
import os
with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
    stream.write("torch-import\\n")
""",
        encoding="utf-8",
    )
    (toolkit / "accelerator.py").write_text(
        """
import os

def get_accelerator():
    with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
        stream.write("accelerator-enter\\n")
    os.environ["ACCELERATOR_READY"] = "1"
    with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
        stream.write("accelerator-return\\n")
    return object()
""",
        encoding="utf-8",
    )
    (bootstrap / "lds_aitk_bridge_runtime.py").write_text(
        """
import os
import sys

with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
    stream.write("runtime-import\\n")
if os.environ.get("UPSTREAM_ENV_READY") != "1":
    raise RuntimeError("upstream environment is not ready")
if os.environ.get("ACCELERATOR_READY") != "1":
    raise RuntimeError("accelerator is not ready")
if "torch" not in sys.modules:
    raise RuntimeError("torch is not imported")

def install_from_environment():
    with open(os.environ["ORDER_LOG"], "a", encoding="utf-8") as stream:
        stream.write("install\\n")
""",
        encoding="utf-8",
    )
    child = """
import os
os.environ["UPSTREAM_ENV_READY"] = "1"
import torch
from toolkit.accelerator import get_accelerator
accelerator = get_accelerator()
"""
    identity = tmp_path / "child-identity.json"
    identity.write_text('{"verified":true}', encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(bootstrap),
            "PYTHONNOUSERSITE": "1",
            "ORDER_LOG": str(log),
            "LDS_AITK_BRIDGE_ENABLE": "1",
            "LDS_AITK_BRIDGE_STRICT": "1",
            "LDS_AITK_BRIDGE_STATUS_FILE": str(status),
            "LDS_AITK_IDENTITY_FILE": str(identity),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-s", "-c", child],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "torch-import",
        "accelerator-enter",
        "accelerator-return",
        "model-verify",
        "runtime-import",
        "install",
    ]

    log.unlink()
    status.unlink(missing_ok=True)
    identity.write_text('{"verified":false}', encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, "-s", "-c", child],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert rejected.returncode == 78
    assert log.read_text(encoding="utf-8").splitlines() == [
        "torch-import",
        "accelerator-enter",
        "accelerator-return",
        "model-verify",
    ]
    failure = json.loads(status.read_text(encoding="utf-8"))
    assert failure["status"] == "bootstrap_error"
    assert "model artifacts were not verified" in failure["reasons"][0]


def test_status_json_refuses_symlink_target_without_touching_victim(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_text('{"owned":true}\n', encoding="utf-8")
    status = tmp_path / "status.json"
    try:
        status.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(OSError):
        atomic_json_nofollow(status, {"status": "patched"})

    assert activation.read_status(status) is None
    assert victim.read_text(encoding="utf-8") == '{"owned":true}\n'


def test_status_reader_rejects_symlink_swap_between_lstat_and_open(
    tmp_path, monkeypatch
):
    status = tmp_path / "status.json"
    status.write_text('{"status":"safe"}\n', encoding="utf-8")
    victim = tmp_path / "victim.json"
    victim.write_text('{"status":"attacker"}\n', encoding="utf-8")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(victim)
        probe.unlink()
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    original_open = bridge_contract.os.open
    swapped = False

    def swap_then_open(path, flags, *args):
        nonlocal swapped
        if not swapped and Path(path) == status:
            swapped = True
            status.unlink()
            status.symlink_to(victim)
        return original_open(path, flags, *args)

    monkeypatch.setattr(bridge_contract.os, "open", swap_then_open)

    assert activation.read_status(status) is None
    assert swapped is True
    assert victim.read_text(encoding="utf-8") == '{"status":"attacker"}\n'


def test_subprocess_activation_is_opt_in_and_process_activation_rolls_back(
    tmp_path, monkeypatch
):
    source = _fake_source_tree(tmp_path / "aitk")
    identity = _identity(tmp_path / "identity.json")
    status = tmp_path / "run" / ".lds-state" / "bridge-status.json"
    base = {"PYTHONPATH": "old-path", "UNCHANGED": "yes"}
    env = activation.subprocess_environment(
        aitoolkit_dir=source,
        status_file=status,
        identity_file=identity,
        base=base,
    )
    assert base == {"PYTHONPATH": "old-path", "UNCHANGED": "yes"}
    assert env[ENV_ENABLE] == "1"
    assert env[ENV_PROTOCOL] == str(BRIDGE_PROTOCOL)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(activation.BRIDGE_DIR)
    assert activation.deactivate_environment(env) == base

    monkeypatch.setenv("BRIDGE_ROLLBACK_SENTINEL", "before")
    before = dict(os.environ)
    with pytest.raises(RuntimeError):
        with activation.temporary_activation(
            aitoolkit_dir=source,
            status_file=status,
            identity_file=identity,
        ):
            assert os.environ[ENV_ENABLE] == "1"
            raise RuntimeError("exercise finally")
    assert dict(os.environ) == before


def test_environment_overlay_refuses_preplanted_status_symlink(
    tmp_path,
):
    source = _fake_source_tree(tmp_path / "aitk-overlay-final")
    identity = _identity(tmp_path / "identity-overlay-final.json")
    victim = tmp_path / "victim.json"
    victim.write_text('{"keep":"victim"}\n', encoding="utf-8")
    status = tmp_path / "status-link.json"
    try:
        status.symlink_to(victim)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ValueError, match="status_file.*link or reparse"):
        activation.environment_overlay(
            aitoolkit_dir=source,
            status_file=status,
            identity_file=identity,
        )

    assert victim.read_text(encoding="utf-8") == '{"keep":"victim"}\n'


def test_environment_overlay_refuses_linked_status_parent(
    tmp_path,
):
    source = _fake_source_tree(tmp_path / "aitk-overlay-parent")
    identity = _identity(tmp_path / "identity-overlay-parent.json")
    outside = tmp_path / "outside-status"
    outside.mkdir()
    linked_parent = tmp_path / "linked-status-parent"
    _make_directory_link(linked_parent, outside)
    try:
        with pytest.raises(ValueError, match="status_file.*link or reparse"):
            activation.environment_overlay(
                aitoolkit_dir=source,
                status_file=linked_parent / "bridge-status.json",
                identity_file=identity,
            )

        assert not (outside / "bridge-status.json").exists()
    finally:
        _remove_directory_link(linked_parent)


def test_environment_overlay_refuses_linked_restore_directory(
    tmp_path,
):
    source = _fake_source_tree(tmp_path / "aitk-overlay-restore")
    identity = _identity(tmp_path / "identity-overlay-restore.json")
    outside = tmp_path / "outside-bundle"
    outside.mkdir()
    restore = tmp_path / "restore-link"
    _make_directory_link(restore, outside)
    try:
        with pytest.raises(ValueError, match="restore_dir.*link or reparse"):
            activation.environment_overlay(
                aitoolkit_dir=source,
                status_file=tmp_path / "bridge-status.json",
                identity_file=identity,
                restore_dir=restore,
            )
    finally:
        _remove_directory_link(restore)


@_requires_torch_runtime
def test_capture_disk_preflight_fails_before_runtime_workdir(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    trainer = _Trainer(tmp_path / "preflight-run")
    runtime.instrument_dataloader(trainer.data_loader, "main")
    next(iter(trainer.data_loader))

    def reject_preflight(*_args, **_kwargs):
        raise bundles.InsufficientSpaceError("insufficient_free_space")

    monkeypatch.setattr(bundles, "preflight_bundle_space", reject_preflight)
    monkeypatch.setattr(runtime, "_bundle_core", lambda: bundles)
    with pytest.raises(runtime.StateBridgeError, match="disk preflight"):
        runtime.capture_exact_bundle(trainer, 3)

    assert not list(Path(trainer.save_root).glob(".lds-bridge-*"))
    assert bundles.list_bundles(trainer.save_root) == []


@_requires_torch_runtime
def test_restore_disk_preflight_fails_before_runtime_workdir_or_copy(
    tmp_path, monkeypatch
):
    runtime._clear_early_staging()
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    trainer = _Trainer(tmp_path / "restore-preflight-run")
    runtime.instrument_dataloader(trainer.data_loader, "main")
    next(iter(trainer.data_loader))
    bundle = runtime.capture_exact_bundle(trainer, 3)
    inspection = bundles.verify_bundle(trainer.save_root, bundle.name)
    reserve = 4096
    free = inspection.size_bytes * 2 + reserve - 1
    monkeypatch.setenv(ENV_RESERVE_BYTES, str(reserve))
    monkeypatch.setattr(runtime, "_bundle_core", lambda: bundles)
    monkeypatch.setattr(
        bundles.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free, used=0, free=free),
    )

    def forbidden_workdir(_save_root):
        raise AssertionError("runtime workdir was created before disk preflight")

    monkeypatch.setattr(runtime, "_runtime_work_directory", forbidden_workdir)

    with pytest.raises(runtime.StateBridgeError, match="restore disk preflight"):
        runtime._prepare_early_dataset_restore(bundle)

    assert not list(Path(trainer.save_root).glob(".lds-bridge-*"))
    assert runtime._EARLY_STAGED_ROOT is None
    assert runtime._EARLY_CACHE_ARCHIVE is None


@_requires_torch_runtime
def test_runtime_scavenger_requires_exact_old_name_and_proven_dead_owner(
    tmp_path, monkeypatch
):
    save_root = tmp_path / "runtime-scavenge"
    save_root.mkdir()
    now = time.time()

    def work(name: str, *, pid: int | None, old: bool = True) -> Path:
        path = save_root / name
        path.mkdir()
        (path / "payload.bin").write_bytes(b"state")
        if pid is not None:
            atomic_json_nofollow(
                path / runtime._WORK_OWNER_FILENAME,
                {
                    "schema": runtime._WORK_OWNER_SCHEMA,
                    "pid": pid,
                    "process_started_at_ns": 123,
                    "created_at_ns": 456,
                },
            )
        stamp = now - 100_000 if old else now
        os.utime(path, (stamp, stamp))
        return path

    dead = work(".lds-bridge-" + "a" * 32, pid=111)
    alive = work(".lds-bridge-" + "b" * 32, pid=222)
    unknown = work(".lds-bridge-" + "c" * 32, pid=333)
    recent = work(".lds-bridge-" + "d" * 32, pid=111, old=False)
    ownerless = work(".lds-bridge-" + "e" * 32, pid=None)
    malformed = work(".lds-bridge-not-a-uuid", pid=111)
    monkeypatch.setattr(
        runtime,
        "_pid_liveness",
        lambda pid: {111: "dead", 222: "alive", 333: "unknown"}[pid],
    )

    removed = runtime.scavenge_runtime_workdirs(
        save_root,
        older_than_seconds=24 * 60 * 60,
        now=now,
    )

    assert removed == (dead.name,)
    assert not dead.exists()
    assert alive.exists()
    assert unknown.exists()
    assert recent.exists()
    assert ownerless.exists()
    assert malformed.exists()


@_requires_torch_runtime
def test_exact_cpu_bundle_restores_rng_raw_ema_optimizer_scheduler_and_cursor(
    tmp_path, monkeypatch
):
    random.seed(1234)
    np.random.seed(2345)
    torch.manual_seed(3456)
    identity = _identity(tmp_path / "identity.json")
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(identity))
    monkeypatch.setenv(ENV_AITK_ROOT, str(tmp_path / "unused-aitk"))

    continuous = _Trainer(tmp_path / "continuous")
    runtime.instrument_dataloader(continuous.data_loader, "main")
    iterator = iter(continuous.data_loader)
    next(iterator)
    next(iterator)
    loss = continuous.parameter.square().sum()
    loss.backward()
    continuous.optimizer.step()
    continuous.optimizer.zero_grad(set_to_none=True)
    continuous.lr_scheduler.step()
    continuous.ema.shadow.add_(0.25)

    bundle = runtime.capture_exact_bundle(continuous, 3)
    inspection = bundles.inspect_bundle(
        continuous.save_root, bundle.name, verify=True
    )
    assert inspection.restorable is True
    assert inspection.state_level == "exact"
    assert inspection.completed_step == 3
    assert inspection.next_step == 4
    assert inspection.optimizer_updates_completed == 4
    expected_parameter = continuous.parameter.detach().clone()
    expected_ema = continuous.ema.shadow.clone()
    expected_next_batch = next(iterator)[0].clone()
    expected_random = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    resumed = _Trainer(tmp_path / "resumed")
    resumed.parameter.data.fill_(99)
    resumed.ema.shadow.fill_(-99)
    runtime.restore_exact_bundle(resumed, bundle)
    assert resumed.step_num == resumed.start_step == resumed.train_config.start_step == 4
    assert resumed.last_save_step == 3
    assert torch.equal(resumed.parameter, expected_parameter)
    assert torch.equal(resumed.ema.shadow, expected_ema)

    scheduler_epoch = resumed.lr_scheduler.last_epoch
    resumed.lr_scheduler.step(resumed.step_num)  # skipped upstream bootstrap
    assert resumed.lr_scheduler.last_epoch == scheduler_epoch
    resumed_iterator = iter(resumed.data_loader)
    assert torch.equal(next(resumed_iterator)[0], expected_next_batch)
    actual_random = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])


@_requires_torch_runtime
def test_exact_capture_refuses_non_boundary_and_worker_prefetch(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    trainer = _Trainer(tmp_path / "run")
    trainer._lds_last_optimizer_boundary_step = None
    with pytest.raises(runtime.StateBridgeError, match="optimizer boundary"):
        runtime.capture_exact_bundle(trainer, 3)
    assert bundles.list_bundles(trainer.save_root) == []

    worker_loader = DataLoader(
        TensorDataset(torch.arange(2)), batch_size=1, shuffle=True, num_workers=1
    )
    with pytest.raises(runtime.StateBridgeError, match="num_workers=1"):
        runtime.instrument_dataloader(worker_loader, "main")


@_requires_torch_runtime
def test_exact_capture_refuses_non_bucket_random_crop(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    trainer = _Trainer(tmp_path / "random-crop")
    trainer.data_loader.dataset.dataset_config = SimpleNamespace(
        buckets=False,
        random_crop=True,
    )
    runtime.instrument_dataloader(trainer.data_loader, "main")

    with pytest.raises(runtime.StateBridgeError, match="unrecognised mutable"):
        runtime.capture_exact_bundle(trainer, 3)
    assert bundles.list_bundles(trainer.save_root) == []


@_requires_torch_runtime
def test_bucket_state_and_exact_cache_bytes_restore_before_fresh_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_EARLY_DATASET_RESTORE_QUEUE", [])
    monkeypatch.setattr(runtime, "_EARLY_CACHE_ARCHIVE", None)
    monkeypatch.setattr(runtime, "_EARLY_CACHE_DESCRIPTORS", {})
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    original_dataset = _BucketDataset(
        tmp_path / "export-original",
        ("alpha", "beta", "gamma"),
        saved_layout=True,
    )
    original = _Trainer(tmp_path / "bucket-original-run")
    original.data_loader = DataLoader(
        original_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
    )
    runtime.instrument_dataloader(original.data_loader, "main")
    next(iter(original.data_loader))
    bundle = runtime.capture_exact_bundle(original, 3)
    inspection = bundles.verify_bundle(original.save_root, bundle.name)
    assert "preprocessing-cache-bytes" in inspection.capabilities
    assert any(
        artifact.name == "latent_cache.tar"
        for artifact in inspection.artifacts
    )

    monkeypatch.setattr(
        runtime, "_ORIGINAL_AITK_SETUP_BUCKETS", _fake_setup_buckets
    )
    monkeypatch.setattr(
        runtime, "_ORIGINAL_AITK_CACHE_LATENTS", _fake_cache_latents
    )
    monkeypatch.setattr(
        runtime, "_ORIGINAL_AITK_CACHE_TEXT", _fake_cache_text
    )
    monkeypatch.setattr(
        runtime, "_ORIGINAL_AITK_SETUP_EPOCH", _fake_setup_epoch
    )
    monkeypatch.setenv(
        "LDS_AITK_STATE_RESTORE_DIR",
        str(bundle),
    )
    runtime._prepare_early_dataset_restore(bundle)
    assert runtime._EARLY_STAGED_ROOT is not None
    assert runtime._EARLY_CACHE_ARCHIVE is not None
    assert runtime._EARLY_CACHE_ARCHIVE.is_relative_to(
        runtime._EARLY_STAGED_ROOT
    )

    # Verification and late deserialisation must not be separated by reads
    # from the caller-controlled bundle.  Corrupt both a tensor artifact and
    # the cache tar after early staging; the retained private copy remains the
    # sole restore source.
    artifact_paths = {
        artifact.name: bundle / artifact.path
        for artifact in inspection.artifacts
    }
    artifact_paths["optimizer.pt"].write_bytes(b"mutated-after-verification")
    artifact_paths["latent_cache.tar"].write_bytes(b"mutated-cache-tar")

    # A fresh export may enumerate differently and has no cache directories.
    resumed_dataset = _BucketDataset(
        tmp_path / "export-resumed",
        ("gamma", "alpha", "beta"),
        saved_layout=False,
    )
    random.seed(9999)
    runtime._patched_setup_epoch(resumed_dataset)

    assert [item.token for item in resumed_dataset.file_list] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert list(resumed_dataset.buckets) == ["8x8"]
    assert resumed_dataset.buckets["8x8"].file_list_idx == [2, 0, 1]
    assert resumed_dataset.batch_indices == [[2, 0], [1, 1]]
    assert resumed_dataset.epoch_num == 3
    assert [item.loaded_latent_bytes for item in resumed_dataset.file_list] == [
        b"latent:alpha",
        b"latent:beta",
        b"latent:gamma",
    ]
    assert [item.loaded_text_bytes for item in resumed_dataset.file_list] == [
        b"text:alpha",
        b"text:beta",
        b"text:gamma",
    ]

    resumed = _Trainer(tmp_path / "bucket-resumed-run")
    resumed.data_loader = DataLoader(
        resumed_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
    )
    runtime.restore_exact_bundle(resumed, bundle)
    assert resumed.step_num == 4


@_requires_torch_runtime
def test_linear_cached_dataset_runs_epoch_zero_cache_hooks_before_saved_epoch(
    tmp_path, monkeypatch
):
    runtime._clear_early_staging()
    monkeypatch.setattr(runtime, "_EARLY_DATASET_RESTORE_QUEUE", [])
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    original_dataset = _LinearDataset(
        tmp_path / "linear-original",
        ("alpha", "beta", "gamma"),
        saved_layout=True,
    )
    original = _Trainer(tmp_path / "linear-original-run")
    original.data_loader = DataLoader(
        original_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    runtime.instrument_dataloader(original.data_loader, "main")
    next(iter(original.data_loader))
    bundle = runtime.capture_exact_bundle(original, 3)

    calls = {"latent": 0, "text": 0}

    def cache_latents(dataset):
        calls["latent"] += 1
        _fake_cache_latents(dataset)

    def cache_text(dataset):
        calls["text"] += 1
        _fake_cache_text(dataset)

    monkeypatch.setattr(runtime, "_ORIGINAL_AITK_SETUP_BUCKETS", _fake_setup_buckets)
    monkeypatch.setattr(runtime, "_ORIGINAL_AITK_CACHE_LATENTS", cache_latents)
    monkeypatch.setattr(runtime, "_ORIGINAL_AITK_CACHE_TEXT", cache_text)
    monkeypatch.setattr(runtime, "_ORIGINAL_AITK_SETUP_EPOCH", _fake_setup_epoch)
    monkeypatch.setenv("LDS_AITK_STATE_RESTORE_DIR", str(bundle))
    runtime._prepare_early_dataset_restore(bundle)

    resumed_dataset = _LinearDataset(
        tmp_path / "linear-resumed",
        ("gamma", "alpha", "beta"),
        saved_layout=False,
    )
    runtime._patched_setup_epoch(resumed_dataset)

    assert calls == {"latent": 1, "text": 1}
    assert resumed_dataset.epoch_num == 3
    assert [item.token for item in resumed_dataset.file_list] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert [item.loaded_latent_bytes for item in resumed_dataset.file_list] == [
        b"latent:alpha",
        b"latent:beta",
        b"latent:gamma",
    ]
    assert [item.loaded_text_bytes for item in resumed_dataset.file_list] == [
        b"text:alpha",
        b"text:beta",
        b"text:gamma",
    ]

    resumed = _Trainer(tmp_path / "linear-resumed-run")
    resumed.data_loader = DataLoader(
        resumed_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    runtime.restore_exact_bundle(resumed, bundle)
    assert resumed.step_num == 4
    assert runtime._EARLY_STAGED_ROOT is None
    assert runtime._EARLY_CACHE_ARCHIVE is None


@_requires_torch_runtime
def test_existing_stream_late_state_load_does_not_draw_a_new_global_seed():
    torch.manual_seed(4321)
    original = DataLoader(
        TensorDataset(torch.arange(12)),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    original_stream = runtime.instrument_dataloader(original, "main")
    original_iterator = iter(original)
    next(original_iterator)
    saved = original_stream.state_dict()
    expected_next = next(original_iterator)[0].clone()

    resumed = DataLoader(
        TensorDataset(torch.arange(12)),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    existing = runtime.instrument_dataloader(resumed, "main")
    assert existing.seed_from_global_on_first_iter is True
    assert runtime.instrument_dataloader(resumed, "main", saved) is existing
    assert existing.seed_from_global_on_first_iter is False

    torch.manual_seed(999)
    before = torch.get_rng_state().clone()
    actual_next = next(iter(resumed))[0]
    assert torch.equal(actual_next, expected_next)
    assert torch.equal(torch.get_rng_state(), before)


@_requires_torch_runtime
def test_sampler_restore_preserves_exhaustion_epoch_transition():
    torch.manual_seed(7123)
    original = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    stream = runtime.instrument_dataloader(original, "main")
    iterator = iter(original)
    for _ in range(len(original)):
        next(iterator)
    state = stream.state_dict()
    assert state["exhaustion_pending"] is True
    global_rng = torch.get_rng_state().clone()

    # The uninterrupted loop first observes StopIteration on the exhausted
    # iterator, advances its dataset epoch, then creates the next iterator.
    with pytest.raises(StopIteration):
        next(iterator)
    uninterrupted = [batch[0].clone() for batch in original]

    resumed = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    runtime.instrument_dataloader(resumed, "main", state)
    torch.set_rng_state(global_rng)
    exhausted_transition = iter(resumed)
    with pytest.raises(StopIteration):
        next(exhausted_transition)
    resumed_next_epoch = [batch[0].clone() for batch in resumed]

    assert len(resumed_next_epoch) == len(uninterrupted)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(resumed_next_epoch, uninterrupted)
    )


@_requires_torch_runtime
def test_dataset_content_or_mask_change_fails_before_cache_materialization(tmp_path):
    original = _BucketDataset(
        tmp_path / "identity-original",
        ("alpha", "beta", "gamma"),
        saved_layout=True,
    )
    saved = runtime._capture_aitk_dataset_state(
        original,
        label="main",
        leaf_index=0,
        cache_sources={},
    )
    changed = _BucketDataset(
        tmp_path / "identity-changed",
        ("gamma", "alpha", "beta"),
        saved_layout=False,
    )
    Path(changed.file_list[1].mask_path).write_bytes(b"different-mask-bytes")

    with pytest.raises(runtime.StateBridgeError, match="content multiset changed"):
        runtime._restore_aitk_dataset_state(
            changed,
            saved,
            label="main",
            leaf_index=0,
            bucket_objects_ready=False,
        )
    assert not (tmp_path / "identity-changed" / "_latent_cache").exists()
    assert not (tmp_path / "identity-changed" / "_t_e_cache").exists()


@_requires_torch_runtime
def test_cached_dataset_rejects_missing_saved_cache_descriptors(tmp_path):
    original = _BucketDataset(
        tmp_path / "cache-original",
        ("alpha", "beta", "gamma"),
        saved_layout=True,
    )
    saved = runtime._capture_aitk_dataset_state(
        original,
        label="main",
        leaf_index=0,
        cache_sources={},
    )
    incomplete = copy.deepcopy(saved)
    for file_state in incomplete["files"]:
        file_state["cache"] = {}
    fresh = _BucketDataset(
        tmp_path / "cache-fresh",
        ("gamma", "alpha", "beta"),
        saved_layout=False,
    )

    with pytest.raises(runtime.StateBridgeError, match="saved cache state is invalid"):
        runtime._restore_aitk_dataset_state(
            fresh,
            incomplete,
            label="main",
            leaf_index=0,
            bucket_objects_ready=False,
        )
    assert not (tmp_path / "cache-fresh" / "_latent_cache").exists()


@_requires_torch_runtime
def test_cache_materialization_refuses_linked_cache_parent_without_external_write(
    tmp_path, monkeypatch
):
    original = _BucketDataset(
        tmp_path / "cache-link-original",
        ("alpha", "beta", "gamma"),
        saved_layout=True,
    )
    cache_sources = {}
    saved = runtime._capture_aitk_dataset_state(
        original,
        label="main",
        leaf_index=0,
        cache_sources=cache_sources,
    )
    archive = tmp_path / "cache-link.tar"
    runtime._write_cache_archive(archive, cache_sources)
    monkeypatch.setattr(runtime, "_EARLY_CACHE_ARCHIVE", archive)
    monkeypatch.setattr(
        runtime,
        "_EARLY_CACHE_DESCRIPTORS",
        runtime._cache_descriptors(saved),
    )

    fresh_root = tmp_path / "cache-link-fresh"
    fresh = _BucketDataset(
        fresh_root,
        ("alpha", "beta", "gamma"),
        saved_layout=False,
    )
    outside = tmp_path / "outside-cache"
    outside.mkdir()
    linked_cache = fresh_root / "_latent_cache"
    _make_directory_link(linked_cache, outside)
    try:
        with pytest.raises(
            runtime.StateBridgeError, match="link or reparse point"
        ):
            runtime._materialize_saved_cache_kind(
                fresh,
                saved,
                kind="latent",
            )
        assert list(outside.iterdir()) == []
    finally:
        _remove_directory_link(linked_cache)


@_requires_torch_runtime
def test_cache_destination_refuses_linked_dataset_ancestor(tmp_path):
    outside = tmp_path / "outside-dataset"
    outside.mkdir()
    (outside / "image.png").write_bytes(b"image")
    linked_dataset = tmp_path / "linked-dataset"
    _make_directory_link(linked_dataset, outside)
    item = SimpleNamespace(path=str(linked_dataset / "image.png"))
    item.get_latent_path = lambda recalculate=False: str(
        linked_dataset / "_latent_cache" / "image.safetensors"
    )
    try:
        with pytest.raises(
            runtime.StateBridgeError, match="link or reparse point"
        ):
            runtime._safe_cache_destination(item, "latent")
        assert not (outside / "_latent_cache").exists()
    finally:
        _remove_directory_link(linked_dataset)


@_requires_torch_runtime
def test_preprocessing_cache_archive_rejects_traversal_members(tmp_path):
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        payload = b"escape"
        member = tarfile.TarInfo("../outside.safetensors")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with tarfile.open(archive_path, mode="r:") as archive:
        with pytest.raises(runtime.StateBridgeError, match="archive is unsafe"):
            runtime._validated_cache_members(archive, {})
    assert not (tmp_path.parent / "outside.safetensors").exists()


@_requires_torch_runtime
def test_terminal_bundle_uses_equal_completed_and_next_step(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_IDENTITY_FILE, str(_identity(tmp_path / "identity.json")))
    trainer = _Trainer(tmp_path / "terminal")
    runtime.instrument_dataloader(trainer.data_loader, "main")
    iterator = iter(trainer.data_loader)
    next(iterator)
    trainer.step_num = 4
    trainer._lds_last_optimizer_boundary_step = 3
    trainer.checkpoint = Path(trainer.save_root) / "job.safetensors"
    trainer.checkpoint.write_bytes(b"final-public-checkpoint")

    bundle = runtime.capture_exact_bundle(
        trainer,
        4,
        next_step=4,
        optimizer_boundary_step=3,
        numbered_checkpoint=False,
    )
    inspection = bundles.inspect_bundle(trainer.save_root, bundle.name, verify=True)
    assert inspection.restorable
    assert inspection.completed_step == inspection.next_step == 4
