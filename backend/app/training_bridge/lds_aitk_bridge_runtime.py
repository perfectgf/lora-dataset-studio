"""In-process save/restore seam for the inspected ai-toolkit lifecycle.

This file is imported as a top-level module by ``sitecustomize`` inside the
ai-toolkit interpreter.  Keep application/Flask imports out of this module.
The only LDS service loaded dynamically is the pure-stdlib canonical bundle
writer/inspector.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import random
import re
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from contextlib import contextmanager
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

try:
    from lds_aitk_bridge_contract import (
        ARTIFACT_FILENAMES,
        BRIDGE_NAME,
        BRIDGE_PROTOCOL,
        BRIDGE_VERSION,
        ENV_AITK_ROOT,
        ENV_IDENTITY_FILE,
        ENV_KEEP,
        ENV_MAX_BUNDLE_BYTES,
        ENV_MAX_STORE_BYTES,
        ENV_PROTOCOL,
        ENV_RESERVE_BYTES,
        ENV_RESTORE_DIR,
        ENV_STATUS_FILE,
        ENV_STRICT,
        SHAPE_REVISION,
        STATE_SCHEMA,
        atomic_json_nofollow,
        load_identity,
        probe_aitoolkit_source,
        read_json_nofollow,
    )
except ImportError:  # Package import used by unit tests.
    from .lds_aitk_bridge_contract import (  # type: ignore
        ARTIFACT_FILENAMES,
        BRIDGE_NAME,
        BRIDGE_PROTOCOL,
        BRIDGE_VERSION,
        ENV_AITK_ROOT,
        ENV_IDENTITY_FILE,
        ENV_KEEP,
        ENV_MAX_BUNDLE_BYTES,
        ENV_MAX_STORE_BYTES,
        ENV_PROTOCOL,
        ENV_RESERVE_BYTES,
        ENV_RESTORE_DIR,
        ENV_STATUS_FILE,
        ENV_STRICT,
        SHAPE_REVISION,
        STATE_SCHEMA,
        atomic_json_nofollow,
        load_identity,
        probe_aitoolkit_source,
        read_json_nofollow,
    )


class StateBridgeError(RuntimeError):
    """The requested state cannot honestly be represented as exact."""


_INSTALLED = False
_SOURCE_PROBE: dict[str, Any] = {}
_ORIGINAL_DATALOADER_ITER = DataLoader.__iter__
_ORIGINAL_BASE_SAVE = None
_ORIGINAL_BASE_END_STEP = None
_ORIGINAL_SD_BEFORE_LOOP = None
_ORIGINAL_SD_TRAIN_LOOP = None
_ORIGINAL_AITK_SETUP_BUCKETS = None
_ORIGINAL_AITK_SETUP_EPOCH = None
_ORIGINAL_AITK_CACHE_LATENTS = None
_ORIGINAL_AITK_CACHE_TEXT = None
_EARLY_DATASET_RESTORE_QUEUE: list[Mapping[str, Any]] = []
_EARLY_CACHE_ARCHIVE: Path | None = None
_EARLY_CACHE_DESCRIPTORS: dict[str, Mapping[str, Any]] = {}
_EARLY_STAGED_ROOT: Path | None = None
_EARLY_STAGED_SOURCE: Path | None = None
_EARLY_STAGED_MANIFEST: Mapping[str, Any] | None = None
_EARLY_STAGED_PATHS: dict[str, Path] = {}
_EARLY_STAGE_OWNER: Any | None = None

_WORK_OWNER_FILENAME = "ACTIVE.json"
_WORK_OWNER_SCHEMA = "lds.aitoolkit-work-owner/v1"
_WORK_PREFIX = ".lds-bridge-"
_WORK_RE = re.compile(r"^\.lds-bridge-[0-9a-f]{32}$")
_WORK_MAX_AGE_SECONDS = 24 * 60 * 60
_PROCESS_STARTED_AT_NS = time.time_ns()

_DATASET_GRAPH_SCHEMA = "lds.aitoolkit-dataset-graph/v2"
_AITK_DATASET_SCHEMA = "lds.aitoolkit-dataset-state/v2"
_SAMPLER_SCHEMA = "lds.aitoolkit-replayable-sampler/v2"
_DATALOADER_STATE_SCHEMA = "lds.aitoolkit-dataloader-state/v2"
_CACHE_ARCHIVE_SCHEMA = "lds.aitoolkit-preprocess-cache/v1"
_RUNTIME_CAPABILITIES = (
    "dataloader-dataset-state",
    "dataloader-order-cursor",
    "deterministic-latent-cache",
    "ema",
    "optimizer",
    "preprocessing-cache-bytes",
    "raw-weights",
    "rng-cuda",
    "rng-numpy",
    "rng-python",
    "rng-torch",
    "scaler",
    "scheduler",
)
_GEOMETRY_FIELDS = (
    "scale_to_width",
    "scale_to_height",
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("atomic bridge JSON must be an object")
    atomic_json_nofollow(path, value)


def _status_path(trainer: Any | None = None) -> Path | None:
    configured = os.environ.get(ENV_STATUS_FILE)
    if configured:
        return Path(configured)
    if trainer is not None and getattr(trainer, "save_root", None):
        return Path(trainer.save_root) / ".lds-state" / "bridge-status.json"
    return None


def _write_status(status: str, *, trainer: Any | None = None, **values: Any) -> None:
    path = _status_path(trainer)
    if path is None:
        return
    payload = {
        "bridge": BRIDGE_NAME,
        "bridge_version": BRIDGE_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "shape_revision": SHAPE_REVISION,
        "status": status,
        "updated_at": _utc_now(),
        "source_probe": _SOURCE_PROBE,
    }
    if trainer is not None and bool(
        getattr(trainer, "_lds_training_started_reported", False)
    ):
        # Sticky evidence used by LDS to distinguish a failed restore/bootstrap
        # from a later training crash. Subsequent status writes must retain it.
        payload["training_started"] = True
    payload.update(values)
    try:
        _atomic_json(path, payload)
    except (OSError, TypeError, ValueError):
        # The status channel must never make a weights save fail.
        pass


def _bundle_core():
    """Load the canonical stdlib bundle implementation without importing Flask."""
    try:
        import training_state_bundle

        return training_state_bundle
    except ImportError:
        pass
    path = Path(__file__).resolve().parent.parent / "services" / "training_state_bundle.py"
    if not path.is_file():
        raise StateBridgeError(f"canonical bundle backend unavailable: {path}")
    name = "_lds_training_state_bundle_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StateBridgeError(f"cannot load canonical bundle backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    for attr in ("BundleMetadata", "create_bundle", "inspect_bundle"):
        if not hasattr(module, attr):
            raise StateBridgeError(f"bundle backend has no {attr}")
    return module


def _identity() -> dict[str, Any]:
    path = os.environ.get(ENV_IDENTITY_FILE)
    if not path:
        raise StateBridgeError(
            f"{ENV_IDENTITY_FILE} is required before exact state can be published"
        )
    value = load_identity(path)
    # Unit-level capture helpers can run without installing the patch. In a real
    # subprocess `_SOURCE_PROBE` is populated first, so independently re-check
    # both source and runtime instead of trusting the Flask preflight across a
    # process boundary.
    if _SOURCE_PROBE:
        actual_revision = _SOURCE_PROBE.get("aitoolkit_revision")
        if value.get("toolkit_revision") != actual_revision:
            raise StateBridgeError(
                "ai-toolkit revision changed after the LDS compatibility preflight"
            )
        expected_runtime = value.get("runtime") or {}
        actual_runtime = _current_runtime_identity()
        differences = [
            key
            for key, actual in actual_runtime.items()
            if expected_runtime.get(key) != actual
        ]
        if differences:
            raise StateBridgeError(
                "training runtime changed after the LDS compatibility preflight: "
                + ", ".join(sorted(differences))
            )
    return value


def _current_runtime_identity() -> dict[str, Any]:
    grouped: dict[str, set[str]] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = re.sub(r"[-_.]+", "-", str(raw_name)).lower()
        version = str(distribution.version or "")
        if not name or not version:
            raise StateBridgeError("runtime package identity is incomplete")
        grouped.setdefault(name, set()).add(version)
    packages = {
        name: "|".join(sorted(versions))
        for name, versions in sorted(grouped.items())
    }
    required_packages = {
        "numpy",
        "accelerate",
        "diffusers",
        "transformers",
        "safetensors",
        "bitsandbytes",
    }
    if not required_packages.issubset(packages):
        missing = ", ".join(sorted(required_packages - set(packages)))
        raise StateBridgeError(f"runtime packages are unavailable: {missing}")
    count = int(torch.cuda.device_count())
    gpus = [
        {
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
            "compute_capability": ".".join(
                str(part) for part in torch.cuda.get_device_capability(index)
            ),
        }
        for index in range(count)
    ]
    driver = "none"
    if count:
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StateBridgeError("GPU driver identity is unavailable") from exc
        values = [
            line.strip()
            for line in (process.stdout or "").splitlines()
            if line.strip()
        ]
        if process.returncode != 0 or len(values) != count:
            raise StateBridgeError("GPU driver identity is incomplete")
        driver = "|".join(values)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version() or ""),
        "cuda_devices": count,
        "gpus": gpus,
        "gpu_driver": driver,
        "packages": packages,
        "protocol": BRIDGE_PROTOCOL,
        "shape_revision": SHAPE_REVISION,
    }


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _rng_json_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": np_state[0],
            "keys": np_state[1].tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
    }


def _restore_rng_json(value: Mapping[str, Any]) -> None:
    random.setstate(_tuple_tree(value["python"]))
    np_value = value["numpy"]
    np.random.set_state(
        (
            str(np_value["bit_generator"]),
            np.asarray(np_value["keys"], dtype=np.uint32),
            int(np_value["position"]),
            int(np_value["has_gauss"]),
            float(np_value["cached_gaussian"]),
        )
    )


def _torch_rng_state(require_cuda: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "cpu": torch.get_rng_state().cpu(),
        "cuda_applicable": bool(require_cuda),
        "cuda": [],
    }
    if require_cuda:
        if not torch.cuda.is_available():
            raise StateBridgeError("CUDA trainer has no available CUDA RNG")
        value["cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return value


def _restore_torch_rng(value: Mapping[str, Any], require_cuda: bool) -> None:
    torch.set_rng_state(value["cpu"].cpu())
    saved_cuda = bool(value.get("cuda_applicable"))
    if require_cuda != saved_cuda:
        raise StateBridgeError(
            f"CUDA RNG applicability changed (saved={saved_cuda}, current={require_cuda})"
        )
    if require_cuda:
        states = list(value.get("cuda") or [])
        if len(states) != torch.cuda.device_count():
            raise StateBridgeError(
                "CUDA device count differs from the exact checkpoint "
                f"({len(states)} saved, {torch.cuda.device_count()} current)"
            )
        torch.cuda.set_rng_state_all(states)


def _json_index(value: Any) -> int | list[Any]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_json_index(item) for item in value]
    raise StateBridgeError(
        f"unsupported sampler index {type(value).__name__}; exact replay disabled"
    )


class ReplayableIndexStream:
    """Materialise one RandomSampler epoch and persist the consumed cursor."""

    def __init__(
        self,
        source: Iterable[Any],
        *,
        label: str,
        sampler_type: str,
        loader_generator: torch.Generator,
        seed_from_global_on_first_iter: bool = False,
    ):
        self.source = source
        self.label = label
        self.sampler_type = sampler_type
        self.loader_generator = loader_generator
        self.seed_from_global_on_first_iter = seed_from_global_on_first_iter
        self.order: list[Any] | None = None
        self.consumed = 0
        self.dispatched = 0
        self.epoch_ordinal = -1
        self.exhaustion_pending = False

    def __len__(self) -> int:
        return len(self.source)  # type: ignore[arg-type]

    def __iter__(self):
        if self.exhaustion_pending:
            # The uninterrupted iterator would raise StopIteration once here.
            # BaseSDTrainProcess observes that edge to advance dataset/trainer
            # epoch state before creating the next iterator.
            self.exhaustion_pending = False
            return
        if self.order is None or self.consumed >= len(self.order):
            self.order = [_json_index(item) for item in self.source]
            self.consumed = 0
            self.dispatched = 0
            self.epoch_ordinal += 1
        else:
            # Dispatched-but-not-consumed work must be replayed after a crash.
            self.dispatched = self.consumed
        while self.dispatched < len(self.order):
            value = self.order[self.dispatched]
            self.dispatched += 1
            yield copy.deepcopy(value)

    def mark_consumed(self) -> None:
        if self.order is None or self.consumed >= len(self.order):
            raise StateBridgeError(f"{self.label} cursor advanced outside its order")
        self.consumed += 1

    def state_dict(self) -> dict[str, Any]:
        if self.order is None:
            raise StateBridgeError(f"{self.label} sampler order has not started")
        return {
            "schema": _SAMPLER_SCHEMA,
            "label": self.label,
            "sampler_type": self.sampler_type,
            "epoch_ordinal": self.epoch_ordinal,
            "order": self.order,
            "cursor": self.consumed,
            "exhaustion_pending": self.consumed == len(self.order),
            # DataLoader draws a base seed even with num_workers=0.  Giving it
            # a dedicated persisted generator prevents iterator recreation on
            # resume from advancing the global torch RNG by two uint32 draws.
            "loader_generator_state": self.loader_generator.get_state().tolist(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema") != _SAMPLER_SCHEMA:
            raise StateBridgeError(f"unsupported sampler state for {self.label}")
        if value.get("label") != self.label:
            raise StateBridgeError(
                f"sampler label differs ({value.get('label')!r} != {self.label!r})"
            )
        if value.get("sampler_type") != self.sampler_type:
            raise StateBridgeError(f"sampler type changed for {self.label}")
        order = value.get("order")
        cursor = value.get("cursor")
        generator_state = value.get("loader_generator_state")
        if (
            not isinstance(order, list)
            or not isinstance(cursor, int)
            or not isinstance(generator_state, list)
        ):
            raise StateBridgeError(f"invalid sampler order/cursor for {self.label}")
        if cursor < 0 or cursor > len(order):
            raise StateBridgeError(f"sampler cursor is out of range for {self.label}")
        self.order = [_json_index(item) for item in order]
        self.consumed = cursor
        self.dispatched = cursor
        self.epoch_ordinal = int(value.get("epoch_ordinal", 0))
        exhaustion_pending = value.get("exhaustion_pending")
        if not isinstance(exhaustion_pending, bool):
            raise StateBridgeError(
                f"invalid sampler exhaustion transition for {self.label}"
            )
        if exhaustion_pending != (cursor == len(order)):
            raise StateBridgeError(
                f"sampler exhaustion transition disagrees with cursor for {self.label}"
            )
        self.exhaustion_pending = exhaustion_pending
        try:
            self.loader_generator.set_state(
                torch.tensor(generator_state, dtype=torch.uint8)
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise StateBridgeError(
                f"invalid DataLoader generator state for {self.label}"
            ) from exc
        # A stream can be instrumented during normal trainer setup and receive
        # its persisted state only later in hook_before_train_loop.  In that
        # two-stage path the constructor left this flag true.  The saved
        # generator is already authoritative: drawing a fresh seed on the first
        # resumed iterator would both overwrite it and advance global torch RNG.
        self.seed_from_global_on_first_iter = False


class _TrackedIterator:
    def __init__(self, inner: Any, stream: ReplayableIndexStream):
        self.inner = inner
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.inner)
        self.stream.mark_consumed()
        return value


def _tracked_dataloader_iter(loader: DataLoader):
    stream = getattr(loader, "_lds_replay_stream", None)
    if stream is not None and stream.exhaustion_pending:
        # Do not construct a real DataLoader iterator: that would consume an
        # extra loader-generator base seed compared with the uninterrupted run.
        stream.exhaustion_pending = False
        return _TrackedIterator(iter(()), stream)
    if stream is not None and stream.seed_from_global_on_first_iter:
        # Preserve unpatched DataLoader timing: its base seed normally advances
        # the global torch RNG at iterator creation, not at trainer setup.
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        stream.loader_generator.manual_seed(seed)
        stream.seed_from_global_on_first_iter = False
    inner = _ORIGINAL_DATALOADER_ITER(loader)
    return _TrackedIterator(inner, stream) if stream is not None else inner


def instrument_dataloader(
    loader: DataLoader | None,
    label: str,
    restore_state: Mapping[str, Any] | None = None,
) -> ReplayableIndexStream | None:
    if DataLoader.__iter__ is not _tracked_dataloader_iter:
        DataLoader.__iter__ = _tracked_dataloader_iter
    if loader is None:
        if restore_state is not None:
            raise StateBridgeError(f"saved {label} dataloader is absent")
        return None
    existing = getattr(loader, "_lds_replay_stream", None)
    if existing is not None:
        if restore_state is not None:
            existing.load_state_dict(restore_state)
        return existing
    if int(getattr(loader, "num_workers", 0)) != 0:
        raise StateBridgeError(
            f"{label} dataloader uses num_workers={loader.num_workers}; "
            "worker RNG/prefetch state is not exactly restorable"
        )

    auto_collation = bool(getattr(loader, "_auto_collation", False))
    source = loader.batch_sampler if auto_collation else loader.sampler
    sampler = getattr(source, "sampler", source)
    if not isinstance(sampler, RandomSampler):
        raise StateBridgeError(
            f"{label} sampler {type(sampler).__module__}.{type(sampler).__qualname__} "
            "is not the recognised RandomSampler"
        )
    sampler_type = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
    loader_generator = getattr(loader, "generator", None)
    if loader_generator is not None and not isinstance(
        loader_generator, torch.Generator
    ):
        raise StateBridgeError(f"{label} DataLoader generator is unrecognised")
    had_loader_generator = loader_generator is not None
    if loader_generator is None:
        loader_generator = torch.Generator()
    stream = ReplayableIndexStream(
        source,
        label=label,
        sampler_type=sampler_type,
        loader_generator=loader_generator,
        seed_from_global_on_first_iter=(
            restore_state is None and not had_loader_generator
        ),
    )
    if restore_state is not None:
        stream.load_state_dict(restore_state)
    object.__setattr__(loader, "generator", loader_generator)
    if auto_collation:
        object.__setattr__(loader, "batch_sampler", stream)
    else:
        object.__setattr__(loader, "sampler", stream)
    object.__setattr__(loader, "_lds_replay_stream", stream)
    return stream


def _optimizer_params(optimizer: Any) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append([param.detach().cpu().clone() for param in group["params"]])
    if not groups or not any(groups):
        raise StateBridgeError("optimizer has no bound parameters")
    return {"param_groups": groups}


def _restore_optimizer_params(optimizer: Any, value: Mapping[str, Any]) -> None:
    saved_groups = value.get("param_groups")
    current_groups = optimizer.param_groups
    if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
        raise StateBridgeError("optimizer parameter-group count changed")
    with torch.no_grad():
        for group_index, (saved, current) in enumerate(
            zip(saved_groups, current_groups)
        ):
            params = current["params"]
            if len(saved) != len(params):
                raise StateBridgeError(
                    f"optimizer parameter count changed in group {group_index}"
                )
            for param_index, (saved_param, param) in enumerate(zip(saved, params)):
                if tuple(saved_param.shape) != tuple(param.shape):
                    raise StateBridgeError(
                        f"raw parameter shape changed at {group_index}:{param_index}"
                    )
                param.copy_(saved_param.to(device=param.device, dtype=param.dtype))


def _trainer_requires_cuda(trainer: Any) -> bool:
    device = getattr(getattr(trainer, "accelerator", None), "device", None)
    return getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"


def _leaf_datasets(value: Any) -> Iterable[Any]:
    """Yield concrete datasets through ai-toolkit/PyTorch container shapes."""
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (list, tuple)):
            pending.extend(reversed(current))
            continue
        children = getattr(current, "datasets", None)
        if isinstance(children, (list, tuple)):
            pending.extend(reversed(children))
            continue
        yield current


def _qualified_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _bounded_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = 2**31 - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise StateBridgeError(f"{label} is not an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise StateBridgeError(f"{label} is out of range")
    return result


def _hash_regular_source(
    path: Any,
    *,
    label: str,
) -> tuple[str, int]:
    """Hash a stable regular file without serialising its machine path."""
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise StateBridgeError(f"{label} has no file")
    source = Path(os.path.abspath(os.fspath(path)))
    if source.is_symlink():
        raise StateBridgeError(f"{label} is a symlink")
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StateBridgeError(f"{label} is not a regular file")
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise StateBridgeError(f"{label} cannot be read: {exc}") from exc
    if (
        size != before.st_size
        or after.st_size != before.st_size
        or getattr(after, "st_mtime_ns", None)
        != getattr(before, "st_mtime_ns", None)
    ):
        raise StateBridgeError(f"{label} changed while it was hashed")
    return digest.hexdigest(), size


def _content_file_descriptor(path: Any, *, label: str) -> dict[str, Any]:
    digest, size = _hash_regular_source(path, label=label)
    return {"kind": "file", "sha256": digest, "size_bytes": size}


def _caption_descriptor(dataset: Any, item: Any, *, label: str) -> dict[str, Any]:
    caption_dict = getattr(dataset, "caption_dict", None)
    item_path = os.fspath(getattr(item, "path", ""))
    if isinstance(caption_dict, Mapping) and item_path in caption_dict:
        try:
            raw = _json_bytes(caption_dict[item_path])
        except (TypeError, ValueError) as exc:
            raise StateBridgeError(f"{label} mapped caption is not canonical") from exc
        return {
            "kind": "mapping",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    config = getattr(dataset, "dataset_config", None)
    extension = getattr(config, "caption_ext", ".txt") if config is not None else ".txt"
    if not isinstance(extension, str):
        raise StateBridgeError(f"{label} caption extension is invalid")
    caption_path = Path(os.path.splitext(item_path)[0] + extension)
    if caption_path.is_file():
        return _content_file_descriptor(caption_path, label=f"{label} caption")
    default_caption = getattr(config, "default_caption", None)
    if default_caption is not None and not isinstance(default_caption, str):
        raise StateBridgeError(f"{label} default caption is invalid")
    raw = (default_caption or "").encode("utf-8")
    return {
        "kind": "default" if default_caption is not None else "absent",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _optional_content_descriptor(path: Any, *, label: str) -> dict[str, Any]:
    if path is None or path == "":
        return {"kind": "absent"}
    return _content_file_descriptor(path, label=label)


def _control_descriptors(item: Any, *, label: str) -> list[dict[str, Any]]:
    raw = getattr(item, "control_path", None)
    if raw is None or raw == "":
        return []
    paths = raw if isinstance(raw, (list, tuple)) else [raw]
    return [
        _content_file_descriptor(path, label=f"{label} control #{index + 1}")
        for index, path in enumerate(paths)
    ]


def _stable_json_value(value: Any, *, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise StateBridgeError(f"{label} is not stable JSON") from exc


def _dataset_config_state(dataset: Any, *, label: str) -> dict[str, Any]:
    config = getattr(dataset, "dataset_config", None)
    if config is None:
        raise StateBridgeError(f"{label} has no dataset_config")
    return {
        "buckets": bool(getattr(config, "buckets", False)),
        "random_crop": bool(getattr(config, "random_crop", False)),
        "random_scale": bool(getattr(config, "random_scale", False)),
        "resolution": _stable_json_value(
            getattr(config, "resolution", None), label=f"{label} resolution"
        ),
        "scale": _stable_json_value(
            getattr(config, "scale", None), label=f"{label} scale"
        ),
        "square_crop": bool(getattr(config, "square_crop", False)),
        "num_repeats": _bounded_int(
            getattr(config, "num_repeats", 1),
            label=f"{label} num_repeats",
            minimum=1,
        ),
        "flip_x": bool(getattr(config, "flip_x", False)),
        "flip_y": bool(getattr(config, "flip_y", False)),
        "standardize_images": bool(
            getattr(config, "standardize_images", False)
        ),
        "caption_ext": str(getattr(config, "caption_ext", ".txt")),
        "cache_latents": bool(getattr(config, "cache_latents", False)),
        "cache_latents_to_disk": bool(
            getattr(config, "cache_latents_to_disk", False)
        ),
        "cache_text_embeddings": bool(
            getattr(config, "cache_text_embeddings", False)
        ),
        "mask_min_value": _stable_json_value(
            getattr(config, "mask_min_value", 0.0),
            label=f"{label} mask_min_value",
        ),
        "invert_mask": bool(getattr(config, "invert_mask", False)),
    }


def _expected_cache_kinds(config_state: Mapping[str, Any]) -> frozenset[str]:
    expected = set()
    if bool(
        config_state.get("cache_latents")
        or config_state.get("cache_latents_to_disk")
    ):
        expected.add("latent")
    if bool(config_state.get("cache_text_embeddings")):
        expected.add("text_embedding")
    return frozenset(expected)


def _unsupported_dataset_reasons(dataset: Any, *, label: str) -> list[str]:
    config = getattr(dataset, "dataset_config", None)
    if config is None:
        return [f"{label} has no dataset_config"]
    reasons = []
    if bool(getattr(config, "random_scale", False)) or bool(
        getattr(dataset, "random_scale", False)
    ):
        reasons.append(f"{label} uses random_scale")
    if bool(getattr(dataset, "is_video", False)) or bool(
        getattr(dataset, "is_audio_model", False)
    ):
        reasons.append(f"{label} is a video/audio dataset")
    if int(getattr(config, "num_frames", 1) or 1) != 1 or bool(
        getattr(config, "auto_frame_count", False)
    ):
        reasons.append(f"{label} uses video frame sampling")
    if not bool(getattr(config, "buckets", False)) and bool(
        getattr(config, "random_crop", False)
        or getattr(dataset, "random_crop", False)
    ):
        reasons.append(f"{label} uses non-bucket random crop/scale")
    if getattr(config, "augments", None) or getattr(
        config, "augmentations", None
    ) or bool(getattr(config, "shuffle_augmentations", False)):
        reasons.append(f"{label} uses unsupported augmentations")
    if (
        getattr(config, "controls", None)
        or getattr(config, "control_path", None)
        or bool(getattr(config, "control_from_same_folder", False))
        or getattr(config, "inpaint_path", None)
        or getattr(config, "unconditional_path", None)
        or getattr(config, "clip_image_path", None)
        or bool(getattr(config, "clip_image_from_same_folder", False))
        or bool(getattr(dataset, "is_generating_controls", False))
    ):
        reasons.append(f"{label} uses unsupported control/conditioning inputs")
    if bool(getattr(config, "alpha_mask", False)):
        reasons.append(f"{label} uses an unsupported alpha-derived mask")
    if bool(getattr(config, "cache_clip_vision_to_disk", False)):
        reasons.append(f"{label} uses an unsupported CLIP vision cache")
    if bool(getattr(config, "load_image_when_caching_latents", False)):
        reasons.append(f"{label} uses an unsupported latent-cache transform")
    if bool(getattr(config, "cache_latents", False)) and not bool(
        getattr(config, "cache_latents_to_disk", False)
    ):
        reasons.append(
            f"{label} keeps stochastic latents only in memory; "
            "exact cache bytes cannot be archived"
        )
    return reasons


def _file_content_identity(
    dataset: Any,
    item: Any,
    *,
    label: str,
) -> dict[str, Any]:
    image = _content_file_descriptor(getattr(item, "path", None), label=label)
    descriptor = {
        "image": image,
        "caption": _caption_descriptor(dataset, item, label=label),
        "mask": _optional_content_descriptor(
            getattr(item, "mask_path", None), label=f"{label} mask"
        ),
        "controls": _control_descriptors(item, label=label),
        "width": _bounded_int(
            getattr(item, "width", None), label=f"{label}.width", minimum=1
        ),
        "height": _bounded_int(
            getattr(item, "height", None), label=f"{label}.height", minimum=1
        ),
        "flip_x": bool(getattr(item, "flip_x", False)),
        "flip_y": bool(getattr(item, "flip_y", False)),
    }
    return {
        "identity": descriptor,
        "identity_sha256": hashlib.sha256(_json_bytes(descriptor)).hexdigest(),
    }


def _cache_file_state(
    path: Any,
    *,
    kind: str,
    label: str,
    cache_sources: dict[str, tuple[Path, str, int]] | None,
) -> dict[str, Any]:
    digest, size = _hash_regular_source(path, label=label)
    entry = f"objects/{digest[:2]}/{digest}.bin"
    source = Path(os.path.abspath(os.fspath(path)))
    if cache_sources is not None:
        previous = cache_sources.get(entry)
        if previous is not None and previous[1:] != (digest, size):
            raise StateBridgeError(f"{label} cache entry collision")
        cache_sources[entry] = (source, digest, size)
    return {
        "kind": kind,
        "entry": entry,
        "sha256": digest,
        "size_bytes": size,
    }


def _file_cache_state(
    dataset: Any,
    item: Any,
    *,
    label: str,
    cache_sources: dict[str, tuple[Path, str, int]] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if bool(getattr(dataset, "is_caching_latents", False)):
        if not bool(getattr(dataset, "is_caching_latents_to_disk", False)):
            raise StateBridgeError(
                f"{label} latent cache is not persisted to disk"
            )
        if not bool(getattr(item, "is_latent_cached", False)):
            raise StateBridgeError(f"{label} latent cache is not ready")
        method = getattr(item, "get_latent_path", None)
        if not callable(method):
            raise StateBridgeError(f"{label} has no latent cache path seam")
        result["latent"] = _cache_file_state(
            method(),
            kind="latent",
            label=f"{label} latent cache",
            cache_sources=cache_sources,
        )
    if bool(getattr(dataset, "is_caching_text_embeddings", False)):
        if not bool(getattr(item, "is_text_embedding_cached", False)):
            raise StateBridgeError(f"{label} text embedding cache is not ready")
        method = getattr(item, "get_text_embedding_path", None)
        if not callable(method):
            raise StateBridgeError(f"{label} has no text cache path seam")
        result["text_embedding"] = _cache_file_state(
            method(),
            kind="text_embedding",
            label=f"{label} text embedding cache",
            cache_sources=cache_sources,
        )
    return result


def _file_item_state(
    dataset: Any,
    item: Any,
    *,
    label: str,
    cache_sources: dict[str, tuple[Path, str, int]] | None,
) -> dict[str, Any]:
    state = _file_content_identity(dataset, item, label=label)
    for field in _GEOMETRY_FIELDS:
        minimum = 1 if field in (
            "scale_to_width",
            "scale_to_height",
            "crop_width",
            "crop_height",
        ) else 0
        state[field] = _bounded_int(
            getattr(item, field, None),
            label=f"{label}.{field}",
            minimum=minimum,
        )
    if (
        state["crop_x"] + state["crop_width"] > state["scale_to_width"]
        or state["crop_y"] + state["crop_height"] > state["scale_to_height"]
    ):
        raise StateBridgeError(f"{label} crop geometry exceeds its scaled image")
    state["cache"] = _file_cache_state(
        dataset,
        item,
        label=label,
        cache_sources=cache_sources,
    )
    return state


def _normalise_indices(
    value: Any,
    *,
    label: str,
    upper_bound: int,
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise StateBridgeError(f"{label} is not an index list")
    result = [
        _bounded_int(item, label=f"{label}[{index}]", maximum=upper_bound - 1)
        for index, item in enumerate(value)
    ]
    return result


def _capture_aitk_dataset_state(
    dataset: Any,
    *,
    label: str,
    leaf_index: int,
    cache_sources: dict[str, tuple[Path, str, int]] | None = None,
) -> dict[str, Any]:
    config = getattr(dataset, "dataset_config", None)
    file_list = getattr(dataset, "file_list", None)
    if config is None or not isinstance(file_list, list) or not file_list:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} has an unrecognised mutable dataset shape"
        )
    dataset_label = f"{label} dataset #{leaf_index + 1}"
    unsupported = _unsupported_dataset_reasons(dataset, label=dataset_label)
    if unsupported:
        raise StateBridgeError("; ".join(unsupported))
    files = [
        _file_item_state(
            dataset,
            item,
            label=f"{dataset_label} file #{index + 1}",
            cache_sources=cache_sources,
        )
        for index, item in enumerate(file_list)
    ]
    identity_occurrences: dict[str, int] = defaultdict(int)
    for file_state in files:
        digest = file_state["identity_sha256"]
        file_state["identity_occurrence"] = identity_occurrences[digest]
        identity_occurrences[digest] += 1
    batch_size = _bounded_int(
        getattr(dataset, "batch_size", None),
        label=f"{dataset_label} batch_size",
        minimum=1,
    )
    epoch_num = _bounded_int(
        getattr(dataset, "epoch_num", None),
        label=f"{dataset_label} epoch_num",
    )
    uses_buckets = bool(getattr(config, "buckets", False))
    config_state = _dataset_config_state(dataset, label=dataset_label)
    expected_cache_kinds = _expected_cache_kinds(config_state)
    for index, file_state in enumerate(files):
        if set(file_state["cache"]) != expected_cache_kinds:
            raise StateBridgeError(
                f"{dataset_label} file #{index + 1} preprocessing cache "
                "does not match its dataset configuration"
            )
    content_multiset = sorted(Counter(
        file_state["identity_sha256"] for file_state in files
    ).items())
    state: dict[str, Any] = {
        "schema": _AITK_DATASET_SCHEMA,
        "mode": "buckets" if uses_buckets else "linear",
        "class": _qualified_type(dataset),
        "leaf_ordinal": leaf_index,
        "batch_size": batch_size,
        "epoch_num": epoch_num,
        "config": config_state,
        "identity_sha256": hashlib.sha256(_json_bytes({
            "leaf_ordinal": leaf_index,
            "class": _qualified_type(dataset),
            "config": config_state,
            "content_multiset": content_multiset,
        })).hexdigest(),
        "files": files,
    }
    if not uses_buckets:
        return state

    buckets = getattr(dataset, "buckets", None)
    raw_batches = getattr(dataset, "batch_indices", None)
    if not isinstance(buckets, Mapping) or not buckets:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} has no bucket mapping"
        )
    bucket_states = []
    all_indices: list[int] = []
    expected_batches: list[list[int]] = []
    seen_keys: set[str] = set()
    for bucket_index, (raw_key, bucket) in enumerate(buckets.items()):
        key = str(raw_key)
        if not key or key in seen_keys:
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} has duplicate/empty bucket keys"
            )
        seen_keys.add(key)
        indices = _normalise_indices(
            getattr(bucket, "file_list_idx", None),
            label=(
                f"{label} dataset #{leaf_index + 1} "
                f"bucket #{bucket_index + 1} indices"
            ),
            upper_bound=len(files),
        )
        if not indices:
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} has an empty bucket"
            )
        width = _bounded_int(
            getattr(bucket, "width", None),
            label=f"{label} dataset #{leaf_index + 1} bucket {key} width",
            minimum=1,
        )
        height = _bounded_int(
            getattr(bucket, "height", None),
            label=f"{label} dataset #{leaf_index + 1} bucket {key} height",
            minimum=1,
        )
        for item_index in indices:
            file_state = files[item_index]
            if (
                file_state["crop_width"] != width
                or file_state["crop_height"] != height
            ):
                raise StateBridgeError(
                    f"{label} dataset #{leaf_index + 1} bucket geometry "
                    "disagrees with its file entries"
                )
        all_indices.extend(indices)
        for start in range(0, len(indices), batch_size):
            batch = list(indices[start:start + batch_size])
            if len(batch) < batch_size:
                batch.extend(
                    batch[index % len(batch)]
                    for index in range(batch_size - len(batch))
                )
            expected_batches.append(batch)
        bucket_states.append({
            "key": key,
            "width": width,
            "height": height,
            "file_list_idx": indices,
        })
    if sorted(all_indices) != list(range(len(files))):
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} bucket membership is not a partition"
        )
    if not isinstance(raw_batches, (list, tuple)):
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} has no batch_indices"
        )
    batches = [
        _normalise_indices(
            batch,
            label=f"{label} dataset #{leaf_index + 1} batch #{index + 1}",
            upper_bound=len(files),
        )
        for index, batch in enumerate(raw_batches)
    ]
    if batches != expected_batches:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} batch_indices do not match "
            "its ordered bucket partition"
        )
    state["buckets"] = bucket_states
    state["batch_indices"] = batches
    return state


def _capture_dataset_topology(
    value: Any,
    *,
    label: str,
    leaves: list[Any],
    cache_sources: dict[str, tuple[Path, str, int]] | None,
) -> dict[str, Any]:
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "class": _qualified_type(value),
            "children": [
                _capture_dataset_topology(
                    child,
                    label=label,
                    leaves=leaves,
                    cache_sources=cache_sources,
                )
                for child in value
            ],
        }
    children = getattr(value, "datasets", None)
    if isinstance(children, (list, tuple)):
        return {
            "kind": "container",
            "class": _qualified_type(value),
            "children": [
                _capture_dataset_topology(
                    child,
                    label=label,
                    leaves=leaves,
                    cache_sources=cache_sources,
                )
                for child in children
            ],
        }
    leaf_index = len(leaves)
    if getattr(value, "dataset_config", None) is None:
        try:
            length = len(value)
        except (TypeError, ValueError) as exc:
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} has no stable length"
            ) from exc
        state = {
            "mode": "static",
            "class": _qualified_type(value),
            "leaf_ordinal": leaf_index,
            "length": _bounded_int(
                length,
                label=f"{label} dataset #{leaf_index + 1} length",
                minimum=1,
            ),
        }
    else:
        state = _capture_aitk_dataset_state(
            value,
            label=label,
            leaf_index=leaf_index,
            cache_sources=cache_sources,
        )
    leaves.append(state)
    return {
        "kind": "leaf",
        "class": _qualified_type(value),
        "leaf_ordinal": leaf_index,
    }


def _capture_loader_dataset_state(
    loader: DataLoader | None,
    *,
    label: str,
    cache_sources: dict[str, tuple[Path, str, int]] | None = None,
) -> dict[str, Any] | None:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise StateBridgeError(f"{label} dataloader has no leaf datasets")
    states: list[Any] = []
    topology = _capture_dataset_topology(
        dataset,
        label=label,
        leaves=states,
        cache_sources=cache_sources,
    )
    if not states:
        raise StateBridgeError(f"{label} dataloader has no leaf datasets")
    return {
        "schema": _DATASET_GRAPH_SCHEMA,
        "label": label,
        "leaves": states,
        "topology": topology,
    }


def _validate_saved_file_state(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    label: str,
    expected_cache_kinds: frozenset[str],
) -> None:
    identity = saved.get("identity")
    identity_digest = saved.get("identity_sha256")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(identity_digest, str)
        or hashlib.sha256(_json_bytes(identity)).hexdigest() != identity_digest
        or identity_digest != current.get("identity_sha256")
        or identity != current.get("identity")
    ):
        raise StateBridgeError(f"{label} file identity/order changed")
    _bounded_int(
        saved.get("identity_occurrence"),
        label=f"{label}.identity_occurrence",
    )
    for field in _GEOMETRY_FIELDS:
        minimum = 1 if field in (
            "scale_to_width",
            "scale_to_height",
            "crop_width",
            "crop_height",
        ) else 0
        _bounded_int(saved.get(field), label=f"{label}.{field}", minimum=minimum)
    if (
        saved["crop_x"] + saved["crop_width"] > saved["scale_to_width"]
        or saved["crop_y"] + saved["crop_height"] > saved["scale_to_height"]
    ):
        raise StateBridgeError(f"{label} saved crop geometry is invalid")
    cache = saved.get("cache")
    if (
        not isinstance(cache, Mapping)
        or set(cache) != expected_cache_kinds
    ):
        raise StateBridgeError(f"{label} saved cache state is invalid")
    for cache_kind, descriptor in cache.items():
        if cache_kind not in ("latent", "text_embedding"):
            raise StateBridgeError(f"{label} saved cache kind is unsupported")
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("kind") != cache_kind
            or not isinstance(descriptor.get("entry"), str)
            or not re.fullmatch(
                r"objects/[0-9a-f]{2}/[0-9a-f]{64}\.bin",
                descriptor["entry"],
            )
            or descriptor["entry"].split("/")[-1][:-4]
            != descriptor.get("sha256")
        ):
            raise StateBridgeError(f"{label} saved cache descriptor is invalid")
        _bounded_int(
            descriptor.get("size_bytes"),
            label=f"{label} {cache_kind} cache size",
        )


def _current_file_identities(
    dataset: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    file_list = getattr(dataset, "file_list", None)
    if not isinstance(file_list, list) or not file_list:
        raise StateBridgeError(f"{label} has no current file list")
    result = [
        _file_content_identity(
            dataset,
            item,
            label=f"{label} current file #{index + 1}",
        )
        for index, item in enumerate(file_list)
    ]
    occurrences: dict[str, int] = defaultdict(int)
    for state in result:
        digest = state["identity_sha256"]
        state["identity_occurrence"] = occurrences[digest]
        occurrences[digest] += 1
    return result


def _restore_aitk_dataset_state(
    dataset: Any,
    saved: Mapping[str, Any],
    *,
    label: str,
    leaf_index: int,
    bucket_objects_ready: bool,
) -> None:
    if saved.get("schema") != _AITK_DATASET_SCHEMA:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} state schema is unsupported"
        )
    if saved.get("class") != _qualified_type(dataset):
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} class changed"
        )
    if saved.get("leaf_ordinal") != leaf_index:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} ordinal changed"
        )
    dataset_label = f"{label} dataset #{leaf_index + 1}"
    unsupported = _unsupported_dataset_reasons(dataset, label=dataset_label)
    if unsupported:
        raise StateBridgeError("; ".join(unsupported))
    config = _dataset_config_state(dataset, label=dataset_label)
    if saved.get("config") != config:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} configuration changed"
        )
    current_mode = "buckets" if config["buckets"] else "linear"
    expected_cache_kinds = _expected_cache_kinds(config)
    if saved.get("mode") != current_mode:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} bucket mode changed"
        )
    batch_size = _bounded_int(
        getattr(dataset, "batch_size", None),
        label=f"{dataset_label} batch_size",
        minimum=1,
    )
    if saved.get("batch_size") != batch_size:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} batch size changed"
        )
    saved_files = saved.get("files")
    current_files = _current_file_identities(dataset, label=dataset_label)
    if (
        not isinstance(saved_files, list)
        or len(saved_files) != len(current_files)
    ):
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} file count changed"
        )
    current_counts = Counter(
        item["identity_sha256"] for item in current_files
    )
    saved_counts = Counter(
        item.get("identity_sha256")
        for item in saved_files
        if isinstance(item, Mapping)
    )
    if saved_counts != current_counts:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} content multiset changed"
        )
    current_by_identity: dict[str, deque[tuple[Any, Mapping[str, Any]]]] = (
        defaultdict(deque)
    )
    for item, current_file in zip(dataset.file_list, current_files):
        current_by_identity[current_file["identity_sha256"]].append(
            (item, current_file)
        )
    reordered_items = []
    for index, saved_file in enumerate(saved_files):
        if not isinstance(saved_file, Mapping):
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} saved file state is invalid"
            )
        digest = saved_file.get("identity_sha256")
        candidates = current_by_identity.get(digest)
        if not candidates:
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} file multiset changed"
            )
        current_item, current_file = candidates.popleft()
        _validate_saved_file_state(
            saved_file,
            current_file,
            label=f"{label} dataset #{leaf_index + 1} file #{index + 1}",
            expected_cache_kinds=expected_cache_kinds,
        )
        reordered_items.append(current_item)
    if any(current_by_identity.values()):
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} file multiset changed"
        )
    dataset.file_list = reordered_items

    expected_identity = hashlib.sha256(_json_bytes({
        "leaf_ordinal": leaf_index,
        "class": _qualified_type(dataset),
        "config": config,
        "content_multiset": sorted(current_counts.items()),
    })).hexdigest()
    if saved.get("identity_sha256") != expected_identity:
        raise StateBridgeError(
            f"{label} dataset #{leaf_index + 1} identity changed"
        )

    saved_epoch = _bounded_int(
        saved.get("epoch_num"),
        label=f"{label} dataset #{leaf_index + 1} saved epoch_num",
    )
    for item, file_state in zip(dataset.file_list, saved_files):
        for field in _GEOMETRY_FIELDS:
            setattr(item, field, int(file_state[field]))
    if not bucket_objects_ready:
        # The fresh AiToolkitDataset must enter its epoch-zero setup so the
        # patched cache hooks can materialise saved bytes.  This matters for
        # linear datasets too: restoring their saved epoch here would make
        # upstream skip cache_latents/cache_text_embeddings altogether.
        return
    if saved.get("mode") == "buckets":
        saved_buckets = saved.get("buckets")
        saved_batches = saved.get("batch_indices")
        if not isinstance(saved_buckets, list) or not isinstance(saved_batches, list):
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} saved bucket state is invalid"
            )
        current_buckets = getattr(dataset, "buckets", None)
        if not isinstance(current_buckets, Mapping):
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} current buckets are unavailable"
            )
        rebuilt: dict[str, Any] = {}
        all_indices: list[int] = []
        expected_batches: list[list[int]] = []
        batch_size = int(saved["batch_size"])
        for bucket_index, item in enumerate(saved_buckets):
            if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
                raise StateBridgeError(
                    f"{label} dataset #{leaf_index + 1} saved bucket is invalid"
                )
            key = item["key"]
            bucket = current_buckets.get(key)
            if bucket is None:
                raise StateBridgeError(
                    f"{label} dataset #{leaf_index + 1} bucket {key!r} changed"
                )
            width = _bounded_int(
                item.get("width"), label=f"{label} saved bucket {key} width", minimum=1
            )
            height = _bounded_int(
                item.get("height"), label=f"{label} saved bucket {key} height", minimum=1
            )
            if (
                _bounded_int(
                    getattr(bucket, "width", None),
                    label=f"{label} current bucket {key} width",
                    minimum=1,
                ) != width
                or _bounded_int(
                    getattr(bucket, "height", None),
                    label=f"{label} current bucket {key} height",
                    minimum=1,
                ) != height
            ):
                raise StateBridgeError(
                    f"{label} dataset #{leaf_index + 1} bucket geometry changed"
                )
            indices = _normalise_indices(
                item.get("file_list_idx"),
                label=f"{label} saved bucket {key} indices",
                upper_bound=len(saved_files),
            )
            if not indices:
                raise StateBridgeError(f"{label} saved bucket {key} is empty")
            all_indices.extend(indices)
            for start in range(0, len(indices), batch_size):
                batch = list(indices[start:start + batch_size])
                if len(batch) < batch_size:
                    batch.extend(
                        batch[index % len(batch)]
                        for index in range(batch_size - len(batch))
                    )
                expected_batches.append(batch)
            bucket.file_list_idx = list(indices)
            bucket.width = width
            bucket.height = height
            rebuilt[key] = bucket
        if set(rebuilt) != set(current_buckets):
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} bucket keys changed"
            )
        if sorted(all_indices) != list(range(len(saved_files))):
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} saved buckets are not a partition"
            )
        normalised_saved_batches = [
            _normalise_indices(
                batch,
                label=f"{label} saved batch #{index + 1}",
                upper_bound=len(saved_files),
            )
            for index, batch in enumerate(saved_batches)
        ]
        if normalised_saved_batches != expected_batches:
            raise StateBridgeError(
                f"{label} dataset #{leaf_index + 1} saved batches disagree "
                "with saved buckets"
            )
        dataset.buckets = rebuilt
        dataset.batch_indices = [list(batch) for batch in normalised_saved_batches]

    # setup_epoch's wrapper applies this after ai-toolkit performs its own
    # increment; the late restore path applies it idempotently.
    dataset.epoch_num = saved_epoch


def _current_dataset_topology(value: Any, leaves: list[Any]) -> dict[str, Any]:
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "class": _qualified_type(value),
            "children": [
                _current_dataset_topology(child, leaves) for child in value
            ],
        }
    children = getattr(value, "datasets", None)
    if isinstance(children, (list, tuple)):
        return {
            "kind": "container",
            "class": _qualified_type(value),
            "children": [
                _current_dataset_topology(child, leaves) for child in children
            ],
        }
    ordinal = len(leaves)
    leaves.append(value)
    return {
        "kind": "leaf",
        "class": _qualified_type(value),
        "leaf_ordinal": ordinal,
    }


def _restore_loader_dataset_state(
    loader: DataLoader | None,
    saved: Mapping[str, Any] | None,
    *,
    label: str,
) -> None:
    if loader is None:
        if saved is not None:
            raise StateBridgeError(f"saved {label} dataset graph is absent")
        return
    if (
        not isinstance(saved, Mapping)
        or saved.get("schema") != _DATASET_GRAPH_SCHEMA
        or saved.get("label") != label
        or not isinstance(saved.get("leaves"), list)
        or not isinstance(saved.get("topology"), Mapping)
    ):
        raise StateBridgeError(f"saved {label} dataset graph is invalid")
    leaves: list[Any] = []
    topology = _current_dataset_topology(
        getattr(loader, "dataset", None), leaves
    )
    if saved["topology"] != topology:
        raise StateBridgeError(f"{label} dataset topology changed")
    states = saved["leaves"]
    if len(leaves) != len(states):
        raise StateBridgeError(f"{label} dataset child count changed")
    for index, (dataset, state) in enumerate(zip(leaves, states)):
        if not isinstance(state, Mapping):
            raise StateBridgeError(f"saved {label} dataset state is invalid")
        if state.get("mode") == "static":
            if (
                state.get("class") != _qualified_type(dataset)
                or state.get("leaf_ordinal") != index
                or state.get("length") != len(dataset)
            ):
                raise StateBridgeError(
                    f"{label} static dataset #{index + 1} changed"
                )
            continue
        marker = getattr(dataset, "_lds_dataset_state_restored_early", None)
        if (
            not isinstance(marker, Mapping)
            or marker.get("label") != label
            or marker.get("leaf_ordinal") != index
            or marker.get("identity_sha256") != state.get("identity_sha256")
        ):
            raise StateBridgeError(
                f"{label} dataset #{index + 1} was not restored before caching"
            )
        _restore_aitk_dataset_state(
            dataset,
            state,
            label=label,
            leaf_index=index,
            bucket_objects_ready=True,
        )


def _dataset_gate_reasons(trainer: Any) -> list[str]:
    """Prove that every loader dataset has a serialisable replay contract."""
    reasons: list[str] = []
    for loader_attr, label in (
        ("data_loader", "main"),
        ("data_loader_reg", "regularization"),
    ):
        loader = getattr(trainer, loader_attr, None)
        if loader is None and label == "regularization":
            continue
        try:
            _capture_loader_dataset_state(loader, label=label)
        except Exception as exc:
            reasons.append(str(exc))
    return reasons


def _cache_archive_manifest(
    cache_sources: Mapping[str, tuple[Path, str, int]],
) -> dict[str, Any]:
    return {
        "schema": _CACHE_ARCHIVE_SCHEMA,
        "entries": [
            {
                "name": name,
                "sha256": digest,
                "size_bytes": size,
            }
            for name, (_source, digest, size) in sorted(cache_sources.items())
        ],
    }


def _write_cache_archive(
    path: Path,
    cache_sources: Mapping[str, tuple[Path, str, int]],
) -> None:
    """Write a deterministic, path-free archive of exact preprocessing bytes."""
    manifest_bytes = _json_bytes(_cache_archive_manifest(cache_sources))
    with tarfile.open(path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        manifest_info = tarfile.TarInfo("CACHE-MANIFEST.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mode = 0o600
        manifest_info.mtime = 0
        import io

        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for name, (source, expected_digest, expected_size) in sorted(
            cache_sources.items()
        ):
            if not re.fullmatch(r"objects/[0-9a-f]{2}/[0-9a-f]{64}\.bin", name):
                raise StateBridgeError("unsafe preprocessing-cache archive name")
            before_digest, before_size = _hash_regular_source(
                source, label=f"preprocessing cache {name}"
            )
            if (
                before_digest != expected_digest
                or before_size != expected_size
            ):
                raise StateBridgeError(
                    f"preprocessing cache {name} changed before archiving"
                )
            info = tarfile.TarInfo(name)
            info.size = expected_size
            info.mode = 0o600
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
            after_digest, after_size = _hash_regular_source(
                source, label=f"preprocessing cache {name}"
            )
            if (
                after_digest != expected_digest
                or after_size != expected_size
            ):
                raise StateBridgeError(
                    f"preprocessing cache {name} changed while archiving"
                )


def _cache_descriptors(saved: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    descriptors: dict[str, Mapping[str, Any]] = {}
    files = saved.get("files")
    if not isinstance(files, list):
        raise StateBridgeError("saved dataset files are invalid")
    for file_state in files:
        if not isinstance(file_state, Mapping):
            raise StateBridgeError("saved dataset file is invalid")
        cache = file_state.get("cache")
        if not isinstance(cache, Mapping):
            raise StateBridgeError("saved dataset cache map is invalid")
        for descriptor in cache.values():
            if not isinstance(descriptor, Mapping):
                raise StateBridgeError("saved dataset cache descriptor is invalid")
            entry = descriptor.get("entry")
            previous = descriptors.get(entry)
            if previous is not None and previous != descriptor:
                raise StateBridgeError("conflicting preprocessing-cache descriptor")
            descriptors[entry] = descriptor
    return descriptors


def _validated_cache_members(
    archive: tarfile.TarFile,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if (
            not member.isfile()
            or member.name in members
            or member.name.startswith(("/", "\\"))
            or "\\" in member.name
            or any(part in ("", ".", "..") for part in member.name.split("/"))
        ):
            raise StateBridgeError("preprocessing-cache archive is unsafe")
        members[member.name] = member
    wanted = {"CACHE-MANIFEST.json", *expected}
    if set(members) != wanted:
        raise StateBridgeError("preprocessing-cache archive entries changed")
    manifest_stream = archive.extractfile(members["CACHE-MANIFEST.json"])
    if manifest_stream is None:
        raise StateBridgeError("preprocessing-cache manifest is missing")
    try:
        manifest = json.loads(manifest_stream.read().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StateBridgeError("preprocessing-cache manifest is invalid") from exc
    wanted_manifest = {
        "schema": _CACHE_ARCHIVE_SCHEMA,
        "entries": [
            {
                "name": name,
                "sha256": descriptor["sha256"],
                "size_bytes": descriptor["size_bytes"],
            }
            for name, descriptor in sorted(expected.items())
        ],
    }
    if manifest != wanted_manifest:
        raise StateBridgeError("preprocessing-cache manifest changed")
    for name, descriptor in expected.items():
        if members[name].size != descriptor["size_bytes"]:
            raise StateBridgeError("preprocessing-cache member size changed")
    return members


def _cache_path_components(path: Path):
    parts = path.parts
    if not parts:
        raise StateBridgeError("preprocessing-cache path is empty")
    current = Path(parts[0])
    yield current
    for part in parts[1:]:
        current = current / part
        yield current


def _assert_cache_path_safe(
    path: Path,
    *,
    label: str,
) -> os.stat_result | None:
    """lstat every existing component, rejecting links and Windows reparses."""
    final_info = None
    for component in _cache_path_components(path):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StateBridgeError(f"{label} cannot be inspected safely") from exc
        if _is_link_or_reparse_info(info):
            raise StateBridgeError(f"{label} contains a link or reparse point")
        if component != path and not stat.S_ISDIR(info.st_mode):
            raise StateBridgeError(f"{label} has a non-directory parent")
        final_info = info
    return final_info


def _safe_cache_destination(item: Any, kind: str) -> Path:
    if kind == "latent":
        method_name = "get_latent_path"
        directory_name = "_latent_cache"
    elif kind == "text_embedding":
        method_name = "get_text_embedding_path"
        directory_name = "_t_e_cache"
    else:
        raise StateBridgeError(f"unsupported preprocessing cache kind {kind!r}")
    method = getattr(item, method_name, None)
    if not callable(method):
        raise StateBridgeError(f"dataset item has no {method_name} seam")
    destination = Path(os.path.abspath(os.fspath(method(recalculate=True))))
    image = Path(os.path.abspath(os.fspath(getattr(item, "path", ""))))
    image_parent = image.parent
    image_info = _assert_cache_path_safe(
        image, label="dataset image path")
    if image_info is None or not stat.S_ISREG(image_info.st_mode):
        raise StateBridgeError("dataset image is not a regular file")
    cache_parent_info = _assert_cache_path_safe(
        destination.parent,
        label="preprocessing-cache directory",
    )
    if (
        cache_parent_info is not None
        and not stat.S_ISDIR(cache_parent_info.st_mode)
    ):
        raise StateBridgeError("preprocessing-cache parent is not a directory")
    destination_info = _assert_cache_path_safe(
        destination,
        label="preprocessing-cache destination",
    )
    if (
        destination_info is not None
        and not stat.S_ISREG(destination_info.st_mode)
    ):
        raise StateBridgeError("preprocessing-cache destination is not regular")
    if (
        destination.parent.name != directory_name
        or destination.parent.parent != image_parent
        or destination.suffix != ".safetensors"
    ):
        raise StateBridgeError("ai-toolkit returned an unsafe cache destination")
    return destination


def _materialize_saved_cache_kind(
    dataset: Any,
    saved: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    archive_path = _EARLY_CACHE_ARCHIVE
    if archive_path is None or not archive_path.is_file() or archive_path.is_symlink():
        raise StateBridgeError("verified preprocessing-cache archive is unavailable")
    all_descriptors = _EARLY_CACHE_DESCRIPTORS
    if not isinstance(all_descriptors, Mapping):
        raise StateBridgeError("preprocessing-cache descriptor index is unavailable")
    saved_files = saved.get("files")
    if not isinstance(saved_files, list) or len(saved_files) != len(dataset.file_list):
        raise StateBridgeError("saved preprocessing-cache mapping changed")
    destinations: dict[Path, tuple[Mapping[str, Any], Any]] = {}
    for item, file_state in zip(dataset.file_list, saved_files):
        cache = file_state.get("cache")
        descriptor = cache.get(kind) if isinstance(cache, Mapping) else None
        if descriptor is None:
            continue
        destination = _safe_cache_destination(item, kind)
        previous = destinations.get(destination)
        if previous is not None and previous[0] != descriptor:
            raise StateBridgeError("preprocessing caches collide at one destination")
        destinations[destination] = (descriptor, item)

    with tarfile.open(archive_path, mode="r:") as archive:
        members = _validated_cache_members(archive, all_descriptors)
        for destination, (descriptor, item) in destinations.items():
            if _safe_cache_destination(item, kind) != destination:
                raise StateBridgeError("preprocessing-cache destination changed")
            try:
                destination.parent.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise StateBridgeError(
                    "preprocessing-cache directory cannot be created safely"
                ) from exc
            parent_info = _assert_cache_path_safe(
                destination.parent,
                label="preprocessing-cache directory",
            )
            if parent_info is None or not stat.S_ISDIR(parent_info.st_mode):
                raise StateBridgeError(
                    "preprocessing-cache directory is unavailable"
                )
            stream = archive.extractfile(members[descriptor["entry"]])
            if stream is None:
                raise StateBridgeError("preprocessing-cache member is missing")
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            digest = hashlib.sha256()
            size = 0
            try:
                if _safe_cache_destination(item, kind) != destination:
                    raise StateBridgeError(
                        "preprocessing-cache destination changed"
                    )
                with temporary.open("xb") as output:
                    while True:
                        chunk = stream.read(1 << 20)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    size != descriptor["size_bytes"]
                    or digest.hexdigest() != descriptor["sha256"]
                ):
                    raise StateBridgeError(
                        "preprocessing-cache member hash changed"
                    )
                if _safe_cache_destination(item, kind) != destination:
                    raise StateBridgeError(
                        "preprocessing-cache destination changed before publish"
                    )
                temporary_info = _assert_cache_path_safe(
                    temporary,
                    label="preprocessing-cache temporary file",
                )
                if temporary_info is None or not stat.S_ISREG(
                    temporary_info.st_mode
                ):
                    raise StateBridgeError(
                        "preprocessing-cache temporary file changed"
                    )
                os.replace(temporary, destination)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _verify_materialized_cache_kind(
    dataset: Any,
    saved: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    for item, file_state in zip(dataset.file_list, saved["files"]):
        cache = file_state["cache"]
        descriptor = cache.get(kind)
        if descriptor is None:
            continue
        digest, size = _hash_regular_source(
            _safe_cache_destination(item, kind),
            label=f"restored {kind} cache",
        )
        if digest != descriptor["sha256"] or size != descriptor["size_bytes"]:
            raise StateBridgeError(
                f"ai-toolkit changed restored {kind} cache bytes"
            )


def _trainer_gate_reasons(trainer: Any) -> list[str]:
    reasons = _dataset_gate_reasons(trainer)
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None:
        reasons.append("accelerator is absent")
    elif int(getattr(accelerator, "num_processes", 1)) != 1:
        reasons.append("distributed/multi-process training is not supported")
    for name in ("optimizer", "lr_scheduler"):
        value = getattr(trainer, name, None)
        if value is None:
            reasons.append(f"{name} is absent")
        elif not all(hasattr(value, method) for method in ("state_dict", "load_state_dict")):
            reasons.append(f"{name} has no state_dict/load_state_dict seam")
    mixed = str(getattr(accelerator, "mixed_precision", "") or "").lower()
    if mixed == "fp16" and getattr(accelerator, "scaler", None) is None:
        reasons.append("fp16 accelerator has no GradScaler state")
    try:
        _bundle_core()
    except Exception as exc:
        reasons.append(str(exc))
    try:
        _identity()
    except Exception as exc:
        reasons.append(str(exc))
    return reasons


def _public_checkpoint(trainer: Any, step: int, *, numbered: bool) -> Path:
    path = trainer.get_latest_save_path()
    if not path:
        raise StateBridgeError("ai-toolkit did not expose the saved checkpoint")
    path = Path(path).resolve()
    root = Path(trainer.save_root).resolve()
    if path.parent != root or not path.is_file():
        raise StateBridgeError("public checkpoint is not a file directly in save_root")
    token = f"_{step:09d}"
    if numbered and token not in path.stem:
        raise StateBridgeError(
            f"public checkpoint {path.name!r} does not identify completed step {step}"
        )
    if not numbered and re.search(r"_\d{9}(?:_|$)", path.stem):
        raise StateBridgeError(
            f"final public checkpoint {path.name!r} is unexpectedly numbered"
        )
    return path


def _is_link_or_reparse_info(info: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(info.st_mode)
        or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
        )
    )


def _remove_runtime_tree_no_follow(path: Path, *, root: Path) -> None:
    root = Path(os.path.abspath(os.fspath(root)))
    path = Path(os.path.abspath(os.fspath(path)))
    if path.parent != root:
        raise OSError("runtime work directory escaped save_root")
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse_info(info):
        raise OSError("runtime work directory is linked")

    def remove_directory(directory: Path) -> None:
        with os.scandir(directory) as entries:
            children = list(entries)
        for entry in children:
            child = Path(entry.path)
            child_info = entry.stat(follow_symlinks=False)
            if (
                stat.S_ISDIR(child_info.st_mode)
                and not _is_link_or_reparse_info(child_info)
            ):
                remove_directory(child)
            elif stat.S_ISDIR(child_info.st_mode):
                os.rmdir(child)
            else:
                os.unlink(child)
        os.rmdir(directory)

    remove_directory(path)


def _pid_liveness(pid: int) -> str:
    """Return alive/dead/unknown; only ``dead`` authorises scavenging."""
    if pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unknown"
    except OSError as exc:
        if getattr(exc, "winerror", None) == 87:
            return "dead"
        return "unknown"
    return "alive"


def scavenge_runtime_workdirs(
    save_root: os.PathLike[str] | str,
    *,
    older_than_seconds: int = _WORK_MAX_AGE_SECONDS,
    now: float | None = None,
) -> tuple[str, ...]:
    """Remove exact-name stale work dirs only when their owner is proven dead."""
    if (
        isinstance(older_than_seconds, bool)
        or not isinstance(older_than_seconds, int)
        or older_than_seconds < 0
    ):
        raise ValueError("older_than_seconds must be a non-negative integer")
    root = Path(os.path.realpath(os.path.abspath(os.fspath(save_root))))
    try:
        root_info = os.lstat(root)
    except OSError:
        return ()
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or _is_link_or_reparse_info(root_info)
    ):
        return ()
    cutoff = (time.time() if now is None else float(now)) - older_than_seconds
    removed = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not _WORK_RE.fullmatch(entry.name):
            continue
        try:
            info = os.lstat(entry)
        except OSError:
            continue
        if (
            not stat.S_ISDIR(info.st_mode)
            or _is_link_or_reparse_info(info)
            or info.st_mtime > cutoff
        ):
            continue
        try:
            owner = read_json_nofollow(entry / _WORK_OWNER_FILENAME)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if (
            not isinstance(owner, Mapping)
            or owner.get("schema") != _WORK_OWNER_SCHEMA
            or isinstance(owner.get("pid"), bool)
            or not isinstance(owner.get("pid"), int)
            or isinstance(owner.get("process_started_at_ns"), bool)
            or not isinstance(owner.get("process_started_at_ns"), int)
            or owner["process_started_at_ns"] <= 0
        ):
            continue
        if _pid_liveness(owner["pid"]) != "dead":
            continue
        try:
            _remove_runtime_tree_no_follow(entry, root=root)
        except OSError:
            continue
        removed.append(entry.name)
    return tuple(removed)


@contextmanager
def _runtime_work_directory(save_root: Path):
    root = Path(os.path.realpath(os.path.abspath(os.fspath(save_root))))
    scavenge_runtime_workdirs(root)
    path = root / f"{_WORK_PREFIX}{uuid.uuid4().hex}"
    path.mkdir(mode=0o700)
    try:
        atomic_json_nofollow(
            path / _WORK_OWNER_FILENAME,
            {
                "schema": _WORK_OWNER_SCHEMA,
                "pid": os.getpid(),
                "process_started_at_ns": _PROCESS_STARTED_AT_NS,
                "created_at_ns": time.time_ns(),
            },
        )
        yield path
    finally:
        try:
            _remove_runtime_tree_no_follow(path, root=root)
        except (FileNotFoundError, OSError):
            pass


def _environment_positive_bytes(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise StateBridgeError(f"{name} must be a positive integer byte count") from exc
    if value <= 0:
        raise StateBridgeError(f"{name} must be a positive integer byte count")
    return value


def _estimated_state_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Conservative tensor/container estimate used before disk staging exists."""
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return 0
    if isinstance(value, torch.Tensor):
        seen.add(marker)
        return int(value.numel()) * int(value.element_size()) + 4096
    if isinstance(value, Mapping):
        seen.add(marker)
        return 4096 + sum(
            _estimated_state_bytes(key, seen)
            + _estimated_state_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        seen.add(marker)
        return 4096 + sum(_estimated_state_bytes(item, seen) for item in value)
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) + 256
    return 256


def _write_json_artifact(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise StateBridgeError("torch.load(weights_only=True) is required") from exc


def _metadata_for_core(
    core: Any,
    identity: Mapping[str, Any],
    *,
    completed_step: int,
    next_step: int,
    optimizer_updates_completed: int,
) -> Any:
    runtime = dict(identity["runtime"])
    runtime.update(
        {
            "bridge": BRIDGE_NAME,
            "bridge_version": BRIDGE_VERSION,
            "protocol": BRIDGE_PROTOCOL,
            "shape_revision": SHAPE_REVISION,
        }
    )
    values = {
        "completed_step": completed_step,
        "next_step": next_step,
        "optimizer_updates_completed": optimizer_updates_completed,
        "toolkit_revision": str(identity["toolkit_revision"]),
        "toolkit_runtime": runtime,
        "config_hash": str(identity["config_hash"]),
        "dataset_hash": str(identity["dataset_hash"]),
        "base_model_hash": str(identity["base_hash"]),
        "network_hash": str(identity["network_hash"]),
        "capabilities": list(_RUNTIME_CAPABILITIES),
        "state_level": "exact",
    }
    try:
        return core.BundleMetadata(**values)
    except TypeError as exc:
        raise StateBridgeError(f"incompatible canonical BundleMetadata API: {exc}") from exc


def capture_exact_bundle(
    trainer: Any,
    completed_step: int,
    *,
    next_step: int | None = None,
    optimizer_boundary_step: int | None = None,
    numbered_checkpoint: bool = True,
) -> Path:
    """Capture a complete K-boundary state through the canonical bundle writer."""
    step = int(completed_step)
    if step < 0:
        raise StateBridgeError("completed_step must be non-negative")
    effective_next_step = step + 1 if next_step is None else int(next_step)
    boundary_step = (
        step if optimizer_boundary_step is None else int(optimizer_boundary_step)
    )
    if effective_next_step < step:
        raise StateBridgeError("next_step cannot precede completed_step")
    if getattr(trainer, "_lds_last_optimizer_boundary_step", None) != boundary_step:
        raise StateBridgeError(
            "save is not immediately after a confirmed optimizer boundary"
        )
    optimizer_updates = getattr(
        trainer, "_lds_optimizer_updates_completed", None)
    if (
        isinstance(optimizer_updates, bool)
        or not isinstance(optimizer_updates, int)
        or optimizer_updates < 0
    ):
        raise StateBridgeError(
            "optimizer update counter is unavailable at the save boundary"
        )
    reasons = _trainer_gate_reasons(trainer)
    if reasons:
        raise StateBridgeError("; ".join(reasons))

    loaders: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    cache_sources: dict[str, tuple[Path, str, int]] = {}
    for attr, label in (("data_loader", "main"), ("data_loader_reg", "regularization")):
        loader = getattr(trainer, attr, None)
        if loader is not None:
            stream = getattr(loader, "_lds_replay_stream", None)
            if stream is None:
                raise StateBridgeError(f"{label} dataloader is not instrumented")
            loaders[label] = stream.state_dict()
            datasets[label] = _capture_loader_dataset_state(
                loader,
                label=label,
                cache_sources=cache_sources,
            )
    if "main" not in loaders:
        raise StateBridgeError("main dataloader is absent")

    identity = _identity()
    public = _public_checkpoint(trainer, step, numbered=numbered_checkpoint)
    save_root = Path(trainer.save_root).resolve()
    accelerator = trainer.accelerator
    scaler = getattr(accelerator, "scaler", None)
    ema = getattr(trainer, "ema", None)
    require_cuda = _trainer_requires_cuda(trainer)
    core = _bundle_core()
    retention = int(os.environ.get(ENV_KEEP, "2"))
    raw_state = _optimizer_params(trainer.optimizer)
    optimizer_state = trainer.optimizer.state_dict()
    scheduler_state = trainer.lr_scheduler.state_dict()
    scaler_state = {
        "applicable": scaler is not None,
        "state": scaler.state_dict() if scaler is not None else {},
    }
    ema_state = {
        "applicable": ema is not None,
        "state": ema.state_dict() if ema is not None else {},
    }
    rng_json_state = _rng_json_state()
    rng_torch_state = _torch_rng_state(require_cuda)
    dataloader_state = {
        "schema": _DATALOADER_STATE_SCHEMA,
        "loaders": loaders,
        "datasets": datasets,
    }
    trainer_state = {
        "completed_step": step,
        "next_step": effective_next_step,
        "epoch_num": int(getattr(trainer, "epoch_num", 0)),
        # ai-toolkit increments this after save(), so persist the value
        # that the next iteration must observe.
        "grad_accumulation_step": int(
            getattr(trainer, "grad_accumulation_step", 0)
        )
        + (
            0
            if int(getattr(trainer, "step_num", step))
            == effective_next_step
            else 1
        ),
        "current_boundary_index": getattr(
            trainer, "current_boundary_index", None
        ),
        "steps_this_boundary": getattr(trainer, "steps_this_boundary", None),
    }
    estimated_sources = (
        int(public.stat().st_size)
        + sum(size for _source, _digest, size in cache_sources.values())
        + len(_json_bytes(_cache_archive_manifest(cache_sources)))
        + len(cache_sources) * 1024
        + 2048
        + len(_json_bytes(rng_json_state))
        + len(_json_bytes(dataloader_state))
        + len(_json_bytes(trainer_state))
        + sum(
            _estimated_state_bytes(value)
            for value in (
                raw_state,
                optimizer_state,
                scheduler_state,
                scaler_state,
                ema_state,
                rng_torch_state,
            )
        )
    )
    reserve_bytes = _environment_positive_bytes(
        ENV_RESERVE_BYTES,
        int(getattr(core, "DEFAULT_FREE_SPACE_RESERVE_BYTES", 5 * (1 << 30))),
    )
    max_bundle_bytes = _environment_positive_bytes(
        ENV_MAX_BUNDLE_BYTES,
        int(getattr(core, "DEFAULT_MAX_BUNDLE_BYTES", 64 * (1 << 30))),
    )
    max_store_bytes = _environment_positive_bytes(
        ENV_MAX_STORE_BYTES,
        int(getattr(core, "DEFAULT_MAX_STORE_BYTES", 128 * (1 << 30))),
    )
    try:
        core.preflight_bundle_space(
            save_root,
            estimated_sources,
            retention=retention,
            reserve_bytes=reserve_bytes,
            max_bundle_bytes=max_bundle_bytes,
            max_store_bytes=max_store_bytes,
            # Temporary serialization plus canonical staging coexist at peak.
            copy_multiplier=2,
        )
    except Exception as exc:
        raise StateBridgeError(f"state disk preflight failed: {exc}") from exc

    with _runtime_work_directory(save_root) as work_path:
        sources: dict[str, Path] = {}

        def torch_artifact(name: str, value: Any) -> None:
            path = work_path / ARTIFACT_FILENAMES[name]
            torch.save(value, path)
            sources[path.name] = path

        def json_artifact(name: str, value: Any) -> None:
            path = work_path / ARTIFACT_FILENAMES[name]
            _write_json_artifact(path, value)
            sources[path.name] = path

        torch_artifact("raw_weights", raw_state)
        torch_artifact("optimizer", optimizer_state)
        torch_artifact("scheduler", scheduler_state)
        torch_artifact("scaler", scaler_state)
        torch_artifact("ema", ema_state)
        json_artifact("rng_json", rng_json_state)
        torch_artifact("rng_torch", rng_torch_state)
        json_artifact("dataloader", dataloader_state)
        cache_archive = work_path / ARTIFACT_FILENAMES["latent_cache"]
        _write_cache_archive(cache_archive, cache_sources)
        sources[cache_archive.name] = cache_archive
        json_artifact("trainer", trainer_state)
        sources[ARTIFACT_FILENAMES["public_checkpoint"]] = public

        metadata = _metadata_for_core(
            core,
            identity,
            completed_step=step,
            next_step=effective_next_step,
            optimizer_updates_completed=optimizer_updates,
        )
        try:
            created = core.create_bundle(
                save_root,
                sources,
                metadata,
                retention=retention,
                reserve_bytes=reserve_bytes,
                max_bundle_bytes=max_bundle_bytes,
                max_store_bytes=max_store_bytes,
            )
        except Exception as exc:
            raise StateBridgeError(f"canonical bundle creation failed: {exc}") from exc
    bundle_id = getattr(created, "bundle_id", None)
    if not isinstance(bundle_id, str):
        raise StateBridgeError("canonical bundle writer returned no bundle_id")
    return save_root / ".lds-state" / bundle_id


def _validated_bundle(bundle_dir: Path) -> tuple[dict[str, Any], Any]:
    core = _bundle_core()
    raw = Path(os.path.abspath(os.fspath(bundle_dir)))
    if raw.is_symlink() or raw.parent.name != ".lds-state":
        raise StateBridgeError("restore directory is not a canonical .lds-state bundle")
    save_root = raw.parent.parent
    try:
        canonical = core.resolve_bundle_path(
            save_root, raw.name, require_exists=True)
        if canonical != raw:
            raise StateBridgeError("restore directory is not canonical")
        inspected = core.verify_bundle(save_root, raw.name)
    except Exception as exc:
        raise StateBridgeError(f"bundle integrity validation failed: {exc}") from exc
    manifest = getattr(inspected, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise StateBridgeError("validated bundle exposed no manifest")
    manifest = dict(manifest)
    if manifest.get("schema") != STATE_SCHEMA or manifest.get("state_level") != "exact":
        raise StateBridgeError("restore bundle is not an exact LDS state bundle")
    return manifest, inspected


def _clear_early_staging() -> None:
    global _EARLY_CACHE_ARCHIVE
    global _EARLY_CACHE_DESCRIPTORS
    global _EARLY_STAGED_ROOT
    global _EARLY_STAGED_SOURCE
    global _EARLY_STAGED_MANIFEST
    global _EARLY_STAGED_PATHS
    global _EARLY_STAGE_OWNER
    owner = _EARLY_STAGE_OWNER
    _EARLY_CACHE_ARCHIVE = None
    _EARLY_CACHE_DESCRIPTORS = {}
    _EARLY_STAGED_ROOT = None
    _EARLY_STAGED_SOURCE = None
    _EARLY_STAGED_MANIFEST = None
    _EARLY_STAGED_PATHS = {}
    _EARLY_STAGE_OWNER = None
    if owner is not None:
        owner.__exit__(None, None, None)


@contextmanager
def _verified_staged_bundle(bundle: os.PathLike[str] | str):
    """Yield only a private, copy-time-rehashed artifact tree."""
    core = _bundle_core()
    raw = Path(os.path.abspath(os.fspath(bundle)))
    manifest, inspection = _validated_bundle(raw)
    _validate_restore_context(manifest)
    save_root = raw.parent.parent
    reserve_bytes = _environment_positive_bytes(
        ENV_RESERVE_BYTES,
        int(getattr(core, "DEFAULT_FREE_SPACE_RESERVE_BYTES", 5 * (1 << 30))),
    )
    try:
        core.preflight_restore_space(
            save_root,
            int(inspection.size_bytes),
            destination=save_root,
            reserve_bytes=reserve_bytes,
            # The private artifact tree remains live while preprocessing cache
            # bytes are materialised.  A second full-bundle allowance is the
            # conservative extraction peak required before any workdir exists.
            copy_multiplier=2,
        )
    except Exception as exc:
        raise StateBridgeError(
            f"restore disk preflight failed: {exc}"
        ) from exc
    owner = _runtime_work_directory(save_root)
    work_root = owner.__enter__()
    try:
        staged_root = work_root / "verified"
        try:
            restored = core.stage_restore(
                save_root,
                raw.name,
                staged_root,
                reserve_bytes=reserve_bytes,
            )
        except Exception as exc:
            raise StateBridgeError(
                f"bundle private staging failed: {exc}") from exc
        inspection = getattr(restored, "inspection", None)
        staged_manifest = getattr(inspection, "manifest", None)
        if not isinstance(staged_manifest, Mapping):
            raise StateBridgeError("private staging exposed no verified manifest")
        staged_manifest = dict(staged_manifest)
        if (
            staged_manifest.get("schema") != STATE_SCHEMA
            or staged_manifest.get("state_level") != "exact"
        ):
            raise StateBridgeError("staged bundle is not exact")
        _validate_restore_context(staged_manifest)
        paths = {
            str(name): Path(path)
            for name, path in getattr(restored, "artifacts", {}).items()
        }
        missing = set(ARTIFACT_FILENAMES.values()) - set(paths)
        if missing:
            raise StateBridgeError(
                "exact bundle is missing artifacts: "
                + ", ".join(sorted(missing))
            )
        yield staged_root, staged_manifest, paths, owner
    finally:
        owner.__exit__(None, None, None)


def _validate_restore_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = _identity()
    expected = {
        "config_hash": identity["config_hash"],
        "dataset_hash": identity["dataset_hash"],
        "base_model_hash": identity["base_hash"],
        "network_hash": identity["network_hash"],
        "toolkit_revision": identity["toolkit_revision"],
    }
    differences = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    expected_runtime = dict(identity["runtime"])
    expected_runtime.update({
        "bridge": BRIDGE_NAME,
        "bridge_version": BRIDGE_VERSION,
        "protocol": BRIDGE_PROTOCOL,
        "shape_revision": SHAPE_REVISION,
    })
    if manifest.get("toolkit_runtime") != expected_runtime:
        differences.append("toolkit_runtime")
    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not set(_RUNTIME_CAPABILITIES).issubset(capabilities)
    ):
        differences.append("capabilities")
    if differences:
        raise StateBridgeError(
            "restore context differs: " + ", ".join(sorted(set(differences)))
        )
    return identity


def _early_dataset_state_from_paths(
    paths: Mapping[str, Path],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]], Path]:
    try:
        dataloader = json.loads(
            paths[ARTIFACT_FILENAMES["dataloader"]].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateBridgeError("saved dataloader state is invalid") from exc
    datasets = dataloader.get("datasets")
    if (
        dataloader.get("schema") != _DATALOADER_STATE_SCHEMA
        or not isinstance(datasets, Mapping)
        or not isinstance(datasets.get("main"), Mapping)
    ):
        raise StateBridgeError("saved dataset restore graph is unavailable")
    queue: list[Mapping[str, Any]] = []
    descriptors: dict[str, Mapping[str, Any]] = {}
    for label in ("main", "regularization"):
        graph = datasets.get(label)
        if graph is None:
            continue
        if (
            not isinstance(graph, Mapping)
            or graph.get("schema") != _DATASET_GRAPH_SCHEMA
            or graph.get("label") != label
            or not isinstance(graph.get("topology"), Mapping)
            or not isinstance(graph.get("leaves"), list)
        ):
            raise StateBridgeError(f"saved {label} dataset graph is invalid")
        for index, state in enumerate(graph["leaves"]):
            if not isinstance(state, Mapping):
                raise StateBridgeError(f"saved {label} dataset state is invalid")
            if state.get("leaf_ordinal") != index:
                raise StateBridgeError(
                    f"saved {label} dataset ordinal is invalid"
                )
            if state.get("mode") == "static":
                continue
            if state.get("schema") != _AITK_DATASET_SCHEMA:
                raise StateBridgeError(
                    f"saved {label} ai-toolkit dataset schema is invalid"
                )
            queue.append({
                "label": label,
                "leaf_ordinal": index,
                "state": state,
            })
            for name, descriptor in _cache_descriptors(state).items():
                previous = descriptors.get(name)
                if previous is not None and previous != descriptor:
                    raise StateBridgeError(
                        "conflicting preprocessing-cache descriptors"
                    )
                descriptors[name] = descriptor
    archive_path = paths[ARTIFACT_FILENAMES["latent_cache"]]
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            _validated_cache_members(archive, descriptors)
    except (OSError, tarfile.TarError) as exc:
        raise StateBridgeError(
            f"preprocessing-cache archive is invalid: {exc}"
        ) from exc
    return queue, descriptors, archive_path


def _prepare_early_dataset_restore(bundle: os.PathLike[str] | str) -> None:
    """Stage and validate dataset/cache state before AiToolkitDataset is built."""
    global _EARLY_CACHE_ARCHIVE
    global _EARLY_CACHE_DESCRIPTORS
    global _EARLY_STAGED_ROOT
    global _EARLY_STAGED_SOURCE
    global _EARLY_STAGED_MANIFEST
    global _EARLY_STAGED_PATHS
    global _EARLY_STAGE_OWNER
    _clear_early_staging()
    _EARLY_DATASET_RESTORE_QUEUE.clear()
    bundle_dir = Path(os.path.abspath(os.fspath(bundle)))
    stage_context = _verified_staged_bundle(bundle_dir)
    entered = False
    try:
        staged_root, manifest, paths, _owner = stage_context.__enter__()
        entered = True
        queue, descriptors, archive_path = _early_dataset_state_from_paths(paths)
    except BaseException:
        # A contextmanager whose __enter__ failed is already closed.  Calling
        # __exit__ in that case can mask the integrity error with a generator
        # lifecycle error.  Once entered, however, parsing failures must tear
        # down the retained private staging tree immediately.
        if entered:
            stage_context.__exit__(*sys.exc_info())
        raise
    _EARLY_DATASET_RESTORE_QUEUE[:] = queue
    _EARLY_CACHE_DESCRIPTORS = descriptors
    _EARLY_CACHE_ARCHIVE = archive_path
    _EARLY_STAGED_ROOT = staged_root
    _EARLY_STAGED_SOURCE = bundle_dir
    _EARLY_STAGED_MANIFEST = manifest
    _EARLY_STAGED_PATHS = dict(paths)
    _EARLY_STAGE_OWNER = stage_context


def _patched_setup_buckets(self: Any, *args: Any, **kwargs: Any):
    result = _ORIGINAL_AITK_SETUP_BUCKETS(self, *args, **kwargs)
    pending = getattr(self, "_lds_pending_dataset_restore", None)
    if isinstance(pending, Mapping) and pending["state"].get("mode") == "buckets":
        _restore_aitk_dataset_state(
            self,
            pending["state"],
            label=pending["label"],
            leaf_index=pending["leaf_ordinal"],
            bucket_objects_ready=True,
        )
        self._lds_bucket_state_restored = True
    return result


def _patched_cache_latents(self: Any, *args: Any, **kwargs: Any):
    pending = getattr(self, "_lds_pending_dataset_restore", None)
    if isinstance(pending, Mapping):
        _materialize_saved_cache_kind(
            self, pending["state"], kind="latent"
        )
    result = _ORIGINAL_AITK_CACHE_LATENTS(self, *args, **kwargs)
    if isinstance(pending, Mapping):
        _verify_materialized_cache_kind(
            self, pending["state"], kind="latent"
        )
        self._lds_latent_cache_restored = True
    return result


def _patched_cache_text(self: Any, *args: Any, **kwargs: Any):
    pending = getattr(self, "_lds_pending_dataset_restore", None)
    if isinstance(pending, Mapping):
        _materialize_saved_cache_kind(
            self, pending["state"], kind="text_embedding"
        )
    result = _ORIGINAL_AITK_CACHE_TEXT(self, *args, **kwargs)
    if isinstance(pending, Mapping):
        _verify_materialized_cache_kind(
            self, pending["state"], kind="text_embedding"
        )
        self._lds_text_cache_restored = True
    return result


def _state_has_cache_kind(saved: Mapping[str, Any], kind: str) -> bool:
    return any(
        isinstance(file_state, Mapping)
        and isinstance(file_state.get("cache"), Mapping)
        and kind in file_state["cache"]
        for file_state in saved.get("files", ())
    )


def _patched_setup_epoch(self: Any, *args: Any, **kwargs: Any):
    if not _EARLY_DATASET_RESTORE_QUEUE:
        if (
            os.environ.get(ENV_RESTORE_DIR)
            and int(getattr(self, "epoch_num", 0)) == 0
            and not bool(getattr(self, "_lds_dataset_state_restored_early", False))
        ):
            raise StateBridgeError(
                "restore bundle has no matching dataset leaf for initial setup"
            )
        return _ORIGINAL_AITK_SETUP_EPOCH(self, *args, **kwargs)

    pending = _EARLY_DATASET_RESTORE_QUEUE.pop(0)
    saved = pending["state"]
    if int(getattr(self, "epoch_num", -1)) != 0:
        raise StateBridgeError(
            "fresh dataset did not enter restore through epoch-zero setup"
        )
    _restore_aitk_dataset_state(
        self,
        saved,
        label=pending["label"],
        leaf_index=pending["leaf_ordinal"],
        bucket_objects_ready=False,
    )
    self._lds_pending_dataset_restore = pending
    self._lds_bucket_state_restored = saved.get("mode") != "buckets"
    self._lds_latent_cache_restored = not _state_has_cache_kind(
        saved, "latent"
    )
    self._lds_text_cache_restored = not _state_has_cache_kind(
        saved, "text_embedding"
    )
    try:
        result = _ORIGINAL_AITK_SETUP_EPOCH(self, *args, **kwargs)
        missing = []
        if not bool(getattr(self, "_lds_bucket_state_restored", False)):
            missing.append("bucket state")
        if not bool(getattr(self, "_lds_latent_cache_restored", False)):
            missing.append("latent cache")
        if not bool(getattr(self, "_lds_text_cache_restored", False)):
            missing.append("text embedding cache")
        if missing:
            raise StateBridgeError(
                "dataset setup skipped restored " + ", ".join(missing)
            )
        self.epoch_num = _bounded_int(
            saved.get("epoch_num"),
            label="saved dataset epoch_num",
        )
        self._lds_dataset_state_restored_early = {
            "label": pending["label"],
            "leaf_ordinal": pending["leaf_ordinal"],
            "identity_sha256": saved.get("identity_sha256"),
        }
        return result
    finally:
        try:
            delattr(self, "_lds_pending_dataset_restore")
        except AttributeError:
            pass


def _skip_scheduler_bootstrap_once(scheduler: Any) -> None:
    original = scheduler.step

    def skip_once(*args: Any, **kwargs: Any):
        scheduler.step = original
        return None

    scheduler.step = skip_once


def _restore_exact_from_staged_paths(
    trainer: Any,
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    raw = _torch_load(paths[ARTIFACT_FILENAMES["raw_weights"]])
    optimizer = _torch_load(paths[ARTIFACT_FILENAMES["optimizer"]])
    scheduler = _torch_load(paths[ARTIFACT_FILENAMES["scheduler"]])
    scaler = _torch_load(paths[ARTIFACT_FILENAMES["scaler"]])
    ema = _torch_load(paths[ARTIFACT_FILENAMES["ema"]])
    torch_rng = _torch_load(paths[ARTIFACT_FILENAMES["rng_torch"]])
    rng_json = json.loads(
        paths[ARTIFACT_FILENAMES["rng_json"]].read_text(encoding="utf-8")
    )
    dataloader = json.loads(
        paths[ARTIFACT_FILENAMES["dataloader"]].read_text(encoding="utf-8")
    )
    trainer_state = json.loads(
        paths[ARTIFACT_FILENAMES["trainer"]].read_text(encoding="utf-8")
    )

    loaders = dataloader.get("loaders", {})
    datasets = dataloader.get("datasets", {})
    if (
        dataloader.get("schema") != _DATALOADER_STATE_SCHEMA
        or not isinstance(loaders, Mapping)
        or not isinstance(datasets, Mapping)
        or _EARLY_DATASET_RESTORE_QUEUE
    ):
        raise StateBridgeError("saved dataloader/dataset state is incomplete")
    _restore_loader_dataset_state(
        getattr(trainer, "data_loader", None),
        datasets.get("main"),
        label="main",
    )
    _restore_loader_dataset_state(
        getattr(trainer, "data_loader_reg", None),
        datasets.get("regularization"),
        label="regularization",
    )
    main = instrument_dataloader(
        getattr(trainer, "data_loader", None), "main", loaders.get("main")
    )
    if main is None:
        raise StateBridgeError("main dataloader cannot be restored")
    instrument_dataloader(
        getattr(trainer, "data_loader_reg", None),
        "regularization",
        loaders.get("regularization"),
    )

    _restore_optimizer_params(trainer.optimizer, raw)
    trainer.optimizer.load_state_dict(optimizer)
    trainer.lr_scheduler.load_state_dict(scheduler)

    current_scaler = getattr(trainer.accelerator, "scaler", None)
    if bool(scaler.get("applicable")) != (current_scaler is not None):
        raise StateBridgeError("GradScaler applicability changed")
    if current_scaler is not None:
        current_scaler.load_state_dict(scaler["state"])
    current_ema = getattr(trainer, "ema", None)
    if bool(ema.get("applicable")) != (current_ema is not None):
        raise StateBridgeError("EMA applicability changed")
    if current_ema is not None:
        current_ema.load_state_dict(ema["state"])

    next_step = int(manifest["next_step"])
    if int(trainer_state["next_step"]) != next_step:
        raise StateBridgeError("trainer artifact and manifest disagree on next_step")
    trainer.step_num = next_step
    trainer.start_step = next_step
    trainer.train_config.start_step = next_step
    trainer.last_save_step = int(manifest["completed_step"])
    trainer._lds_optimizer_updates_completed = int(
        manifest["optimizer_updates_completed"])
    trainer.epoch_num = int(trainer_state["epoch_num"])
    trainer.grad_accumulation_step = int(
        trainer_state["grad_accumulation_step"]
    )
    for attr in ("current_boundary_index", "steps_this_boundary"):
        if trainer_state.get(attr) is not None and hasattr(trainer, attr):
            setattr(trainer, attr, trainer_state[attr])

    _skip_scheduler_bootstrap_once(trainer.lr_scheduler)
    # RNG is restored last: iterator creation and the next training operation
    # must see exactly the state captured after K.
    require_cuda = _trainer_requires_cuda(trainer)
    _restore_rng_json(rng_json)
    _restore_torch_rng(torch_rng, require_cuda)


def restore_exact_bundle(trainer: Any, bundle: os.PathLike[str] | str) -> None:
    """Restore only from a private artifact tree rehashed while it was copied."""
    bundle_dir = Path(os.path.abspath(os.fspath(bundle)))
    reasons = _trainer_gate_reasons(trainer)
    if reasons:
        raise StateBridgeError("; ".join(reasons))
    if _EARLY_STAGE_OWNER is not None:
        if (
            _EARLY_STAGED_SOURCE != bundle_dir
            or not isinstance(_EARLY_STAGED_MANIFEST, Mapping)
            or not _EARLY_STAGED_PATHS
        ):
            raise StateBridgeError(
                "early verified staging belongs to a different restore bundle"
            )
        try:
            _validate_restore_context(_EARLY_STAGED_MANIFEST)
            _restore_exact_from_staged_paths(
                trainer,
                _EARLY_STAGED_MANIFEST,
                _EARLY_STAGED_PATHS,
            )
        finally:
            _clear_early_staging()
        return
    with _verified_staged_bundle(bundle_dir) as (
        _staged_root,
        manifest,
        paths,
        _owner,
    ):
        _restore_exact_from_staged_paths(trainer, manifest, paths)


def _patched_before_loop(self: Any):
    # Restore only after SDTrainer has finished creating unconditional embeds
    # and other tensors that may consume RNG.  The next operation after this
    # complete hook must observe the captured state.
    result = _ORIGINAL_SD_BEFORE_LOOP(self)
    # A weights-only/native ai-toolkit resume intentionally starts a new
    # optimizer trajectory. An exact restore below replaces this with the
    # bundle's real count.
    self._lds_optimizer_updates_completed = 0
    reasons = _trainer_gate_reasons(self)
    if not reasons:
        try:
            instrument_dataloader(getattr(self, "data_loader", None), "main")
            instrument_dataloader(
                getattr(self, "data_loader_reg", None), "regularization"
            )
        except StateBridgeError as exc:
            reasons.append(str(exc))
    restore_dir = os.environ.get(ENV_RESTORE_DIR)
    if restore_dir:
        if reasons:
            message = "; ".join(reasons)
            _write_status(
                "restore_error",
                trainer=self,
                exact_supported=False,
                restore_dir=str(Path(restore_dir).resolve()),
                reasons=[message],
            )
            raise StateBridgeError(message)
        try:
            restore_exact_bundle(self, restore_dir)
        except Exception as exc:
            _write_status(
                "restore_error",
                trainer=self,
                exact_supported=False,
                restore_dir=str(Path(restore_dir).resolve()),
                reasons=[f"{type(exc).__name__}: {exc}"],
            )
            raise
        _write_status(
            "restored",
            trainer=self,
            exact_supported=True,
            restore_dir=str(Path(restore_dir).resolve()),
            next_step=int(self.step_num),
        )
        self._lds_exact_restored = True
        self._lds_training_started_reported = False
    else:
        _write_status(
            "ready" if not reasons else "unsupported",
            trainer=self,
            exact_supported=not reasons,
            reasons=reasons,
        )
    return result


def _patched_sd_train_loop(self: Any, *args: Any, **kwargs: Any):
    result = _ORIGINAL_SD_TRAIN_LOOP(self, *args, **kwargs)
    if not bool(getattr(self, "is_grad_accumulation_step", True)):
        self._lds_last_optimizer_boundary_step = int(self.step_num)
        self._lds_optimizer_updates_completed = int(
            getattr(self, "_lds_optimizer_updates_completed", 0)
        ) + 1
        if (
            bool(getattr(self, "_lds_exact_restored", False))
            and not bool(getattr(self, "_lds_training_started_reported", False))
        ):
            self._lds_training_started_reported = True
            _write_status(
                "training_started",
                trainer=self,
                exact_supported=True,
                optimizer_updates_completed=self._lds_optimizer_updates_completed,
                step_num=int(self.step_num),
            )
    else:
        self._lds_last_optimizer_boundary_step = None
    return result


def _patched_save(self: Any, step: Any = None):
    result = _ORIGINAL_BASE_SAVE(self, step)
    if not bool(getattr(self.accelerator, "is_main_process", True)):
        return result
    if step is None:
        completed = int(getattr(self, "step_num", -1))
        boundary = completed - 1
        try:
            bundle = capture_exact_bundle(
                self,
                completed,
                next_step=completed,
                optimizer_boundary_step=boundary,
                numbered_checkpoint=False,
            )
            _write_status(
                "saved",
                trainer=self,
                exact_supported=True,
                latest_bundle=str(bundle),
                completed_step=completed,
                next_step=completed,
                terminal=True,
            )
        except Exception as exc:
            _write_status(
                "save_unsupported",
                trainer=self,
                exact_supported=False,
                reasons=[f"{type(exc).__name__}: {exc}"],
                completed_step=completed,
                terminal=True,
            )
            if os.environ.get(ENV_STRICT) == "1":
                raise
            print(f"[LDS bridge] exact final state not published: {exc}", file=sys.stderr)
        return result
    # save() runs before optional sampling/logging and before ai-toolkit updates
    # step_num to K+1.  Publish only from end_step_hook so RNG and cursor describe
    # the actual boundary observed by the next iteration.
    if getattr(self, "_lds_last_optimizer_boundary_step", None) == int(step):
        self._lds_pending_exact_step = int(step)
    else:
        _write_status(
            "save_unsupported",
            trainer=self,
            exact_supported=False,
            reasons=["save is not immediately after a confirmed optimizer boundary"],
            completed_step=int(step),
        )
    return result


def _patched_end_step(self: Any):
    result = _ORIGINAL_BASE_END_STEP(self)
    step = getattr(self, "_lds_pending_exact_step", None)
    if step is None:
        return result
    self._lds_pending_exact_step = None
    try:
        if int(getattr(self, "step_num", -1)) != int(step) + 1:
            raise StateBridgeError("end-step next_step does not equal K+1")
        bundle = capture_exact_bundle(self, int(step))
        _write_status(
            "saved",
            trainer=self,
            exact_supported=True,
            latest_bundle=str(bundle),
            completed_step=int(step),
            next_step=int(step) + 1,
        )
    except Exception as exc:
        _write_status(
            "save_unsupported",
            trainer=self,
            exact_supported=False,
            reasons=[f"{type(exc).__name__}: {exc}"],
            completed_step=int(step),
        )
        if os.environ.get(ENV_STRICT) == "1":
            raise
        print(f"[LDS bridge] exact state not published: {exc}", file=sys.stderr)
    return result


def install_from_environment() -> dict[str, Any]:
    """Validate and install the reversible monkeypatches once per process."""
    global _INSTALLED
    global _SOURCE_PROBE
    global _ORIGINAL_BASE_SAVE
    global _ORIGINAL_BASE_END_STEP
    global _ORIGINAL_SD_BEFORE_LOOP
    global _ORIGINAL_SD_TRAIN_LOOP
    global _ORIGINAL_AITK_SETUP_BUCKETS
    global _ORIGINAL_AITK_SETUP_EPOCH
    global _ORIGINAL_AITK_CACHE_LATENTS
    global _ORIGINAL_AITK_CACHE_TEXT

    if _INSTALLED:
        return _SOURCE_PROBE
    protocol = os.environ.get(ENV_PROTOCOL)
    if protocol != str(BRIDGE_PROTOCOL):
        raise StateBridgeError(
            f"bridge protocol mismatch ({protocol!r} != {BRIDGE_PROTOCOL!r})"
        )
    root = Path(os.environ.get(ENV_AITK_ROOT) or os.getcwd()).resolve()
    _SOURCE_PROBE = probe_aitoolkit_source(root)
    if not _SOURCE_PROBE.get("supported"):
        raise StateBridgeError("; ".join(_SOURCE_PROBE.get("reasons") or []))
    # sitecustomize runs while Python is still initialising ``sys.path``; the
    # script directory is not reliably present yet (notably on Windows).
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from jobs.process import BaseSDTrainProcess
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.data_loader import AiToolkitDataset

    restore_dir = os.environ.get(ENV_RESTORE_DIR)
    _EARLY_DATASET_RESTORE_QUEUE.clear()
    _clear_early_staging()
    if restore_dir:
        _prepare_early_dataset_restore(restore_dir)

    if tuple(inspect.signature(BaseSDTrainProcess.save).parameters) != ("self", "step"):
        raise StateBridgeError("BaseSDTrainProcess.save signature changed")
    if tuple(inspect.signature(BaseSDTrainProcess.end_step_hook).parameters) != (
        "self",
    ):
        raise StateBridgeError("end_step_hook signature changed")
    if tuple(inspect.signature(SDTrainer.hook_before_train_loop).parameters) != (
        "self",
    ):
        raise StateBridgeError("SDTrainer.hook_before_train_loop signature changed")
    sd_loop_params = tuple(inspect.signature(SDTrainer.hook_train_loop).parameters)
    if sd_loop_params[:2] != ("self", "batch"):
        raise StateBridgeError("SDTrainer.hook_train_loop signature changed")
    if tuple(inspect.signature(AiToolkitDataset.setup_epoch).parameters) != (
        "self",
    ):
        raise StateBridgeError("AiToolkitDataset.setup_epoch signature changed")
    bucket_params = tuple(
        inspect.signature(AiToolkitDataset.setup_buckets).parameters
    )
    if bucket_params != ("self", "quiet"):
        raise StateBridgeError("AiToolkitDataset.setup_buckets signature changed")
    if tuple(
        inspect.signature(AiToolkitDataset.cache_latents_all_latents).parameters
    ) != ("self",):
        raise StateBridgeError(
            "AiToolkitDataset.cache_latents_all_latents signature changed"
        )
    if tuple(
        inspect.signature(AiToolkitDataset.cache_text_embeddings).parameters
    ) != ("self",):
        raise StateBridgeError(
            "AiToolkitDataset.cache_text_embeddings signature changed"
        )

    _ORIGINAL_BASE_SAVE = BaseSDTrainProcess.save
    _ORIGINAL_BASE_END_STEP = BaseSDTrainProcess.end_step_hook
    _ORIGINAL_SD_BEFORE_LOOP = SDTrainer.hook_before_train_loop
    _ORIGINAL_SD_TRAIN_LOOP = SDTrainer.hook_train_loop
    _ORIGINAL_AITK_SETUP_BUCKETS = AiToolkitDataset.setup_buckets
    _ORIGINAL_AITK_SETUP_EPOCH = AiToolkitDataset.setup_epoch
    _ORIGINAL_AITK_CACHE_LATENTS = AiToolkitDataset.cache_latents_all_latents
    _ORIGINAL_AITK_CACHE_TEXT = AiToolkitDataset.cache_text_embeddings
    BaseSDTrainProcess.save = _patched_save
    BaseSDTrainProcess.end_step_hook = _patched_end_step
    SDTrainer.hook_before_train_loop = _patched_before_loop
    SDTrainer.hook_train_loop = _patched_sd_train_loop
    AiToolkitDataset.setup_buckets = _patched_setup_buckets
    AiToolkitDataset.setup_epoch = _patched_setup_epoch
    AiToolkitDataset.cache_latents_all_latents = _patched_cache_latents
    AiToolkitDataset.cache_text_embeddings = _patched_cache_text
    DataLoader.__iter__ = _tracked_dataloader_iter
    _INSTALLED = True
    _write_status(
        "patched",
        exact_supported=None,
        reasons=["trainer capabilities are checked after optimizer/dataloader setup"],
    )
    return _SOURCE_PROBE


def uninstall_for_tests() -> None:
    """Rollback monkeypatches (production rollback is removing the env overlay)."""
    global _INSTALLED
    if not _INSTALLED:
        _EARLY_DATASET_RESTORE_QUEUE.clear()
        _clear_early_staging()
        return
    from jobs.process import BaseSDTrainProcess
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from toolkit.data_loader import AiToolkitDataset

    BaseSDTrainProcess.save = _ORIGINAL_BASE_SAVE
    BaseSDTrainProcess.end_step_hook = _ORIGINAL_BASE_END_STEP
    SDTrainer.hook_before_train_loop = _ORIGINAL_SD_BEFORE_LOOP
    SDTrainer.hook_train_loop = _ORIGINAL_SD_TRAIN_LOOP
    AiToolkitDataset.setup_buckets = _ORIGINAL_AITK_SETUP_BUCKETS
    AiToolkitDataset.setup_epoch = _ORIGINAL_AITK_SETUP_EPOCH
    AiToolkitDataset.cache_latents_all_latents = _ORIGINAL_AITK_CACHE_LATENTS
    AiToolkitDataset.cache_text_embeddings = _ORIGINAL_AITK_CACHE_TEXT
    DataLoader.__iter__ = _ORIGINAL_DATALOADER_ITER
    _EARLY_DATASET_RESTORE_QUEUE.clear()
    _clear_early_staging()
    _INSTALLED = False
