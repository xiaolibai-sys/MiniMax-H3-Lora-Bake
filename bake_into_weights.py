"""Offline-bake a complete MiniMax H3 pruned LoRA into a base model.

The script preserves ComfyUI's safetensors quantization layout.  Quantized
weights are dequantized through comfy_kitchen's QuantizedTensor, the LoRA is
folded on the same device used by inference, then the result is requantized
with the original layout and serialized back as ComfyUI weight + weight_scale
+ comfy_quant entries.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")
_TOKEN_RE = re.compile(r"^token_refiner\.blocks\.(\d+)\.")

_LAYOUT_BY_FORMAT = {
    "float8_e4m3fn": "TensorCoreFP8E4M3Layout",
    "float8_e5m2": "TensorCoreFP8E5M2Layout",
    "nvfp4": "TensorCoreNVFP4Layout",
    "int8_tensorwise": "TensorWiseINT8Layout",
    "convrot_w4a4": "TensorCoreConvRotW4A4Layout",
    "mxfp8": "TensorCoreMXFP8Layout",
}

_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
}

_PREAD_LOCKS: dict[int, threading.Lock] = {}


def _pread(fd: int, n: int, offset: int) -> bytes:
    if hasattr(os, "pread"):
        return os.pread(fd, n, offset)
    try:
        import msvcrt
        import pywintypes
        import win32file
        h = msvcrt.get_osfhandle(fd)
        ov = pywintypes.OVERLAPPED()
        ov.Offset = offset & 0xFFFFFFFF
        ov.OffsetHigh = (offset >> 32) & 0xFFFFFFFF
        try:
            _, data = win32file.ReadFile(h, n, ov)
        except pywintypes.error as e:
            if e.winerror != win32file.ERROR_IO_PENDING:
                raise
            _, data = win32file.GetOverlappedResult(h, ov, True)
        return data
    except ImportError:
        lock = _PREAD_LOCKS.setdefault(fd, threading.Lock())
        with lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, n)


class SafetensorsReader:
    """Positioned safetensors reader that avoids mmap offset issues on Windows."""

    def __init__(self, path: str):
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        self._fd = os.open(path, flags)
        self._header_len = struct.unpack("<Q", os.read(self._fd, 8))[0]
        self._header = json.loads(os.read(self._fd, self._header_len))
        self._data_offset = 8 + self._header_len
        self._keys = [
            k for k, v in self._header.items()
            if isinstance(v, dict) and "data_offsets" in v
        ]

    def keys(self) -> list[str]:
        return list(self._keys)

    def metadata(self) -> dict | None:
        meta = self._header.get("__metadata__")
        return dict(meta) if isinstance(meta, dict) else None

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self._header[name]
        shape = list(info["shape"])
        dtype = _DTYPE_MAP[info["dtype"]]
        begin, end = info["data_offsets"]
        raw = _pread(self._fd, end - begin, self._data_offset + begin)
        return torch.frombuffer(raw, dtype=dtype).reshape(shape)

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


@dataclass
class LoraInput:
    path: str
    strength: float = 1.0


@dataclass
class LoraEntry:
    target: str
    a: Any = None
    b: Any = None
    alpha: Any = None
    strength: float = 1.0
    diff: Any = None
    diff_b: Any = None


@dataclass
class BakeConfig:
    base_model: str
    loras: list[LoraInput] = field(default_factory=list)
    output: str = ""
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    fold_dtype: str = "float32"
    convrot_groupsize: int = 256
    comfy_root: str = ""
    max_blocks: int = 0


def load_config(path: str) -> BakeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    loras_raw = data.get("loras") or []
    if not loras_raw and data.get("lora"):
        loras_raw = [
            {"path": data["lora"], "strength": data.get("strength", 1.0)}
        ]
    loras = [
        LoraInput(path=item["path"], strength=float(item.get("strength", 1.0)))
        for item in loras_raw
    ]
    cfg = BakeConfig(
        base_model=str(data["base_model"]),
        loras=loras,
        output=str(data["output"]),
        device=str(data.get("device", "cuda")),
        compute_dtype=str(data.get("compute_dtype", "bfloat16")),
        fold_dtype=str(data.get("fold_dtype", "float32")),
        convrot_groupsize=int(data.get("convrot_groupsize", 256)),
        comfy_root=str(data.get("comfy_root", "")),
        max_blocks=int(data.get("max_blocks", 0)),
    )
    if not cfg.base_model:
        raise ValueError("config.base_model is required")
    if not cfg.loras:
        raise ValueError("config.loras is required")
    if not cfg.output:
        raise ValueError("config.output is required")
    if Path(cfg.base_model).resolve() == Path(cfg.output).resolve():
        raise ValueError("output must not overwrite base_model")
    return cfg


def setup_paths(cfg: BakeConfig) -> None:
    root = Path(cfg.comfy_root or os.environ.get("COMFY_ROOT", ""))
    if not root.is_absolute():
        root = Path(r"D:\ComfyUI-installs\ComfyUI\ComfyUI")
    node_root = root / "custom_nodes" / "ComfyUI-MiniMaxH3"
    if not node_root.is_dir():
        raise FileNotFoundError(f"ComfyUI-MiniMaxH3 not found under {node_root}")
    sys.path.insert(0, str(node_root))
    sys.path.insert(0, str(root))


def parse_lora(h: Any, lora_input: LoraInput) -> Any:
    return h(lora_input.path, strength=lora_input.strength, silu_grid_path="")


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _read_json_bytes(sf: Any, key: str) -> dict:
    raw = bytes(sf.get_tensor(key).tolist())
    return json.loads(raw.decode("utf-8"))


def _make_quantized_tensor(
    sf: Any,
    weight_key: str,
    conf: dict,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> Any:
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class

    fmt = conf["format"]
    layout_name = _LAYOUT_BY_FORMAT.get(fmt)
    if layout_name is None:
        raise ValueError(f"Unsupported ComfyUI quant format: {fmt}")
    layout_cls = get_layout_class(layout_name)
    base = weight_key[: -len(".weight")]
    qdata = sf.get_tensor(weight_key).to(device)
    shape = tuple(qdata.shape)

    if fmt == "int8_tensorwise":
        scale = sf.get_tensor(base + ".weight_scale").to(device)
        params = layout_cls.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=shape,
            is_weight=True,
            convrot=bool(conf.get("convrot", False)),
            convrot_groupsize=int(
                conf.get("convrot_groupsize", 256)
            ),
            transposed=False,
        )
    elif fmt in ("float8_e4m3fn", "float8_e5m2", "mxfp8"):
        scale = sf.get_tensor(base + ".weight_scale").to(device)
        params = layout_cls.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=shape,
        )
    elif fmt == "nvfp4":
        tensor_scale = sf.get_tensor(base + ".weight_scale_2").to(device)
        block_scale = sf.get_tensor(base + ".weight_scale").to(device)
        params = layout_cls.Params(
            scale=tensor_scale,
            orig_dtype=compute_dtype,
            orig_shape=shape,
            block_scale=block_scale,
        )
    elif fmt == "convrot_w4a4":
        scale = sf.get_tensor(base + ".weight_scale").to(device)
        params = layout_cls.Params(
            scale=scale,
            orig_dtype=compute_dtype,
            orig_shape=shape,
            convrot_groupsize=int(conf.get("convrot_groupsize", 256)),
            quant_group_size=int(conf.get("quant_group_size", 64)),
            linear_dtype=str(conf.get("linear_dtype", "int4")),
        )
    else:
        raise ValueError(f"Unsupported ComfyUI quant format: {fmt}")

    return QuantizedTensor(qdata, layout_name, params)


def _standardize_key(k: str) -> str:
    for pre in ("transformer.", "pipe.dit.", "base_model.model.", "diffusion_model."):
        while k.startswith(pre):
            k = k[len(pre):]

    if k.startswith("lora_unet_") and not k.startswith("lora_unet__"):
        body = k[len("lora_unet_"):]
        parts = body.split(".")
        tokens = parts[0].split("_")
        path_tokens: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "blocks" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["blocks", tokens[i + 1]])
                i += 2
                continue
            if t in ("self", "cross") and i + 1 < len(tokens) and tokens[i + 1] == "attn":
                path_tokens.append(f"{t}_attn")
                i += 2
                continue
            if t in ("q", "k", "v", "o"):
                path_tokens.append(t)
                i += 1
                continue
            if t == "ffn" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
                path_tokens.extend(["ffn", tokens[i + 1]])
                i += 2
                continue
            path_tokens.append(t)
            i += 1
        k = ".".join(path_tokens) + ("." + ".".join(parts[1:]) if len(parts) > 1 else "")

    k = k.replace("self_attn.q.", "attn.q.")
    k = k.replace("self_attn.k.", "attn.k.")
    k = k.replace("self_attn.v.", "attn.v.")
    k = k.replace("self_attn.o.", "attn.out_proj.")
    k = k.replace(".lora_down.weight", ".lora_A.weight")
    k = k.replace(".lora_up.weight", ".lora_B.weight")
    return k


def _group_lora_parts(sd: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        base = None
        kind = None
        for suffix, kk in (
            (".lora_A.weight", "a"),
            (".lora_B.weight", "b"),
            (".alpha", "alpha"),
            (".diff_b", "diff_b"),
            (".diff", "diff"),
        ):
            if k.endswith(suffix):
                base = k[: -len(suffix)] + ".weight"
                kind = kk
                break
        if base is None:
            continue
        groups.setdefault(base, {})[kind] = v
    return groups


def _make_lora_entry(
    base: str, parts: dict, strength: float, target: str
) -> LoraEntry:
    return LoraEntry(
        target=target,
        a=parts.get("a"),
        b=parts.get("b"),
        alpha=parts.get("alpha"),
        strength=strength,
        diff=parts.get("diff"),
        diff_b=parts.get("diff_b"),
    )


def _parse_lora(path: str, strength: float) -> Any:
    from safetensors.torch import load_file

    raw = load_file(path)
    sd = {_standardize_key(k): v for k, v in raw.items()}
    groups = _group_lora_parts(sd)

    block_groups: dict[int, list[LoraEntry]] = {}
    token_groups: dict[int, list[LoraEntry]] = {}
    final_entries: list[LoraEntry] = []

    for base, parts in groups.items():
        if base.startswith("blocks."):
            rest = base[len("blocks."):]
            idx_s, _, comp = rest.partition(".")
            if not idx_s.isdigit():
                continue
            comp = comp[: -len(".weight")] if comp.endswith(".weight") else comp
            block_groups.setdefault(int(idx_s), []).append(
                _make_lora_entry(base, parts, strength, comp)
            )
        elif base.startswith("token_refiner.blocks."):
            rest = base[len("token_refiner.blocks."):]
            idx_s, _, comp = rest.partition(".")
            if not idx_s.isdigit():
                continue
            comp = comp[: -len(".weight")] if comp.endswith(".weight") else comp
            token_groups.setdefault(int(idx_s), []).append(
                _make_lora_entry(base, parts, strength, comp)
            )
        elif base == "final_layer.adaln_proj.linear.weight":
            final_entries.append(
                _make_lora_entry(base, parts, strength, "adaln_proj.linear")
            )

    table = None
    block_w: dict[int, torch.Tensor] = {}
    block_b: dict[int, torch.Tensor] = {}
    final_w = None
    final_b = None
    for k, v in sd.items():
        if k == "adaln_t_table":
            table = v
        elif k.startswith("blocks.") and ".adaln_proj.linear.weight" in k:
            block_w[int(k.split(".")[1])] = v
        elif k.startswith("blocks.") and ".adaln_proj.linear.bias" in k:
            block_b[int(k.split(".")[1])] = v
        elif k == "final_layer.adaln_proj.linear.weight":
            final_w = v
        elif k == "final_layer.adaln_proj.linear.bias":
            final_b = v

    override = None
    if table is not None:
        override = SimpleNamespace(
            table=table,
            block_weights=block_w,
            block_biases=block_b,
            final_weight=final_w,
            final_bias=final_b,
        )
    return SimpleNamespace(
        block_groups=block_groups,
        token_refiner_groups=token_groups,
        final_adaln_entries=final_entries,
        adaln_override=override,
    )


def _lora_delta(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: Any,
    strength: float,
    base_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, int]:
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if base_shape is not None and len(base_shape) == 2:
        out, in_ = base_shape
        if b.shape[1] == a.shape[0] and (b @ a).shape == (out, in_):
            delta, rank = b @ a, a.shape[0]
        elif a.shape[1] == b.shape[0] and (a @ b).T.shape == (out, in_):
            delta, rank = (a @ b).T, a.shape[1]
        else:
            delta, rank = b @ a, a.shape[0]
    else:
        if a.shape[1] == b.shape[0]:
            delta, rank = (a @ b).T, a.shape[1]
        else:
            delta, rank = b @ a, a.shape[0]
    if alpha is not None:
        try:
            alpha_val = float(alpha.item() if alpha.numel() == 1 else alpha)
        except Exception:
            alpha_val = float(rank)
    else:
        alpha_val = float(rank)
    return delta * (strength * (alpha_val / rank)), rank


def _chunk_add_delta(w: torch.Tensor, entry: LoraEntry, chunk_rows: int = 8192) -> None:
    if entry.a is None or entry.b is None:
        return
    a = entry.a.to(device=w.device, dtype=torch.float32)
    n_rows = w.shape[0]
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        rows = slice(start, end)
        b = entry.b[rows].to(device=w.device, dtype=torch.float32)
        d, _ = _lora_delta(a, b, entry.alpha, entry.strength, (end - start, w.shape[1]))
        w[rows] += d


def _fold_standard_chunked(
    w: torch.Tensor, entries: list[LoraEntry], qkv_idx: int | None = None
) -> torch.Tensor:
    target = w
    if qkv_idx is not None:
        hd = w.shape[0] // 3
        target = w[slice(qkv_idx * hd, (qkv_idx + 1) * hd)]
    for e in entries:
        _chunk_add_delta(target, e)
    return w


def _fold_dora_chunked(
    w: torch.Tensor, entries: list[LoraEntry], qkv_idx: int | None = None
) -> torch.Tensor:
    target = w
    if qkv_idx is not None:
        hd = w.shape[0] // 3
        target = w[slice(qkv_idx * hd, (qkv_idx + 1) * hd)]
    for e in entries:
        if e.diff_b is not None:
            before_norm = target.norm(dim=1, keepdim=True).clamp(min=1e-8)
            _chunk_add_delta(target, e)
            after_norm = target.norm(dim=1, keepdim=True).clamp(min=1e-8)
            diff_b = e.diff_b.to(device=target.device, dtype=target.dtype)
            m = (before_norm + diff_b.float().reshape(-1, 1)).clamp(min=0.0)
            target = m * target / after_norm
        else:
            _chunk_add_delta(target, e)
    if qkv_idx is not None:
        hd = w.shape[0] // 3
        w[slice(qkv_idx * hd, (qkv_idx + 1) * hd)] = target
    return w


def _fold_entries(
    w: torch.Tensor, entries: list[LoraEntry], qkv_idx: int | None = None
) -> torch.Tensor:
    w = w.float()
    if w.dim() == 2:
        if any(e.diff_b is not None for e in entries):
            return _fold_dora_chunked(w, entries, qkv_idx)
        if not any(e.diff is not None for e in entries):
            return _fold_standard_chunked(w, entries, qkv_idx)
    for e in entries:
        before_norm = None
        if e.diff_b is not None and w.dim() == 2:
            before_norm = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if e.a is not None and e.b is not None:
            d, _ = _lora_delta(e.a, e.b, e.alpha, e.strength, tuple(w.shape))
            w = w + d.to(device=w.device, dtype=w.dtype)
        if before_norm is not None:
            diff_b = e.diff_b.to(device=w.device, dtype=w.dtype)
            m = (before_norm + diff_b.float().reshape(-1, 1)).clamp(min=0.0)
            w = m * w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    diff = next((e.diff for e in entries if e.diff is not None), None)
    if diff is not None and w.dim() == 1:
        w = w + diff.to(device=w.device, dtype=w.dtype).float()
    return w


def _resolve_entries_for_key(
    key: str, entries: list[LoraEntry]
) -> tuple[list[LoraEntry], int | None]:
    out: list[LoraEntry] = []
    qkv_idx: int | None = None
    for e in entries:
        if key.endswith(f".{e.target}.weight"):
            out.append(e)
            continue
        if e.target in ("attn.q", "attn.k", "attn.v") and key.endswith(
            ".attn.qkv_proj.weight"
        ):
            out.append(e)
            qkv_idx = {"attn.q": 0, "attn.k": 1, "attn.v": 2}[e.target]
            continue
        if e.target == "attn.o" and key.endswith(".attn.out_proj.weight"):
            out.append(e)
    return out, qkv_idx


def _comfy_quant_bytes(qt: Any, orig_conf: dict) -> torch.Tensor:
    conf = dict(orig_conf)
    layout = qt._layout_cls
    if layout == "TensorWiseINT8Layout":
        conf["format"] = "int8_tensorwise"
        conf["convrot"] = bool(getattr(qt._params, "convrot", False))
        if conf["convrot"]:
            conf["convrot_groupsize"] = int(
                getattr(qt._params, "convrot_groupsize", 256)
            )
    elif layout == "TensorCoreFP8E4M3Layout":
        conf["format"] = "float8_e4m3fn"
    elif layout == "TensorCoreFP8E5M2Layout":
        conf["format"] = "float8_e5m2"
    elif layout == "TensorCoreNVFP4Layout":
        conf["format"] = "nvfp4"
    elif layout == "TensorCoreConvRotW4A4Layout":
        conf["format"] = "convrot_w4a4"
        conf["convrot_groupsize"] = int(
            getattr(qt._params, "convrot_groupsize", 256)
        )
        linear_dtype = getattr(qt._params, "linear_dtype", "int4")
        if linear_dtype != "int4":
            conf["linear_dtype"] = linear_dtype
    elif layout == "TensorCoreMXFP8Layout":
        conf["format"] = "mxfp8"
    else:
        raise ValueError(f"Unsupported layout for serialization: {layout}")
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _process_block(
    sf: Any,
    keys: list[str],
    prefix: str,
    idx: int,
    entries: list[LoraEntry],
    out_sd: dict[str, torch.Tensor],
    cfg: BakeConfig,
) -> None:
    compute_dtype = _torch_dtype(cfg.compute_dtype)
    fold_dtype = _torch_dtype(cfg.fold_dtype)
    device = torch.device(cfg.device)
    conf_map: dict[str, dict] = {}
    for k in keys:
        if k.startswith(prefix) and k.endswith(".comfy_quant"):
            conf_map[k[: -len(".comfy_quant")]] = _read_json_bytes(sf, k)

    for k in keys:
        if not k.startswith(prefix):
            continue
        if k.endswith(".comfy_quant"):
            continue
        if k.endswith(".weight_scale"):
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in conf_map:
            conf = conf_map[k[: -len(".weight")]]
            qt = _make_quantized_tensor(sf, k, conf, compute_dtype, device)
            key_entries, qkv_idx = _resolve_entries_for_key(k, entries)
            if key_entries:
                w = qt.dequantize().to(fold_dtype)
                if qkv_idx is not None:
                    hd = w.shape[0] // 3
                    rows = slice(qkv_idx * hd, (qkv_idx + 1) * hd)
                    w[rows] = _fold_entries(w[rows], key_entries, qkv_idx)
                else:
                    w = _fold_entries(w, key_entries, qkv_idx)
                qt = qt.requantize_from_float(
                    w.to(qt.dtype), scale="recalculate"
                )
            base = k[: -len(".weight")]
            out_sd[k] = qt._qdata.detach().to("cpu")
            out_sd[base + ".weight_scale"] = qt._params.scale.detach().to("cpu")
            out_sd[base + ".comfy_quant"] = _comfy_quant_bytes(qt, conf)
            continue
        data = sf.get_tensor(k).to(device)
        key_entries, qkv_idx = _resolve_entries_for_key(k, entries)
        if key_entries and k.endswith(".weight"):
            orig_dtype = data.dtype
            data = data.to(fold_dtype)
            data = _fold_entries(data, key_entries, qkv_idx)
            data = data.to(orig_dtype)
        out_sd[k] = data.detach().to("cpu")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _read_base_adaln(
    sf: Any, keys: list[str]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.dtype]]:
    values: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    for k in keys:
        if (
            k == "adaln_t_table"
            or (k.startswith("blocks.") and ".adaln_proj.linear." in k)
            or (k.startswith("final_layer.") and ".adaln_proj.linear." in k)
        ):
            values[k] = sf.get_tensor(k).to("cpu")
            dtypes[k] = values[k].dtype
    return values, dtypes


def _merge_adaln_overrides(
    parsed_loras: list[Any],
    strengths: list[float],
    base_adaln: dict[str, torch.Tensor],
) -> Any:
    overrides = []
    for h, strength in zip(parsed_loras, strengths):
        if h.adaln_override is None:
            raise ValueError(
                "Only complete pruned LoRA files can be baked into a pruned "
                "base. Convert the original LoRA first."
            )
        overrides.append((h.adaln_override, strength))
    if not overrides:
        raise ValueError("No AdaLN override found in LoRA input")

    table = overrides[0][0].table.detach().to("cpu")
    for ov, _ in overrides[1:]:
        if ov.table.shape != table.shape or (
            ov.table.to("cpu") - table
        ).abs().max().item() > 1e-4:
            raise ValueError(
                "Multiple LoRAs use different adaln_t_table; they are not "
                "linearly combinable. Bake one combined LoRA first."
            )

    def _delta(base: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.numel() != base.numel():
            raise ValueError(
                f"AdaLN override shape mismatch: {tuple(value.shape)} != "
                f"{tuple(base.shape)}"
            )
        return value.float() - base

    def combine_block(base_key: str, idx: int) -> torch.Tensor:
        base = base_adaln[base_key].float()
        out = base.clone()
        for ov, strength in overrides:
            values = (
                ov.block_weights
                if base_key.endswith(".weight")
                else ov.block_biases
            )
            if idx not in values:
                continue
            out = out + strength * _delta(base, values[idx])
        return out

    def combine_final(base_key: str) -> torch.Tensor:
        base = base_adaln[base_key].float()
        out = base.clone()
        for ov, strength in overrides:
            values = (
                ov.final_weight
                if base_key.endswith(".weight")
                else ov.final_bias
            )
            if values is None:
                continue
            out = out + strength * _delta(base, values)
        return out

    block_w: dict[int, torch.Tensor] = {}
    block_b: dict[int, torch.Tensor] = {}
    block_idxs = sorted(
        {
            int(k.split(".")[1])
            for k in base_adaln
            if k.startswith("blocks.") and k.endswith(".adaln_proj.linear.weight")
        }
    )
    for idx in block_idxs:
        wk = f"blocks.{idx}.adaln_proj.linear.weight"
        bk = f"blocks.{idx}.adaln_proj.linear.bias"
        if wk not in base_adaln or bk not in base_adaln:
            continue
        block_w[idx] = combine_block(wk, idx)
        block_b[idx] = combine_block(bk, idx)

    final_w = combine_final("final_layer.adaln_proj.linear.weight")
    final_b = combine_final("final_layer.adaln_proj.linear.bias")

    return SimpleNamespace(
        table=table,
        block_weights=block_w,
        block_biases=block_b,
        final_weight=final_w,
        final_bias=final_b,
    )


def _apply_override(
    out_sd: dict[str, torch.Tensor],
    override: Any,
    base_dtypes: dict[str, torch.dtype],
) -> None:
    table_dtype = base_dtypes.get("adaln_t_table", torch.float32)
    out_sd["adaln_t_table"] = override.table.to(table_dtype)
    for idx, w in override.block_weights.items():
        wk = f"blocks.{idx}.adaln_proj.linear.weight"
        bk = f"blocks.{idx}.adaln_proj.linear.bias"
        out_sd[wk] = w.to(base_dtypes.get(wk, torch.float16))
        if idx in override.block_biases:
            out_sd[bk] = override.block_biases[idx].to(
                base_dtypes.get(bk, torch.float16)
            )
    if override.final_weight is not None:
        wk = "final_layer.adaln_proj.linear.weight"
        bk = "final_layer.adaln_proj.linear.bias"
        out_sd[wk] = override.final_weight.to(base_dtypes.get(wk, torch.float16))
        if override.final_bias is not None:
            out_sd[bk] = override.final_bias.to(base_dtypes.get(bk, torch.float16))


def _indices(keys: list[str], regex: re.Pattern[str]) -> list[int]:
    seen: set[int] = set()
    for k in keys:
        m = regex.match(k)
        if m:
            seen.add(int(m.group(1)))
    return sorted(seen)


def bake(cfg: BakeConfig) -> None:
    start = time.time()
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parsed_loras = [
        _parse_lora(item.path, strength=item.strength)
        for item in cfg.loras
    ]
    block_groups: dict[int, list] = {}
    token_groups: dict[int, list] = {}
    for h in parsed_loras:
        for idx, entries in h.block_groups.items():
            block_groups.setdefault(idx, []).extend(entries)
        for idx, entries in h.token_refiner_groups.items():
            token_groups.setdefault(idx, []).extend(entries)

    out_sd: dict[str, torch.Tensor] = {}
    reader = SafetensorsReader(cfg.base_model)
    try:
        keys = reader.keys()
        metadata = reader.metadata()
        base_adaln, base_dtypes = _read_base_adaln(reader, keys)
        override = _merge_adaln_overrides(
            parsed_loras, [item.strength for item in cfg.loras], base_adaln
        )

        for idx in _indices(keys, _BLOCK_RE):
            if cfg.max_blocks and idx >= cfg.max_blocks:
                continue
            entries = block_groups.get(idx, [])
            _process_block(
                reader, keys, f"blocks.{idx}.", idx, entries, out_sd, cfg
            )
            print(
                f"baked blocks.{idx} "
                f"({time.time() - start:.1f}s, "
                f"{len(out_sd)} keys staged)",
                flush=True,
            )

        for idx in _indices(keys, _TOKEN_RE):
            entries = token_groups.get(idx, [])
            _process_block(
                reader,
                keys,
                f"token_refiner.blocks.{idx}.",
                idx,
                entries,
                out_sd,
                cfg,
            )
            print(f"baked token_refiner.blocks.{idx}", flush=True)

        for k in keys:
            if k.startswith("blocks.") or k.startswith("token_refiner.blocks."):
                continue
            if (
                k == "adaln_t_table"
                or ".adaln_proj.linear." in k
            ):
                continue
            out_sd[k] = reader.get_tensor(k).to("cpu")

        _apply_override(out_sd, override, base_dtypes)
    finally:
        reader.close()

    print(
        f"all blocks baked; saving output to {output} "
        f"({time.time() - start:.1f}s elapsed)...",
        flush=True,
    )
    save_file(out_sd, str(output), metadata=metadata)
    print(f"wrote {output} ({time.time() - start:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_paths(cfg)
    bake(cfg)


if __name__ == "__main__":
    main()
