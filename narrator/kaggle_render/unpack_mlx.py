"""Turn a trained voice model as the Mac keeps it (MLX: Orpheus 4-bit, Qwen 6-bit) back into an ordinary 16-bit
model that PyTorch loads, on a cloud machine. Used by Create in the cloud and Connections → Modal → Upload: the Mac
copy is sent (instead of the full model) and unpacked here, so the cloud voice is exactly the one the Mac plays.

MLX "affine" quantization: each row of a weight is cut into groups of `group_size` numbers; each number is a
`bits`-bit integer q, packed into the row's uint32 words as one little-end-first stream of bits (a 6-bit number
can straddle two words), and stands for scale * q + bias of its group.

Usage:  python unpack_mlx.py SRC_DIR DEST_DIR
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SHARD_BYTES = 2_000_000_000


def unpack(packed, scales, biases, bits=4, group_size=64):
    """The float32 weight from MLX's packed uint32 words (rows x words), scales and biases (rows x groups)."""
    packed = np.ascontiguousarray(packed, dtype="<u4")
    rows = packed.shape[0]
    stream = np.unpackbits(packed.view(np.uint8).reshape(rows, -1), axis=1, bitorder="little")
    cols = stream.shape[1] // bits
    q = stream[:, :cols * bits].reshape(rows, cols, bits).astype(np.uint32)
    q = (q << np.arange(bits, dtype=np.uint32)).sum(-1, dtype=np.uint32).astype(np.float32)
    q = q.reshape(rows, cols // group_size, group_size)
    w = q * scales[..., None].astype(np.float32) + biases[..., None].astype(np.float32)
    return w.reshape(rows, cols)


def main(src, dest):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, dest = Path(src), Path(dest)
    cfg = json.loads((src / "config.json").read_text())
    quant = cfg.pop("quantization", None) or cfg.pop("quantization_config", None) or {}
    cfg.pop("quantization_config", None)
    bits, group = int(quant.get("bits", 4)), int(quant.get("group_size", 64))
    cfg["torch_dtype"] = "bfloat16"
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files = sorted(src.glob("*.safetensors"))
    tensors = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                tensors[k] = (f, k)
    names = sorted({k.rsplit(".", 1)[0] for k in tensors if k.endswith(".scales")})
    shards, shard, size, weight_map = [], {}, 0, {}

    def flush():
        nonlocal shard, size
        if shard:
            shards.append(shard)
        shard, size = {}, 0

    def get(k):
        f, key = tensors[k]
        with safe_open(str(f), "pt") as h:
            return h.get_tensor(key)

    done = set()
    for base in names:  # the packed weights
        w = unpack(get(base + ".weight").numpy(), get(base + ".scales").float().numpy(),
                   get(base + ".biases").float().numpy(), bits, group)
        t = torch.from_numpy(w).to(torch.bfloat16)
        shard[base + ".weight"] = t
        size += t.numel() * 2
        done |= {base + ".weight", base + ".scales", base + ".biases"}
        print(f"  unpacked {base}: {tuple(t.shape)}", flush=True)
        if size > SHARD_BYTES:
            flush()
    for k in sorted(tensors):  # everything else as it is (norms)
        if k in done:
            continue
        t = get(k).to(torch.bfloat16)
        shard[k] = t
        size += t.numel() * 2
        if size > SHARD_BYTES:
            flush()
    flush()
    total = len(shards)
    for i, s in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file({k: v.contiguous() for k, v in s.items()}, str(tmp / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in s})
    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(v.numel() * 2 for s in shards for v in s.values())}, "weight_map": weight_map}))
    (tmp / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in src.iterdir():
        if f.suffix in (".json", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(f, tmp / f.name)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"DONE: {len(weight_map)} tensors in {total} file(s) → {dest}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
