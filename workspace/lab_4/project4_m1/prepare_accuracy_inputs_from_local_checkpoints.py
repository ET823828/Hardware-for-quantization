from __future__ import annotations

from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
ACCURACY_DIR = ROOT / "accuracy_inputs"


def load_2d_tensor_slice(
    model_dir: Path,
    tensor_name: str,
    out_rows: int,
    out_cols: int,
) -> np.ndarray:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors found under {model_dir}")

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            if tensor_name not in reader.keys():
                continue
            tensor = reader.get_tensor(tensor_name)
            if tensor.ndim != 2:
                raise ValueError(f"Tensor {tensor_name} is not 2D: {tuple(tensor.shape)}")
            rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
            if rows < out_rows or cols < out_cols:
                raise ValueError(
                    f"Tensor {tensor_name} shape {tuple(tensor.shape)} is smaller than requested {(out_rows, out_cols)}"
                )
            sliced = tensor[:out_rows, :out_cols].detach().float().cpu().numpy().astype(np.float32, copy=False)
            return sliced

    raise KeyError(f"Tensor {tensor_name} not found in {model_dir}")


def sample_activations(m_rows: int, k_cols: int, distribution: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        a = rng.normal(0.0, 0.9, size=(m_rows, k_cols))
    elif distribution == "gaussian_narrow":
        a = rng.normal(0.0, 0.5, size=(m_rows, k_cols))
    elif distribution == "heavy_tail":
        base = rng.normal(0.0, 0.7, size=(m_rows, k_cols))
        tail = rng.normal(0.0, 0.2, size=(m_rows, k_cols))
        denom = np.abs(rng.normal(1.0, 0.35, size=(m_rows, k_cols)))
        denom = np.maximum(0.25, denom)
        a = (base + tail) / denom
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return a.astype(np.float32, copy=False)


def write_snapshot(path: Path, a: np.ndarray, w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if a.ndim != 2 or w.ndim != 2:
        raise ValueError("Both activation and weight must be 2D")
    if a.shape[1] != w.shape[1]:
        raise ValueError(f"K mismatch: a={a.shape}, w={w.shape}")
    np.savez(path, a=a, w=w)
    print(f"Wrote {path.name}: a{a.shape}, w{w.shape}")


def main() -> None:
    qwen_dir = ACCURACY_DIR / "Qwen2-VL-2B"
    openvla_dir = ACCURACY_DIR / "openvla-7b"

    # LLM workload target: N=11008, K=4096
    llm_w = load_2d_tensor_slice(
        openvla_dir,
        tensor_name="language_model.model.layers.0.mlp.gate_proj.weight",
        out_rows=11008,
        out_cols=4096,
    )
    llm_a = sample_activations(m_rows=256, k_cols=4096, distribution="gaussian", seed=11)
    write_snapshot(ACCURACY_DIR / "llm_ffn_layer.npz", llm_a, llm_w)

    # VLM workload target: N=3072, K=3072
    vlm_w = load_2d_tensor_slice(
        qwen_dir,
        tensor_name="visual.merger.mlp.0.weight",
        out_rows=3072,
        out_cols=3072,
    )
    vlm_a = sample_activations(m_rows=256, k_cols=3072, distribution="gaussian_narrow", seed=23)
    write_snapshot(ACCURACY_DIR / "vlm_vision_gemm.npz", vlm_a, vlm_w)

    # VLA workload target: N=256, K=4096
    vla_w = load_2d_tensor_slice(
        openvla_dir,
        tensor_name="language_model.model.layers.0.mlp.gate_proj.weight",
        out_rows=256,
        out_cols=4096,
    )
    vla_a = sample_activations(m_rows=256, k_cols=4096, distribution="heavy_tail", seed=37)
    write_snapshot(ACCURACY_DIR / "vla_action_head.npz", vla_a, vla_w)


if __name__ == "__main__":
    main()
