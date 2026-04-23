from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def resolve_dtype(dtype_name: str) -> Any:
    import torch

    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        raise KeyError(f"Unsupported dtype: {dtype_name}")
    return table[dtype_name]


def resolve_device(device_name: str) -> str:
    import torch

    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def flatten_activation(tensor: Any, max_rows: int) -> Any:
    import torch
    import numpy as np

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor activation.")
    if tensor.ndim < 2:
        raise ValueError(f"Expected activation with ndim >= 2, got shape {tuple(tensor.shape)}")
    flat = tensor.reshape(-1, tensor.shape[-1]).detach().float().cpu().numpy()
    if max_rows > 0:
        flat = flat[:max_rows]
    if flat.ndim != 2 or flat.shape[0] == 0:
        raise ValueError(f"Activation flattening produced invalid shape {flat.shape}")
    return flat


def tensor_to_numpy(tensor: Any) -> Any:
    import torch
    import numpy as np

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor weight.")
    array = tensor.detach().float().cpu().numpy()
    if array.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {array.shape}")
    return array


def resolve_module(model: Any, module_path: str) -> Any:
    target = model
    for part in module_path.split("."):
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)
    return target


def save_snapshot(path: Path, a: Any, w: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D activation matrix, got {a.shape}")
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got {w.shape}")
    if a.shape[1] != w.shape[1]:
        raise ValueError(
            f"Activation K dimension {a.shape[1]} does not match weight K dimension {w.shape[1]}"
        )
    np.savez(path, a=a.astype(np.float32, copy=False), w=w.astype(np.float32, copy=False))


def command_from_hf_causal_lm(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    target = resolve_module(model, args.module_path)
    if not hasattr(target, "weight"):
        raise ValueError(f"Target module {args.module_path} does not expose a .weight tensor")

    captured: dict[str, Any] = {}

    def pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("Target module received no positional inputs")
        captured["activation"] = inputs[0]

    handle = target.register_forward_pre_hook(pre_hook)
    try:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError("Failed to capture activation for the target module")

    a = flatten_activation(captured["activation"], args.max_rows)
    w = tensor_to_numpy(target.weight)
    save_snapshot(Path(args.output), a, w)

    print(f"Saved snapshot to {args.output}")
    print(f"  module: {args.module_path}")
    print(f"  activation shape: {a.shape}")
    print(f"  weight shape: {w.shape}")


def load_array(path: Path, key: str | None) -> Any:
    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            selected_key = key or sorted(payload.files)[0]
            array = payload[selected_key]
    elif suffix in {".pt", ".pth"}:
        import torch

        payload = torch.load(path, map_location="cpu")
        if key is not None:
            payload = payload[key]
        if isinstance(payload, dict):
            raise ValueError("Torch payload is still a dict; pass --a-key/--w-key to select a tensor.")
        array = payload.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor file format: {path}")
    return np.asarray(array, dtype=np.float32)


def command_from_files(args: argparse.Namespace) -> None:
    a = load_array(Path(args.a_path), args.a_key)
    w = load_array(Path(args.w_path), args.w_key)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    save_snapshot(Path(args.output), a, w)
    print(f"Saved snapshot to {args.output}")
    print(f"  activation shape: {a.shape}")
    print(f"  weight shape: {w.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create proposal accuracy snapshots (.npz with keys a and w)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from_hf = subparsers.add_parser(
        "from-hf-causal-lm",
        help="Load a text-only causal LM from Hugging Face, hook one linear module, and save a snapshot.",
    )
    from_hf.add_argument("--model-id", required=True, help="Hugging Face model id, e.g. meta-llama/Llama-2-7b-hf")
    from_hf.add_argument("--module-path", required=True, help="Python attribute path to the target module")
    from_hf.add_argument("--output", required=True, help="Output .npz path")
    from_hf.add_argument("--prompt", required=True, help="Representative text prompt")
    from_hf.add_argument("--max-rows", type=int, default=256, help="Keep at most this many activation rows")
    from_hf.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    from_hf.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    from_hf.add_argument("--cache-dir", help="Optional Hugging Face cache dir")
    from_hf.add_argument("--trust-remote-code", action="store_true")
    from_hf.set_defaults(func=command_from_hf_causal_lm)

    from_files = subparsers.add_parser(
        "from-files",
        help="Pack existing activation/weight tensors into the required .npz format.",
    )
    from_files.add_argument("--a-path", required=True, help="Path to activation tensor (.npy/.npz/.pt/.pth)")
    from_files.add_argument("--w-path", required=True, help="Path to weight tensor (.npy/.npz/.pt/.pth)")
    from_files.add_argument("--output", required=True, help="Output .npz path")
    from_files.add_argument("--a-key", help="Optional key for .npz/.pt activation payloads")
    from_files.add_argument("--w-key", help="Optional key for .npz/.pt weight payloads")
    from_files.set_defaults(func=command_from_files)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
