from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable


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


def move_batch_to_device(batch: Any, device: str, tensor_dtype: Any | None = None) -> Any:
    import torch

    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except TypeError:
            pass

    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                kwargs = {"device": device}
                if tensor_dtype is not None and torch.is_floating_point(value):
                    kwargs["dtype"] = tensor_dtype
                moved[key] = value.to(**kwargs)
            else:
                moved[key] = value
        return moved

    raise TypeError(f"Unsupported batch type for device transfer: {type(batch)!r}")


def flatten_activation(tensor: Any, max_rows: int) -> Any:
    import torch

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


def iter_named_weight_modules(model: Any) -> Iterable[tuple[str, tuple[int, int]]]:
    import torch

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            yield name, tuple(weight.shape)


def build_qwen2_vl_conversation(image_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(Path(image_path).expanduser().resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_openvla_prompt(instruction: str) -> str:
    cleaned = instruction.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return f"In: What action should the robot take to {cleaned}?\nOut:"


def capture_module_input(target: Any, run_forward: Any) -> Any:
    captured: dict[str, Any] = {}

    def pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("Target module received no positional inputs")
        captured["activation"] = inputs[0]

    handle = target.register_forward_pre_hook(pre_hook)
    try:
        run_forward()
    finally:
        handle.remove()

    if "activation" not in captured:
        raise RuntimeError("Failed to capture activation for the target module")
    return captured["activation"]


def print_matching_modules(
    model: Any,
    contains: str | None = None,
    rows: int | None = None,
    cols: int | None = None,
    limit: int | None = None,
) -> None:
    matched = 0
    for name, shape in iter_named_weight_modules(model):
        if contains and contains not in name:
            continue
        if rows is not None and shape[0] != rows:
            continue
        if cols is not None and shape[1] != cols:
            continue
        print(f"{name}\t{shape[0]}x{shape[1]}")
        matched += 1
        if limit is not None and matched >= limit:
            break
    if matched == 0:
        print("No matching 2D weight modules found.")


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

    def run_forward() -> None:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)

    activation = capture_module_input(target, run_forward)
    a = flatten_activation(activation, args.max_rows)
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


def load_qwen2_vl_model(args: argparse.Namespace) -> tuple[Any, Any, str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    processor_kwargs: dict[str, Any] = {}
    if getattr(args, "min_pixels", None) is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if getattr(args, "max_pixels", None) is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, processor, device, torch_dtype


def command_from_qwen2_vl(args: argparse.Namespace) -> None:
    import torch

    model, processor, device, _torch_dtype = load_qwen2_vl_model(args)
    target = resolve_module(model, args.module_path)
    if not hasattr(target, "weight"):
        raise ValueError(f"Target module {args.module_path} does not expose a .weight tensor")

    conversation = build_qwen2_vl_conversation(args.image_path, args.prompt)

    def run_forward() -> None:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_device(inputs, device)
        with torch.no_grad():
            model(**inputs)

    activation = capture_module_input(target, run_forward)
    a = flatten_activation(activation, args.max_rows)
    w = tensor_to_numpy(target.weight)
    save_snapshot(Path(args.output), a, w)

    print(f"Saved snapshot to {args.output}")
    print(f"  model: {args.model_path}")
    print(f"  module: {args.module_path}")
    print(f"  activation shape: {a.shape}")
    print(f"  weight shape: {w.shape}")


def resolve_openvla_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "instruction", None):
        return build_openvla_prompt(args.instruction)
    raise ValueError("Provide either --prompt or --instruction for OpenVLA snapshot dumping.")


def load_openvla_model(args: argparse.Namespace) -> tuple[Any, Any, str, Any]:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, processor, device, torch_dtype


def command_from_openvla(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    model, processor, device, torch_dtype = load_openvla_model(args)
    target = resolve_module(model, args.module_path)
    if not hasattr(target, "weight"):
        raise ValueError(f"Target module {args.module_path} does not expose a .weight tensor")

    prompt = resolve_openvla_prompt(args)
    image = Image.open(args.image_path).convert("RGB")

    def run_forward() -> None:
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = move_batch_to_device(inputs, device, tensor_dtype=torch_dtype)
        with torch.no_grad():
            if args.invoke == "predict_action":
                if not hasattr(model, "predict_action"):
                    raise ValueError(
                        "Loaded OpenVLA checkpoint does not expose predict_action; try --invoke forward instead."
                    )
                model.predict_action(
                    **inputs,
                    unnorm_key=args.unnorm_key,
                    do_sample=False,
                )
            else:
                model(**inputs)

    activation = capture_module_input(target, run_forward)
    a = flatten_activation(activation, args.max_rows)
    w = tensor_to_numpy(target.weight)
    save_snapshot(Path(args.output), a, w)

    print(f"Saved snapshot to {args.output}")
    print(f"  model: {args.model_path}")
    print(f"  module: {args.module_path}")
    print(f"  invoke: {args.invoke}")
    print(f"  activation shape: {a.shape}")
    print(f"  weight shape: {w.shape}")


def load_model_for_inspection(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

    torch_dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)

    if args.model_type == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_type == "qwen2-vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_type == "openvla":
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.to(device)
    model.eval()
    return model


def command_inspect_modules(args: argparse.Namespace) -> None:
    model = load_model_for_inspection(args)
    print_matching_modules(
        model,
        contains=args.contains,
        rows=args.rows,
        cols=args.cols,
        limit=args.limit,
    )


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

    inspect = subparsers.add_parser(
        "inspect-modules",
        help="Load a local checkpoint and print 2D weight modules so you can choose a module path.",
    )
    inspect.add_argument("--model-type", required=True, choices=["causal-lm", "qwen2-vl", "openvla"])
    inspect.add_argument("--model-path", required=True, help="Local checkpoint directory or Hugging Face id")
    inspect.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    inspect.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    inspect.add_argument("--contains", help="Only print modules whose path contains this substring")
    inspect.add_argument("--rows", type=int, help="Only print modules with this output dimension")
    inspect.add_argument("--cols", type=int, help="Only print modules with this input dimension")
    inspect.add_argument("--limit", type=int, help="Stop after printing this many matches")
    inspect.add_argument("--trust-remote-code", action="store_true")
    inspect.set_defaults(func=command_inspect_modules)

    qwen2_vl = subparsers.add_parser(
        "from-qwen2-vl",
        help="Load a local Qwen2-VL checkpoint, hook one module, and save a snapshot from an image+text example.",
    )
    qwen2_vl.add_argument("--model-path", required=True, help="Local Qwen2-VL checkpoint directory")
    qwen2_vl.add_argument("--module-path", required=True, help="Python attribute path to the target module")
    qwen2_vl.add_argument("--image-path", required=True, help="Representative image path")
    qwen2_vl.add_argument("--prompt", required=True, help="Representative user prompt for the image")
    qwen2_vl.add_argument("--output", required=True, help="Output .npz path")
    qwen2_vl.add_argument("--max-rows", type=int, default=256, help="Keep at most this many activation rows")
    qwen2_vl.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    qwen2_vl.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    qwen2_vl.add_argument("--min-pixels", type=int, help="Optional Qwen2-VL processor min_pixels override")
    qwen2_vl.add_argument("--max-pixels", type=int, help="Optional Qwen2-VL processor max_pixels override")
    qwen2_vl.add_argument("--trust-remote-code", action="store_true")
    qwen2_vl.set_defaults(func=command_from_qwen2_vl)

    openvla = subparsers.add_parser(
        "from-openvla",
        help="Load a local OpenVLA checkpoint, hook one module, and save a snapshot from an image+instruction example.",
    )
    openvla.add_argument("--model-path", required=True, help="Local OpenVLA checkpoint directory")
    openvla.add_argument("--module-path", required=True, help="Python attribute path to the target module")
    openvla.add_argument("--image-path", required=True, help="Representative robot scene image path")
    prompt_group = openvla.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Full OpenVLA prompt, e.g. 'In: What action should the robot take to pick up the red block?\\nOut:'")
    prompt_group.add_argument("--instruction", help="Task instruction to wrap in the standard OpenVLA prompt template")
    openvla.add_argument("--output", required=True, help="Output .npz path")
    openvla.add_argument("--max-rows", type=int, default=256, help="Keep at most this many activation rows")
    openvla.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    openvla.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    openvla.add_argument("--invoke", choices=["forward", "predict_action"], default="forward")
    openvla.add_argument("--unnorm-key", default="bridge_orig", help="Normalization key for --invoke predict_action")
    openvla.set_defaults(func=command_from_openvla)

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
