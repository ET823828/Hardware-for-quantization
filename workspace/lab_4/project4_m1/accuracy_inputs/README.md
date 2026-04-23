Representative proposal accuracy inputs live in this directory.

Expected default files:

- `llm_ffn_layer.npz`
- `vlm_vision_gemm.npz`
- `vla_action_head.npz`

Each snapshot should contain:

- `a`: 2D activation matrix shaped `[m_snapshot, k]`
- `w`: 2D weight matrix shaped `[n_snapshot, k]`

The proposal accuracy runner slices the same snapshot by phase:

- `decode`: uses the first `m` rows needed for decode
- `prefill`: uses the first `m` rows needed for prefill, capped by CLI limits

If a snapshot is missing, `run_sweeps.py run-accuracy` records a
`missing_inputs` row instead of using synthetic debug data.

## Fastest path

Use `dump_accuracy_snapshot.py` from `project4_m1/`:

```bash
python workspace/lab_4/project4_m1/dump_accuracy_snapshot.py from-hf-causal-lm \
  --model-id meta-llama/Llama-2-7b-hf \
  --module-path model.layers.0.mlp.up_proj \
  --prompt "Write a short summary of quantization-aware accelerator design." \
  --output workspace/lab_4/project4_m1/accuracy_inputs/llm_ffn_layer.npz \
  --max-rows 256 \
  --dtype float16
```

This path is ideal for the LLM snapshot.

For VLM/VLA models, if you already have model-specific code that extracts the
activation matrix `a` and weight matrix `w`, repack them with:

```bash
python workspace/lab_4/project4_m1/dump_accuracy_snapshot.py from-files \
  --a-path /path/to/activation.npy \
  --w-path /path/to/weight.npy \
  --output workspace/lab_4/project4_m1/accuracy_inputs/vlm_vision_gemm.npz
```

Supported input tensor formats for `from-files`:

- `.npy`
- `.npz`
- `.pt`
- `.pth`

## Recommended target shapes

- `llm_ffn_layer.npz`: `w.shape == (11008, 4096)`
- `vlm_vision_gemm.npz`: `w.shape == (3072, 3072)`
- `vla_action_head.npz`: `w.shape == (256, 4096)`

In all cases, `a.shape[1]` must equal `w.shape[1]`.
