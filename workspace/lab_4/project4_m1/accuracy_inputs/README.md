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
