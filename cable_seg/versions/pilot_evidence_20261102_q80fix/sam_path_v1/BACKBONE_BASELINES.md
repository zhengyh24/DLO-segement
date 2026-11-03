# Backbone baselines (2026-10-31)

## Native SAM3.1 LoRA arm S

`native_sam.py` retains the pretrained SAM3.1 multiplex detector's native 200
instance queries, text encoder, visual-language fusion, detection decoder,
joint presence score and instance mask head. Text is always `cable`; there are
no GT prompts, learned automatic-condition queries or path modules from our
method. Only Q/V LoRA in ViT blocks 24–31 is trained: rank 8, alpha 16,
dropout 0, 262,144 parameters. Initial B matrices are zero. The first backward
therefore gives zero A gradients and nonzero B gradients; subsequent optimizer
steps can train both. All native head and base weights remain frozen.

The wrapper preserves the common tensor/checkpoint interface and sets
`aux.native_detection=True`. Its object logits already include the native
presence probability (the official multiplex detector uses
`supervise_joint_box_scores=True`); do not multiply presence twice. Train with
the common set matching, mask BCE/Dice and object losses only. Quality, path,
descriptor and geometry losses are inactive. Common exclusive-pixel decoding
is an evaluation protocol adapter, not the original SAM3 video tracker/NMS.

The retained detector is built with the exact three SAM3 FPN scales from the
multiplex constructor. Unused interactive/propagation/video branches are not
loaded. Its checkpoint omits `language_backbone.encoder.text_projection`, a
pooled CLIP projection whose output VEText discards; that unused parameter is
removed explicitly. Every retained learned tensor must be present with exact
shape; missing tensors never get random fallback weights. RoPE geometry is
regenerated for the configured native padded canvas (1036 for 1024 RGB).

Official ViT MLP code uses an inference-only fused linear/GELU that raises when
autograd is enabled. The first 24 frozen blocks run under no_grad. Only the
last eight MLPs use equivalent standard PyTorch linear/GELU operations so the
input gradient can reach LoRA; their pretrained weights are unchanged. BF16
rounding can differ, so this is not an exact numerical-parity claim. The native
modules remain in eval mode to avoid DAC query doubling, stochastic depth,
dropout and native GT-dependent matching; eval mode does not disable autograd.
Only the last eight blocks use activation checkpointing. The parent model's
training flag still follows the engine's train/eval lifecycle.

Real local 1036 smoke: output `[1,200,296,296]`, one visual encoder and one native
detection-decoder entry, finite gradients for all 32 LoRA matrices. Peak CUDA
allocation was 5,154,798,080 bytes. This uses random RGB and a diagnostic scalar
objective, not the full criterion; a full-criterion admission check is still
required on the training GPU. The retained model has 839,395,126 parameters,
262,144 trainable. Do not compare it to C as a parameter-matched ablation.

## SAM2.1 Hiera-Tiny

`backbone_sam2.py` manually constructs the official Hiera-Tiny image encoder,
FPN, prompt encoder and original mask decoder from the released configuration.
All 311 retained state tensors are strictly loaded. The image encoder and
prompt encoder are frozen; the original decoder remains trainable. Video
memory modules and predictors are unused. Feature interface:

- `encode(images)` returns `low [B,256,H/16,W/16]`, `high` raw 256-channel maps
  at stride 4 and 8, with the official no-memory embedding added to `low`.
- `decode(features,tokens,guidance)` applies official decoder high-resolution
  projections, adds its no-mask embedding and invokes that decoder once.
- The official decoder's quality head already includes sigmoid, so it is
  converted back to a clipped logit for the shared engine's quality loss.
- The input canvas must be divisible by 32. Use 1024 for native synthetic
  1024 RGB; no resize is applied. ImageNet normalization is retained. Different
  SAM stride/canvas costs must be reported when comparing the backbones.

The actual local 1024 GPU smoke passed encode/decode/backward: low64×64,
high256×256/128×128; prompt token and dense guidance gradients nonzero, encoder
gradients absent. The smoke used two queries (capacity testing is separate).

Official assets:

- Repository: https://github.com/facebookresearch/sam2
- Frozen commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Tiny checkpoint: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
- SHA256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`
- Local root: `assets/pilot_evidence_20261031/` (repository `sam2/`, weight
  `sam2.1_hiera_tiny.pt`). Existing SAM3 source/base weights were not copied.

No installation or modification of the shared Python environment was needed.
Runtime uses existing torch 2.7.1/cu126 and hydra-core/iopath. SAM2 is imported
directly from the frozen repository. Its optional CUDA connected-components
extension is not used by this tensor interface and need not be compiled.

## Reproduce software checks

From the version parent (the directory containing this `sam_path_v1` package):

```bash
python -m unittest sam_path_v1.tests.test_native_backbones -v
python -m sam_path_v1.backbone_smoke --backend sam31_native --repo SAM3_REPO --checkpoint SAM31_BASE --canvas 1036 --device cuda --output NEW_NATIVE_RECEIPT.json
python -m sam_path_v1.backbone_smoke --backend sam21 --repo SAM2_REPO --checkpoint SAM21_TINY --canvas 1024 --device cuda --output NEW_SAM2_RECEIPT.json
```

Smoke receipts are exclusive-create. Five CPU contracts cover zero-LoRA
identity and unchanged K, two-step Q/V learning, checkpointing with no-grad
input, frozen upstream blocks, unfused MLP input gradients, and strict delta
checkpoint loading. These are software checks, not model accuracy evidence.
Local real receipts: `outputs/pilot_evidence_20261031/backbone_checks/`.

For server admission, transfer/clone the exact SAM2 commit and checkpoint to a
new assets directory, verify its SHA above, and point `sam2_repo` /
`sam2_checkpoint` at those files. Never change shared historical source or base
weights. Run the smokes and the engine's complete two-update criterion check
before any official run. Keep these new source files in engine provenance.
