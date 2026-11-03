"""SAM 2.1 Hiera-Tiny adapter using official frozen image features and decoder.

Only the image branch is retained. No video memory/predictor, resizing, automatic
mask generation or external prompts are used. Raw FPN maps remain 256-channel;
the decoder's pretrained high-resolution projections are applied during decode.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn

from .backbone import Sam31Backbone, file_sha256


class Sam21Backbone(Sam31Backbone):
    def __init__(self, repo, checkpoint, canvas_size=1024, encoder_amp="bf16"):
        nn.Module.__init__(self)
        if canvas_size < 64 or canvas_size % 32:
            raise ValueError("SAM2.1 canvas must be a positive multiple of 32, >=64")
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "sam2/modeling/backbones/hieradet.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Supply official SAM2 repository and sam2.1_hiera_tiny.pt")
        sys.path.insert(0, str(repo))
        from sam2.modeling.backbones.hieradet import Hiera
        from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
        from sam2.modeling.position_encoding import PositionEmbeddingSine
        from sam2.modeling.sam.prompt_encoder import PromptEncoder
        from sam2.modeling.sam.mask_decoder import MaskDecoder
        from sam2.modeling.sam.transformer import TwoWayTransformer
        import sam2.modeling.backbones.hieradet as hiera_module
        if Path(hiera_module.__file__).resolve() != repo / "sam2/modeling/backbones/hieradet.py":
            raise RuntimeError("Another sam2 installation is imported; use a fresh process")
        self.canvas_size, self.encoder_amp = int(canvas_size), encoder_amp
        self.encoder = ImageEncoder(
            trunk=Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
                        global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7)),
            neck=FpnNeck(PositionEmbeddingSine(256, normalize=True, temperature=10000),
                         d_model=256, backbone_channel_list=[768, 384, 192, 96],
                         fpn_top_down_levels=[2, 3], fpn_interp_model="nearest"), scalp=1)
        grid = canvas_size // 16
        self.prompt_encoder = PromptEncoder(256, (grid, grid), (canvas_size, canvas_size), 16)
        self.decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256,
            use_high_res_features=True, iou_prediction_use_sigmoid=True,
            pred_obj_scores=True, pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=False)
        self.no_mem = nn.Parameter(torch.zeros(1, 1, 256), requires_grad=False)
        self.register_buffer("pixel_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        source = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        source = source.get("model", source)
        prefixes = {"encoder.": "image_encoder.", "prompt_encoder.": "sam_prompt_encoder.",
                    "decoder.": "sam_mask_decoder."}
        selected = {}
        for name, expected in self.state_dict().items():
            key = "no_mem_embed" if name == "no_mem" else None
            for local, remote in prefixes.items():
                if name.startswith(local):
                    key = remote + name[len(local):]
                    break
            if key not in source or source[key].shape != expected.shape:
                raise RuntimeError(f"SAM2.1 pretrained tensor missing or wrong shape: {name} / {key}")
            selected[name] = source[key]
        self.load_state_dict(selected, strict=True)
        files = ("sam2/modeling/backbones/hieradet.py", "sam2/modeling/backbones/utils.py",
                 "sam2/modeling/backbones/image_encoder.py", "sam2/modeling/position_encoding.py",
                 "sam2/modeling/sam/prompt_encoder.py", "sam2/modeling/sam/mask_decoder.py",
                 "sam2/modeling/sam/transformer.py", "sam2/modeling/sam2_utils.py",
                 "sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
        self.receipt = dict(backend="sam21", repo=str(repo), checkpoint=str(checkpoint),
                            checkpoint_sha256=file_sha256(checkpoint), canvas_size=canvas_size,
                            source_sha256={name: file_sha256(repo/name) for name in files},
                            strict_retained_state=True, loaded_tensor_count=len(selected),
                            architecture="SAM2.1 Hiera-Tiny", native_rgb_resize=False,
                            normalization="ImageNet mean/std", raw_high_feature_channels=[256, 256],
                            no_video_memory=True, original_iou_sigmoid_inverted_for_common_loss=True)
        self.encoder.requires_grad_(False)
        self.prompt_encoder.requires_grad_(False)
        self.train(False)

    @property
    def encoder_module(self):
        return self.encoder

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.encoder.eval()
        self.prompt_encoder.eval()
        return self

    def encode(self, images):
        if images.shape[-2:] != (self.canvas_size, self.canvas_size):
            raise ValueError("SAM2 input must already be natively padded to the configured canvas")
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if images.device.type == "cuda" and self.encoder_amp == "bf16" else nullcontext())
        with torch.no_grad(), context:
            output = self.encoder(((images-self.pixel_mean)/self.pixel_std).contiguous())
            features = [x.float() for x in output["backbone_fpn"]]
        return {"low": features[-1]+self.no_mem.reshape(1, 256, 1, 1), "high": features[:2]}

    def decode(self, features, tokens, guidance):
        masks, quality, presence = super().decode(features, tokens, guidance)
        # Official SAM2 predicts sigmoid quality; the shared engine expects logits.
        return masks, torch.logit(quality.float().clamp(1e-6, 1-1e-6)), presence
