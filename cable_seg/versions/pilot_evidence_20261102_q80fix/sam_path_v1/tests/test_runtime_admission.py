"""Admission wiring only; fixtures are not SAM accuracy/runtime evidence."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sam_path_v1.runtime_admission import runtime_admission, verify_loaded_state


class ReloadedFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1))
        self.config=SimpleNamespace(variant="A",backend="toy",canvas_size=16,num_queries=2,encoder_amp="none")
        self.base=torch.nn.Module(); self.base.encoder_module=torch.nn.Identity(); self.base.decoder=torch.nn.Identity()
        self.condition=torch.nn.Identity(); self.last_calls={}; self.forward_count=0
    def checkpoint_state_dict(self): return dict(weight=self.weight.detach().cpu().clone())
    def forward(self,rgb,valid):
        self.forward_count+=1
        rgb=self.base.decoder(self.condition(self.base.encoder_module(rgb)))
        self.last_calls=dict(encoder=1,decoder=1)
        return dict(mask_logits=rgb[:,:1].repeat(1,2,1,1),object_logits=torch.zeros(1,2),
                    quality_logits=torch.zeros(1,2),calls=self.last_calls)


class RuntimeAdmissionTests(unittest.TestCase):
    def test_fresh_reload_train_rgb_only_two_forwards_and_real_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); image=root/"train.png"
            Image.fromarray(np.zeros((8,10,3),np.uint8)).save(image)
            manifest=root/"manifest.json"; manifest.write_text("{}")
            checkpoint=root/"last.pt"; checkpoint.write_bytes(b"mock checkpoint")
            model=ReloadedFixture()
            payload=dict(model_config=vars(model.config),model_state=model.checkpoint_state_dict(),train_config=dict(cpu_threads=1))
            records=[dict(id="test_forbidden",split="test",image="nonexistent_test.png"),
                     dict(id="first_train",split="train",image="train.png",masks="forbidden_GT.npz")]
            with patch("sam_path_v1.runtime_admission.build_from_checkpoint",return_value=(model,payload)) as build, \
                 patch("sam_path_v1.runtime_admission.read_manifest",return_value=dict(records=records)):
                result=runtime_admission(checkpoint,manifest,root,device="cpu")
            build.assert_called_once_with(checkpoint,device="cpu")
            self.assertTrue(result["complete"])
            self.assertEqual(result["image_id"],"first_train")
            self.assertEqual(model.forward_count,2)
            self.assertEqual(set(result["modules_seconds"]),{"encoder","condition","decoder"})
            self.assertEqual(result["output_shapes"]["mask_logits"],[1,2,16,16])
            self.assertFalse(result["gt_loaded"]); self.assertFalse(result["gt_scored"])

    def test_bad_reloaded_state_and_untrained_native_lora_rejected(self):
        model=ReloadedFixture()
        with self.assertRaisesRegex(ValueError,"mismatch/nonfinite"):
            verify_loaded_state(model,dict(model_state=dict(weight=torch.zeros(1)),model_config=dict(variant="A")))
        keys={f"block{i}.{projection}_b":torch.ones(4,2) for i in range(8) for projection in ("q","v")}
        native=SimpleNamespace(checkpoint_state_dict=lambda: keys)
        payload=dict(model_state=keys,model_config=dict(variant="S"))
        self.assertEqual(len(verify_loaded_state(native,payload)["native_lora_b_nonzero_elements"]),16)
        keys["block0.q_b"].zero_()
        with self.assertRaisesRegex(ValueError,"nonzero trained"):
            verify_loaded_state(native,payload)


if __name__ == "__main__": unittest.main()
