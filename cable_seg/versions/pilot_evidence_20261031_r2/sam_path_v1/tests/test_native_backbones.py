"""Gradient and checkpoint contracts, without downloading model assets."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sam_path_v1.native_sam import (QVLoRA, CheckpointLoRABlock, FrozenViTBlock,
                                   DifferentiableViTMLP, NativeSam31LoRAModel)


class NativeBackboneContracts(unittest.TestCase):
    def test_zero_lora_exact_and_key_projection_frozen_after_update(self):
        torch.manual_seed(31)
        layer = nn.Linear(12, 36)
        adapter = QVLoRA(layer, rank=3, alpha=6)
        x = torch.randn(2, 7, 12)
        original = layer(x).detach()
        torch.testing.assert_close(adapter(x), original, rtol=0, atol=0)
        optimizer = torch.optim.SGD([p for p in adapter.parameters() if p.requires_grad], lr=.1)
        for _ in range(2):
            optimizer.zero_grad(); adapter(x).square().mean().backward(); optimizer.step()
        self.assertIsNone(layer.weight.grad)
        self.assertGreater(float(adapter.q_a.grad.abs().sum()), 0.)
        self.assertGreater(float(adapter.v_a.grad.abs().sum()), 0.)
        result = adapter(x)
        torch.testing.assert_close(result[..., 12:24], original[..., 12:24], rtol=0, atol=0)
        self.assertGreater(float((result[..., :12]-original[..., :12]).abs().sum()), 0.)

    def test_checkpoint_supports_lora_when_input_has_no_gradient(self):
        adapter = QVLoRA(nn.Linear(8, 24), rank=2, alpha=4)
        block = CheckpointLoRABlock(adapter).eval()
        x = torch.randn(2, 8)
        block(x).square().sum().backward()
        self.assertGreater(float(adapter.q_b.grad.abs().sum()), 0.)
        self.assertFalse(block.training)

    def test_frozen_block_disables_grad(self):
        linear = nn.Linear(8, 8).requires_grad_(False)
        x = torch.randn(2, 8, requires_grad=True)
        self.assertFalse(FrozenViTBlock(linear)(x).requires_grad)

    def test_unfused_mlp_preserves_standard_math_and_input_gradient(self):
        original = nn.Module()
        original.fc1, original.fc2 = nn.Linear(8, 16), nn.Linear(16, 8)
        original.act, original.norm = nn.GELU(), nn.Identity()
        original.drop1, original.drop2 = nn.Dropout(0.), nn.Dropout(0.)
        original.requires_grad_(False)
        mlp = DifferentiableViTMLP(original)
        x = torch.randn(3, 8, requires_grad=True)
        expected = original.fc2(original.act(original.fc1(x)))
        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)
        mlp(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in mlp.parameters()))

    def test_native_delta_is_only_lora_and_rejects_extra_or_wrong_shape(self):
        class FakeBase(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.native = QVLoRA(nn.Linear(4, 12), rank=2, alpha=4)
                self.receipt = {"backend": "test_double_only"}
        config = SimpleNamespace(num_queries=200, sam_repo="", sam_checkpoint="", canvas_size=224, init_seed=42)
        with patch("sam_path_v1.native_sam.NativeSam31Base", FakeBase):
            model = NativeSam31LoRAModel(config)
        delta = model.checkpoint_state_dict()
        self.assertEqual(len(delta), 4)
        self.assertFalse(any("base.weight" in key for key in delta))
        model.load_trainable_state_dict(delta)
        with self.assertRaisesRegex(ValueError, "keys"):
            model.load_trainable_state_dict({**delta, "unexpected": torch.zeros(1)})
        key = next(iter(delta))
        with self.assertRaisesRegex(ValueError, "shape"):
            model.load_trainable_state_dict({**delta, key: torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()
