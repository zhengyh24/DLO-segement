"""Empty native prompts must not register nondeterministic grid backward."""
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F

from sam_path_v1.native_sam import install_empty_point_shortcut, install_empty_box_shortcut


class EmptyPointTests(unittest.TestCase):
    def test_empty_forward_and_feature_coordinate_grad_equal_native(self):
        torch.manual_seed(42)
        image=torch.randn(2,4,5,6,requires_grad=True)
        points=torch.empty(0,2,2,requires_grad=True)
        mask=torch.zeros(2,0,dtype=torch.bool)
        def official(points,points_mask,points_labels,img_feats):
            grid=points.transpose(0,1).unsqueeze(2)*2-1
            return F.grid_sample(img_feats,grid,align_corners=False).squeeze(-1).permute(2,0,1),points_mask
        expected,_=official(points,mask,None,image)
        reference_grad=torch.autograd.grad(expected.sum(),(image,points))
        encoder=SimpleNamespace(d_model=4,_encode_points=official)
        install_empty_point_shortcut(encoder)
        actual,actual_mask=encoder._encode_points(points,mask,None,image)
        actual_grad=torch.autograd.grad(actual.sum(),(image,points))
        self.assertTrue(torch.equal(expected,actual))
        self.assertIs(actual_mask,mask)
        for a,b in zip(reference_grad,actual_grad): self.assertTrue(torch.equal(a,b))
        self.assertEqual(encoder.empty_point_shortcut_calls,1)

    def test_nonempty_dispatch_is_exact_official_call(self):
        sent=[]; marker=object()
        def official(*args): sent.append(args); return marker
        encoder=SimpleNamespace(d_model=4,_encode_points=official)
        install_empty_point_shortcut(encoder)
        args=(torch.ones(1,1,2),object(),object(),object())
        self.assertIs(encoder._encode_points(*args),marker)
        self.assertEqual(len(sent),1)
        self.assertEqual(encoder.empty_point_shortcut_calls,0)
        for a,b in zip(sent[0],args): self.assertIs(a,b)

    def test_empty_boxes_zero_gradient_and_nonempty_unchanged(self):
        sent=[];marker=object()
        def official(*args): sent.append(args);return marker
        encoder=SimpleNamespace(d_model=4,_encode_boxes=official)
        install_empty_box_shortcut(encoder)
        image=torch.randn(2,4,5,6,requires_grad=True)
        boxes=torch.empty(0,2,4,requires_grad=True);mask=torch.zeros(2,0,dtype=torch.bool)
        output,actual_mask=encoder._encode_boxes(boxes,mask,None,image)
        grads=torch.autograd.grad(output.sum(),(image,boxes))
        self.assertEqual(tuple(output.shape),(0,2,4));self.assertIs(actual_mask,mask)
        self.assertTrue(torch.equal(grads[0],torch.zeros_like(image)))
        self.assertTrue(torch.equal(grads[1],torch.zeros_like(boxes)))
        self.assertEqual(sent,[])
        self.assertIs(encoder._encode_boxes(torch.ones(1,2,4),mask,None,image),marker)
        self.assertEqual(len(sent),1)


if __name__ == "__main__": unittest.main()
