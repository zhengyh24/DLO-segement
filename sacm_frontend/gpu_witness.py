"""Read-only seeded CUDA/CPU kernel witness. No training/downloads."""
import json
import torch
torch.manual_seed(20261014)
torch.backends.cuda.matmul.allow_tf32=False
x=torch.randn(64,64);expected=x@x
actual=(x.cuda()@x.cuda()).cpu()
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
print('WITNESS',json.dumps(dict(shape=list(actual.shape),device=torch.cuda.get_device_name(),
    torch=str(torch.__version__),cuda=torch.version.cuda,finite=bool(actual.isfinite().all()),
    max_abs_error=float((actual-expected).abs().max()),training_performed=False)))
