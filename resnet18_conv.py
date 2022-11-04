import torch
import torchdynamo
from torchdynamo.testing import rand_strided

aten = torch.ops.aten


def fn(inp0, inp1):
    return aten.convolution(inp0, inp1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)


optimize_ctx = torchdynamo.optimize("inductor")
optimize_fn = optimize_ctx(fn)

inp0 = rand_strided(
    (8, 3, 224, 224), (150528, 50176, 224, 1), device="cpu", dtype=torch.float32
)
inp1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device="cpu", dtype=torch.float32)

# GPU inference on triton
c_out = fn(inp0, inp1)

# CPU inference on TIR
c_tvm = optimize_fn(inp0.cpu(), inp1.cpu())

torch.testing.assert_close(c_out, c_tvm.cpu(), rtol=1e-4, atol=1e-4)
print("correctness check passed!")
