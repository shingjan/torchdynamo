import torch
import torchdynamo


@torchdynamo.optimize("inductor")
def fn(x, y):
    return torch.relu(x + y)


x = torch.randn(802816)
y = torch.randn(802816)

# TorchScript for comparison
def fn_eager(x, y):
    return torch.relu(x + y)


traced_cell = torch.jit.trace(fn_eager, (x, y))
# print("-------Torchscript printout-------")
# print(traced_cell.code)
# print("-------Torchscript printout-------")

# GPU inference on triton
c_triton = fn(x.cuda(), y.cuda())

# CPU inference on TIR
c_tvm = fn(x.cpu(), y.cpu())

# torch.allclose(c_triton.cpu(), c_tvm)
# print("correctness check passed!")
