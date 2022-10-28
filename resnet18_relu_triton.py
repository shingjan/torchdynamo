import triton
import triton.language as tl
from torchinductor.triton_ops.autotune import pointwise_heuristics

import torch, triton
from triton import language as tl

# fmt: off
@triton.jit
def kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    print(x3)
    print(x1)
    tmp0 = tl.load(in_ptr0 + x3, xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + x1, xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + x1, xmask).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + x1, xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr4 + x1, xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sqrt(tmp8)
    tmp10 = 1 / tmp9
    tmp11 = 1
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.maximum(0, tmp20)
    tl.store(out_ptr0 + x3 + tl.zeros([XBLOCK], tl.int32), tmp21, xmask)

# fmt: on
import torch

in_ptr0 = torch.randn(128, device="cuda")
in_ptr1 = torch.randn(128, device="cuda")
in_ptr2 = torch.empty(128, device="cuda")
in_ptr3 = torch.randn(128, device="cuda")
in_ptr4 = torch.randn(128, device="cuda")
out_ptr_0 = torch.randn(128, device="cuda")

kernel[(1,)](in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr_0, 128, 128)
