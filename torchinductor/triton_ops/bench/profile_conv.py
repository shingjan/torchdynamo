import torch
import torchinductor.triton_ops

from torch.profiler import profile, record_function, ProfilerActivity

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, stride, padding, dilation, groups, dtype = \
    32, 56, 56, 64, 3, 3, 64, (1, 1), (0, 0), (1, 1), 1, torch.float32

def profile_op(
        # provider
        provider,
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=1,
        dtype=torch.float16, layout="nhwc",
        warmup=25, rep=50):


    # allocate inputs, nchw
    x = torch.randn((BATCH, IN_C, IN_H, IN_W), dtype=dtype, device='cuda')
    w = torch.randn((KERNEL_N, IN_C // groups, KERNEL_H, KERNEL_W),
                    dtype=dtype, device='cuda')
    bias = torch.randn((KERNEL_N), dtype=dtype, device='cuda')
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    OUT_H = (IN_H + 2 * padding[0] - dilation[0] * (KERNEL_H - 1) - 1 + stride[0]) // stride[0]
    OUT_W = (IN_W + 2 * padding[1] - dilation[1] * (KERNEL_W - 1) - 1 + stride[1]) // stride[1]

    if provider == "cublas":
        fn = lambda: torch.conv2d(x, w, bias, stride, padding, dilation, groups)
    if provider == "triton":
        fn = lambda: torchinductor.triton_ops.conv(
            x, w, bias, stride, padding, dilation, False, (0, 0), groups
        )
    else:
        ValueError(f"{provider} not supported")
    # warm up
    for _ in range(warmup):
        fn()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(rep):
                fn()

    print("Profiling ", provider)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


for provider in ["cublas", "triton"]:
    profile_op(
        # provider
        provider,
        # Tensor dimensions
        BATCH, IN_C, IN_H, IN_W,
        KERNEL_N, KERNEL_H, KERNEL_W,
        # parameters of conv
        stride, padding,
        dilation, groups,
        dtype=dtype, layout="nhwc",
        warmup=25, rep=50
    )