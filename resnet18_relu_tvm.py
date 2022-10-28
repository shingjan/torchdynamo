import tvm
from tvm.script import tir as T
from ctypes import c_void_p
from tvm.target import Target
from tvm import meta_schedule as ms
import tempfile

import torch

try:
    import torch._dynamo as torchdynamo
except ImportError:
    import torchdynamo
import torchinductor
from torchinductor.codecache import CppCodeCache
from torchinductor.utils import print_performance


# fmt: off
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        in_ptr0: T.Buffer[(802816,), "float32"],
        in_ptr1: T.Buffer[(802816,), "float32"],
        out_ptr0: T.Buffer[(802816,), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0 in T.serial(802816):
            with T.block("T_relu"):
                out_ptr0[i0] = T.max(in_ptr0[i0] + in_ptr1[i0], T.float32(0))

@tvm.script.ir_module
class Module_manual_rewrite:
    @T.prim_func
    def main(in_ptr0: T.Buffer[(8, 16, 28, 28, 8), "float32"], in_ptr1: T.Buffer[(8, 16, 28, 28, 8), "float32"], out_ptr0: T.Buffer[(8, 16, 28, 28, 8), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_i1_i2_fused in T.parallel(3584):
            for i3 in T.serial(28):
                for i4_fused in T.vectorized(8):
                    with T.block("T_relu"):
                        ax0 = T.axis.spatial(8, i0_i1_i2_fused // 448)
                        ax1 = T.axis.spatial(16, i0_i1_i2_fused % 448 // 28)
                        ax2 = T.axis.spatial(28, i0_i1_i2_fused % 28)
                        ax3, ax4 = T.axis.remap("SS", [i3, i4_fused])
                        T.reads(in_ptr0[ax0, ax1, ax2, ax3, ax4], in_ptr1[ax0, ax1, ax2, ax3, ax4])
                        T.writes(out_ptr0[ax0, ax1, ax2, ax3, ax4])
                        out_ptr0[ax0, ax1, ax2, ax3, ax4] = T.max(in_ptr0[ax0, ax1, ax2, ax3, ax4] + in_ptr1[ax0, ax1, ax2, ax3, ax4], T.float32(0))

@tvm.script.ir_module
class Module_tuned:
    @T.prim_func
    def main(in_ptr0: T.Buffer[802816, "float32"], in_ptr1: T.Buffer[802816, "float32"], out_ptr0: T.Buffer[802816, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_fused in T.parallel(802816):
            with T.block("T_relu"):
                ax0 = T.axis.spatial(802816, i0_fused)
                T.reads(in_ptr0[ax0], in_ptr1[ax0])
                T.writes(out_ptr0[ax0])
                out_ptr0[ax0] = T.max(in_ptr0[ax0] + in_ptr1[ax0], T.float32(0))


cpp_str = '''
#include "/tmp/torchinductor_yj/qr/cqrr7t6pdy7hpf76525uh7ddg65eljkjlgxa5dhk7amq6xb6a3ia.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel
    {
        #pragma omp for
        for(long i0=0; i0<802816; ++i0)
        {
            {
                {
                    auto tmp0 = in_ptr0[i0];
                    auto tmp1 = in_ptr1[i0];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = tmp2 * (tmp2>0);
                    out_ptr0[i0] = tmp3;
                }
            }
        }
    }
}
'''

# fmt: on

# kernel generation
cpp_kernel = CppCodeCache.load(cpp_str).kernel
tvm_kernel = tvm.build(Module, target="llvm --num-cores=16")
tuned_tvm_kernel = tvm.build(Module_tuned, "llvm --num-cores=16")

# input definition and kernel execution
inp0 = torch.rand(802816)
inp1 = torch.rand(802816)
out0 = torch.zeros(802816)
inp0_tvm = tvm.nd.array(inp0.numpy())
inp1_tvm = tvm.nd.array(inp1.numpy())
out0_tvm = tvm.nd.array(out0.numpy())


# correctness check
cpp_kernel(
    c_void_p(inp0.data_ptr()), c_void_p(inp1.data_ptr()), c_void_p(out0.data_ptr())
)
tvm_kernel(inp0_tvm, inp1_tvm, out0_tvm)
if torch.allclose(torch.from_numpy(out0_tvm.numpy()), out0):
    print("Inductor-tir result is correct!")
# clear out the output from last run
out0_tvm = tvm.nd.array(torch.zeros(802816).numpy())
tuned_tvm_kernel(inp0_tvm, inp1_tvm, out0_tvm)
if torch.allclose(out0, torch.from_numpy(out0_tvm.numpy())):
    print("Tuned Inductor-tir result is correct!")

# inference perf
print("Inductor TIR codegen perf:")
print_performance(tvm_kernel, (inp0_tvm, inp1_tvm, out0_tvm))
print("Inductor CPP/OpenMP codegen perf:")
print_performance(
    cpp_kernel,
    (c_void_p(inp0.data_ptr()), c_void_p(inp1.data_ptr()), c_void_p(out0.data_ptr())),
)
print("Meta schedule tuned TIR codegen perf:")
print_performance(tuned_tvm_kernel, (inp0_tvm, inp1_tvm, out0_tvm))
## tune perf
# with tempfile.TemporaryDirectory() as work_dir:
#     target = Target("llvm --num-cores=16")
# database = ms.tir_integration.tune_tir(
#     mod=Module_manual_rewrite,
#     target=target,
#     work_dir=work_dir,
#     max_trials_global=64,
#     num_trials_per_iter=64,
# )
# sch = ms.tir_integration.compile_tir(database, Module_manual_rewrite, target)
# print(sch.mod.script())
# print("handwritten TIR codegen perf:")
# rt_mod = tvm.build(Module_manual_rewrite, target)

# inp0 = torch.rand(8, 16, 28, 28, 8)
# print_performance(
#     rt_mod,
#     (
#         tvm.nd.array(inp0.numpy()),
#         tvm.nd.array(inp0.numpy()),
#         tvm.nd.array(inp0.numpy()),
#     ),
# )
