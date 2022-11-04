import tvm
from tvm.script import tir as T
from ctypes import c_void_p
from tvm.target import Target
from tvm import meta_schedule as ms
import tempfile

import torch
import numpy as np

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
    def main(p0: T.Buffer[(64, 3, 7, 7), "float32"], p1: T.Buffer[(8, 3, 224, 224), "float32"], out_ptr0: T.Buffer[(8, 64, 112, 112), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_layout_trans = T.alloc_buffer([16, 1, 7, 7, 3, 4], dtype="float32")
        T_layout_trans_1 = T.alloc_buffer([8, 1, 224, 224, 3], dtype="float32")
        data_pad = T.alloc_buffer([8, 1, 230, 230, 3], dtype="float32")
        conv2d_NCHWc = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        for i0, i1, i2, i3, i4, i5 in T.grid(16, 1, 7, 7, 3, 4):
            with T.block("T_layout_trans"):
                ax0, ax1, ax2, ax3, ax4, ax5 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                T.reads(p0[ax0 * 4 + ax5, ax1 * 3 + ax4, ax2, ax3])
                T.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4, ax5])
                T_layout_trans[ax0, ax1, ax2, ax3, ax4, ax5] = T.if_then_else(ax0 * 4 + ax5 < 64 and ax1 * 3 + ax4 < 3 and ax2 < 7 and ax3 < 7, p0[ax0 * 4 + ax5, ax1 * 3 + ax4, ax2, ax3], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(8, 1, 224, 224, 3):
            with T.block("T_layout_trans_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(p0[ax0, ax1 * 3 + ax4, ax2, ax3])
                T.writes(T_layout_trans_1[ax0, ax1, ax2, ax3, ax4])
                T_layout_trans_1[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(ax0 < 8 and ax1 * 3 + ax4 < 3 and ax2 < 224 and ax3 < 224, p1[ax0, ax1 * 3 + ax4, ax2, ax3], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(8, 1, 230, 230, 3):
            with T.block("data_pad"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1])
                T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227, T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(8, 16, 112, 112, 4, 3, 7, 7):
            with T.block("conv2d_NCHWc"):
                n, oc_chunk, oh, ow, oc_block, ic, kh, kw = T.axis.remap("SSSSSRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                T.reads(data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3], T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
                T.writes(conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block])
                T.block_attr({"workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                with T.init():
                    conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3] * T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block]
        for i0, i1, i2, i3 in T.grid(8, 64, 112, 112):
            with T.block("out_ptr0"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(conv2d_NCHWc[ax0, ax1 // 4, ax2, ax3, ax1 % 4])
                T.writes(out_ptr0[ax0, ax1, ax2, ax3])
                out_ptr0[ax0, ax1, ax2, ax3] = T.if_then_else(ax0 < 8 and ax1 < 64 and ax2 < 112 and ax3 < 112, conv2d_NCHWc[ax0, ax1 // 4, ax2, ax3, ax1 % 4], T.float32(0), dtype="float32")



# from tvm.script import tir as T
@tvm.script.ir_module
class Module_tuned:
    @T.prim_func
    def main(p0: T.Buffer[(64, 3, 7, 7), "float32"], p1: T.Buffer[(8, 3, 224, 224), "float32"], out_ptr0: T.Buffer[(8, 64, 112, 112), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_layout_trans = T.alloc_buffer([16, 1, 7, 7, 3, 4], dtype="float32")
        T_layout_trans_1 = T.alloc_buffer([8, 1, 224, 224, 3], dtype="float32")
        data_pad = T.alloc_buffer([8, 1, 230, 230, 3], dtype="float32")
        conv2d_NCHWc = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        conv2d_NCHWc_global = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        for i0_i1_i2_i3_fused in T.parallel(784, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i4 in T.serial(3):
                for i5_fused in T.vectorized(4):
                    with T.block("T_layout_trans"):
                        ax0 = T.axis.spatial(16, i0_i1_i2_i3_fused // 49)
                        ax1 = T.axis.spatial(1, 0)
                        ax2 = T.axis.spatial(7, i0_i1_i2_i3_fused % 49 // 7)
                        ax3 = T.axis.spatial(7, i0_i1_i2_i3_fused % 7)
                        ax4, ax5 = T.axis.remap("SS", [i4, i5_fused])
                        T.reads(p0[ax0 * 4 + ax5, ax1 * 3 + ax4, ax2, ax3])
                        T.writes(T_layout_trans[ax0, ax1, ax2, ax3, ax4, ax5])
                        T_layout_trans[ax0, ax1, ax2, ax3, ax4, ax5] = T.if_then_else(ax0 * 4 + ax5 < 64 and ax1 * 3 + ax4 < 3 and ax2 < 7 and ax3 < 7, p0[ax0 * 4 + ax5, ax1 * 3 + ax4, ax2, ax3], T.float32(0), dtype="float32")
        for i0_i1_i2_fused in T.parallel(1792, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i3 in T.serial(224):
                for i4_fused in T.vectorized(3):
                    with T.block("T_layout_trans_1"):
                        ax0 = T.axis.spatial(8, i0_i1_i2_fused // 224)
                        ax1 = T.axis.spatial(1, 0)
                        ax2 = T.axis.spatial(224, i0_i1_i2_fused % 224)
                        ax3, ax4 = T.axis.remap("SS", [i3, i4_fused])
                        T.reads(p0[ax0, ax1 * 3 + ax4, ax2, ax3])
                        T.writes(T_layout_trans_1[ax0, ax1, ax2, ax3, ax4])
                        T_layout_trans_1[ax0, ax1, ax2, ax3, ax4] = T.if_then_else(ax0 < 8 and ax1 * 3 + ax4 < 3 and ax2 < 224 and ax3 < 224, p1[ax0, ax1 * 3 + ax4, ax2, ax3], T.float32(0), dtype="float32")
        for i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused in T.parallel(56, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 229, 37):
                for ax4_fused in T.vectorized(3):
                    with T.block("data_pad"):
                        i0_1 = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused % 8 + ax0)
                        i1_1 = T.axis.spatial(1, ax1)
                        i2_1 = T.axis.spatial(230, ax2)
                        i3_1 = T.axis.spatial(230, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused // 8 * 32 + ax3)
                        i4_1 = T.axis.spatial(3, ax4_fused)
                        T.reads(T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1])
                        T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                        data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227, T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1], T.float32(0), dtype="float32")
            for i1_1, i2_1, i3_1, i4_1 in T.grid(1, 14, 4, 2):
                for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i4_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init, i4_3_init in T.grid(1, 8, 4, 1, 2, 1, 2, 2, 4, 1):
                    with T.block("conv2d_NCHWc_init"):
                        n = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused % 8 + i0_2_init + i0_3_init)
                        oc_chunk = T.axis.spatial(16, i1_1 * 16 + i1_2_init * 2 + i1_3_init)
                        oh = T.axis.spatial(112, i2_1 * 8 + i2_2_init * 2 + i2_3_init)
                        ow = T.axis.spatial(112, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused // 8 * 16 + i3_1 * 4 + i3_2_init * 4 + i3_3_init)
                        oc_block = T.axis.spatial(4, i4_3_init + i4_1 * 2 + i4_2_init)
                        T.reads()
                        T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                        conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                for i5_0, i6_0, i7_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 7, 7, 1, 8, 4, 1, 2, 3, 1, 1, 1, 2, 2, 4, 1):
                    with T.block("conv2d_NCHWc_update"):
                        n = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused % 8 + i0_2 + i0_3)
                        oc_chunk = T.axis.spatial(16, i1_1 * 16 + i1_2 * 2 + i1_3)
                        oh = T.axis.spatial(112, i2_1 * 8 + i2_2 * 2 + i2_3)
                        ow = T.axis.spatial(112, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused // 8 * 16 + i3_1 * 4 + i3_2 * 4 + i3_3)
                        oc_block = T.axis.spatial(4, i4_3 + i4_1 * 2 + i4_2)
                        ic = T.axis.reduce(3, i5_0 * 3 + i5_1)
                        kh = T.axis.reduce(7, i6_1 + i6_0)
                        kw = T.axis.reduce(7, i7_0 + i7_1)
                        T.reads(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block], data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3], T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
                        T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                        T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                        conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3] * T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block]
                for ax0, ax1, ax2, ax3 in T.grid(1, 16, 8, 4):
                    for ax4_fused in T.vectorized(2):
                        with T.block("conv2d_NCHWc_global"):
                            v0 = T.axis.spatial(8, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused % 8 + ax0)
                            v1 = T.axis.spatial(16, ax1)
                            v2 = T.axis.spatial(112, i2_1 * 8 + ax2)
                            v3 = T.axis.spatial(112, i0_0_i1_0_i2_0_i3_0_i4_0_i0_1_fused // 8 * 16 + i3_1 * 4 + ax3)
                            v4 = T.axis.spatial(4, i4_1 * 2 + ax4_fused)
                            T.reads(conv2d_NCHWc_global[v0, v1, v2, v3, v4])
                            T.writes(conv2d_NCHWc[v0, v1, v2, v3, v4])
                            conv2d_NCHWc[v0, v1, v2, v3, v4] = conv2d_NCHWc_global[v0, v1, v2, v3, v4]
        for i0_i1_fused in T.parallel(512, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i2, i3 in T.grid(112, 112):
                with T.block("out_ptr0"):
                    ax0 = T.axis.spatial(8, i0_i1_fused // 64)
                    ax1 = T.axis.spatial(64, i0_i1_fused % 64)
                    ax2, ax3 = T.axis.remap("SS", [i2, i3])
                    T.reads(conv2d_NCHWc[ax0, ax1 // 4, ax2, ax3, ax1 % 4])
                    T.writes(out_ptr0[ax0, ax1, ax2, ax3])
                    out_ptr0[ax0, ax1, ax2, ax3] = T.if_then_else(ax0 < 8 and ax1 < 64 and ax2 < 112 and ax3 < 112, conv2d_NCHWc[ax0, ax1 // 4, ax2, ax3, ax1 % 4], T.float32(0), dtype="float32")

@tvm.script.ir_module
class Module_tuned_full:
    @T.prim_func
    def main(p0: T.Buffer[(64, 3, 7, 7), "float32"], p1: T.Buffer[(8, 3, 224, 224), "float32"], out_ptr0: T.Buffer[(8, 64, 112, 112), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_layout_trans = T.alloc_buffer([16, 1, 7, 7, 3, 4], dtype="float32")
        T_layout_trans_1 = T.alloc_buffer([8, 1, 224, 224, 3], dtype="float32")
        data_pad = T.alloc_buffer([8, 1, 230, 230, 3], dtype="float32")
        conv2d_NCHWc = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        conv2d_NCHWc_global = T.alloc_buffer([8, 16, 112, 112, 4], dtype="float32")
        for i0_i1_i2_fused in T.parallel(1840, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 224):
                for ax4_fused in T.vectorized(3):
                    with T.block("T_layout_trans_1"):
                        T.where(3 <= i0_i1_i2_fused % 230 and i0_i1_i2_fused % 230 < 227)
                        ax0_1 = T.axis.spatial(8, i0_i1_i2_fused // 230 + ax0)
                        ax1_1 = T.axis.spatial(1, ax1)
                        ax2_1 = T.axis.spatial(224, i0_i1_i2_fused % 230 + -3 + ax2)
                        ax3_1, ax4 = T.axis.remap("SS", [ax3, ax4_fused])
                        T.reads(p0[ax0_1, ax1_1 * 3 + ax4, ax2_1, ax3_1])
                        T.writes(T_layout_trans_1[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                        T_layout_trans_1[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = T.if_then_else(ax0_1 < 8 and ax1_1 * 3 + ax4 < 3 and ax2_1 < 224 and ax3_1 < 224, p1[ax0_1, ax1_1 * 3 + ax4, ax2_1, ax3_1], T.float32(0), dtype="float32")
            for i3 in T.serial(230):
                for i4_fused in T.vectorized(3):
                    with T.block("data_pad"):
                        i0_1 = T.axis.spatial(8, i0_i1_i2_fused // 230)
                        i1_1 = T.axis.spatial(1, 0)
                        i2_1 = T.axis.spatial(230, i0_i1_i2_fused % 230)
                        i3_1, i4_1 = T.axis.remap("SS", [i3, i4_fused])
                        T.reads(T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1])
                        T.writes(data_pad[i0_1, i1_1, i2_1, i3_1, i4_1])
                        data_pad[i0_1, i1_1, i2_1, i3_1, i4_1] = T.if_then_else(3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227, T_layout_trans_1[i0_1, i1_1, i2_1 - 3, i3_1 - 3, i4_1], T.float32(0), dtype="float32")
        for i0_0_i1_0_i2_0_fused in T.parallel(896, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(4, 1, 7, 7, 3):
                for ax5_fused in T.vectorized(4):
                    with T.block("T_layout_trans"):
                        ax0_2 = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + ax0)
                        ax1_2, ax2_2, ax3_2, ax4_1, ax5 = T.axis.remap("SSSSS", [ax1, ax2, ax3, ax4, ax5_fused])
                        T.reads(p0[ax0_2 * 4 + ax5, ax1_2 * 3 + ax4_1, ax2_2, ax3_2])
                        T.writes(T_layout_trans[ax0_2, ax1_2, ax2_2, ax3_2, ax4_1, ax5])
                        T_layout_trans[ax0_2, ax1_2, ax2_2, ax3_2, ax4_1, ax5] = T.if_then_else(ax0_2 * 4 + ax5 < 64 and ax1_2 * 3 + ax4_1 < 3 and ax2_2 < 7 and ax3_2 < 7, p0[ax0_2 * 4 + ax5, ax1_2 * 3 + ax4_1, ax2_2, ax3_2], T.float32(0), dtype="float32")
            for i3_0, i4_0, i0_1, i1_1, i2_1, i3_1, i4_1 in T.grid(1, 1, 4, 1, 1, 56, 1):
                for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i4_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init in T.grid(1, 1, 1, 1, 1, 1, 4, 1, 2):
                    for i4_3_fused_init in T.vectorized(4):
                        with T.block("conv2d_NCHWc_init"):
                            n = T.axis.spatial(8, i0_2_init + i0_3_init + i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1)
                            oc_chunk = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + i1_1 * 4 + i1_2_init * 4 + i1_3_init)
                            oh = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + i2_1 + i2_2_init + i2_3_init)
                            ow = T.axis.spatial(112, i3_0 * 112 + i3_1 * 2 + i3_2_init * 2 + i3_3_init)
                            oc_block = T.axis.spatial(4, i4_0 * 4 + i4_1 * 4 + i4_2_init * 4 + i4_3_fused_init)
                            T.reads()
                            T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                            conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = T.float32(0)
                for i5_0, i6_0, i7_0, i0_2, i1_2, i2_2, i3_2, i4_2, i5_1, i6_1, i7_1, i0_3, i1_3, i2_3, i3_3 in T.grid(3, 7, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 4, 1, 2):
                    for i4_3_fused in T.vectorized(4):
                        with T.block("conv2d_NCHWc_update"):
                            n = T.axis.spatial(8, i0_2 + i0_3 + i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1)
                            oc_chunk = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + i1_1 * 4 + i1_2 * 4 + i1_3)
                            oh = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + i2_1 + i2_2 + i2_3)
                            ow = T.axis.spatial(112, i3_0 * 112 + i3_1 * 2 + i3_2 * 2 + i3_3)
                            oc_block = T.axis.spatial(4, i4_0 * 4 + i4_1 * 4 + i4_2 * 4 + i4_3_fused)
                            ic = T.axis.reduce(3, i5_1 + i5_0)
                            kh = T.axis.reduce(7, i6_0 + i6_1)
                            kw = T.axis.reduce(7, i7_0 * 7 + i7_1)
                            T.reads(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block], data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3], T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block])
                            T.writes(conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block])
                            T.block_attr({"meta_schedule.tiling_structure":"SSRSRS", "workload":["conv2d_NCHWc.x86", ["TENSOR", [8, 1, 224, 224, 3], "float32"], ["TENSOR", [16, 1, 7, 7, 3, 4], "float32"], [2, 2], [3, 3, 3, 3], [1, 1], "NCHW3c", "NCHW4c", "float32"]})
                            conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_global[n, oc_chunk, oh, ow, oc_block] + data_pad[n, ic // 3, oh * 2 + kh, ow * 2 + kw, ic % 3] * T_layout_trans[oc_chunk, ic // 3, kh, kw, ic % 3, oc_block]
                for ax0_3, ax1_3, ax2_3 in T.grid(1, 4, 1):
                    for ax3_ax4_fused in T.vectorized(8):
                        with T.block("conv2d_NCHWc_global"):
                            v0 = T.axis.spatial(8, i0_0_i1_0_i2_0_fused // 448 * 4 + i0_1 + ax0_3)
                            v1 = T.axis.spatial(16, i0_0_i1_0_i2_0_fused % 448 // 112 * 4 + ax1_3)
                            v2 = T.axis.spatial(112, i0_0_i1_0_i2_0_fused % 112 + ax2_3)
                            v3 = T.axis.spatial(112, i3_1 * 2 + ax3_ax4_fused // 4)
                            v4 = T.axis.spatial(4, ax3_ax4_fused % 4)
                            T.reads(conv2d_NCHWc_global[v0, v1, v2, v3, v4])
                            T.writes(conv2d_NCHWc[v0, v1, v2, v3, v4])
                            conv2d_NCHWc[v0, v1, v2, v3, v4] = conv2d_NCHWc_global[v0, v1, v2, v3, v4]
        for i0_i1_fused in T.parallel(512, annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
            for i2, i3 in T.grid(112, 112):
                with T.block("out_ptr0"):
                    ax0_4 = T.axis.spatial(8, i0_i1_fused // 64)
                    ax1_4 = T.axis.spatial(64, i0_i1_fused % 64)
                    ax2_4, ax3_3 = T.axis.remap("SS", [i2, i3])
                    T.reads(conv2d_NCHWc[ax0_4, ax1_4 // 4, ax2_4, ax3_3, ax1_4 % 4])
                    T.writes(out_ptr0[ax0_4, ax1_4, ax2_4, ax3_3])
                    out_ptr0[ax0_4, ax1_4, ax2_4, ax3_3] = T.if_then_else(ax0_4 < 8 and ax1_4 < 64 and ax2_4 < 112 and ax3_3 < 112, conv2d_NCHWc[ax0_4, ax1_4 // 4, ax2_4, ax3_3, ax1_4 % 4], T.float32(0), dtype="float32")


# fmt: on

# kernel generation

aten = torch.ops.aten


def eager_fn(inp0, inp1):
    return aten.convolution(inp0, inp1, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1)


tvm_kernel = tvm.build(Module_manual_rewrite, target="llvm --num-cores=16")
tuned_tvm_kernel = tvm.build(Module_tuned, "llvm --num-cores=16")
tuned_tvm_kernel_full = tvm.build(Module_tuned_full, "llvm --num-cores=16")

# input definition and kernel execution
from torchdynamo.testing import rand_strided

inp0 = rand_strided(
    (8, 3, 224, 224), (150528, 50176, 224, 1), device="cpu", dtype=torch.float32
)
inp1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device="cpu", dtype=torch.float32)

out0 = torch.zeros((8, 64, 112, 112), device="cpu", dtype=torch.float32)
inp0_tvm = tvm.nd.array(inp0)
inp1_tvm = tvm.nd.array(inp1)
out0_tvm = tvm.nd.array(out0.numpy())


# correctness check
# eager_out = eager_fn(inp0, inp1)
# print(eager_out.shape)
# tvm_kernel(inp1_tvm, inp0_tvm, out0_tvm)
# print(out0_tvm.numpy().shape)
# np.testing.assert_allclose(eager_out.numpy(), out0_tvm.numpy(), rtol=1e-4, atol=1e-4)
# print("Inductor-tir result is correct!")
# clear out the output from last run
# out0_tvm = tvm.nd.array(out0.numpy())
# tuned_tvm_kernel(inp1_tvm, inp0_tvm, out0_tvm)
# np.testing.assert_allclose(eager_out.numpy(), out0_tvm.numpy(), rtol=1e-4, atol=1e-4)
# print("Tuned Inductor-tir result is correct!")
# from tvm import relay

# input_data = [inp0, inp1]
# trace = torch.jit.trace(eager_fn, [input.clone() for input in input_data])
# input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
# input_shapes = list(zip(input_names, [inp.shape for inp in input_data]))
# mod, params = relay.frontend.from_pytorch(trace, input_shapes, {})
# from tvm.contrib import graph_executor

# relay_kernel = relay.create_executor(
#     kind="graph", mod=mod, params=params, device=tvm.device("cuda", 0), target="cuda"
# ).evaluate()
# input_names = [f"input{idx}" for idx, _ in enumerate(input_data)]
# compiled_input = dict(
#     zip(
#         input_names,
#         [inp.clone().cpu().numpy() for inp in input_data],
#     )
# )
# print_performance(relay_kernel, compiled_input, times=1, repeat=1)

# # inference perf
out0_tvm = tvm.nd.array(out0.numpy())
print("Inductor TIR codegen perf:")
print_performance(tvm_kernel, (inp1_tvm, inp0_tvm, out0_tvm), times=1, repeat=1)
out0_tvm = tvm.nd.array(out0.numpy())
print("Inductor CPP/OpenMP codegen perf:")
print_performance(eager_fn, (inp0, inp1), times=10, repeat=1)
out0_tvm = tvm.nd.array(out0.numpy())
print("Meta schedule tuned TIR codegen perf:")
print_performance(tuned_tvm_kernel, (inp1_tvm, inp0_tvm, out0_tvm), times=10, repeat=1)

# # ## tune perf
# with tempfile.TemporaryDirectory() as work_dir:
#     target = Target("llvm --num-cores=16")
#     database = ms.tir_integration.tune_tir(
#         mod=Module_manual_rewrite,
#         target=target,
#         work_dir=work_dir,
#         max_trials_global=1000,
#         num_trials_per_iter=64,
#     )
#     sch = ms.tir_integration.compile_tir(database, Module_manual_rewrite, target)
#     print(sch.mod.script())
#     print("handwritten TIR codegen perf:")
#     rt_mod = tvm.build(sch.mod, target)
#     print_performance(
#         rt_mod,
#         (
#             inp1_tvm,
#             inp0_tvm,
#             tvm.nd.array(out0.numpy()),
#         ),
#     )
