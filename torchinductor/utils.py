import functools
import operator
import time
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import sympy
import torch
from torch.cuda import synchronize
from torch.fx.immutable_collections import immutable_dict
from torch.fx.immutable_collections import immutable_list

VarRanges = Dict[sympy.Expr, sympy.Expr]


@functools.lru_cache(None)
def has_triton():
    try:
        import triton

        return triton is not None
    except (ImportError, ModuleNotFoundError):
        return False

@functools.lru_cache(None)
def has_tvm():
    try:
        import tvm

        return tvm is not None
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_triton_libdevice():
    try:
        from triton.language import libdevice

        return libdevice is not None
    except (ImportError, ModuleNotFoundError):
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it):
    return {id(x): x for x in it}.values()


def ceildiv(numer: int, denom: int):
    assert isinstance(numer, int) and isinstance(denom, int)
    return (numer + (denom - 1)) // denom


def gen_gm_and_inputs(target, args, kwargs):
    g = torch.fx.Graph()
    g_args = []
    a_args = []
    for n, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            g_args.append(g.placeholder(f"arg{n}"))
            a_args.append(arg)
        else:
            g_args.append(arg)
    assert all(not isinstance(x, torch.Tensor) for x in kwargs.values())
    node = g.call_function(target, tuple(g_args), kwargs)
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, a_args


def timed(model, example_inputs, times=1):
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return t1 - t0


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    timings = [timed(fn, args, times) for _ in range(repeat)]
    took = np.median(timings)
    print(f"{took/baseline:.6f}")
    return took


immutable_dict.__hash__ = lambda self: hash(tuple(self.items()))
immutable_list.__hash__ = lambda self: hash(tuple(self))


def freeze_inputs(f):
    """
    Useful for wrapping lists in tuples for caching purposes
    """

    def freeze_value(x):
        if isinstance(x, (immutable_dict, immutable_list)):
            return x
        if isinstance(x, list):
            return immutable_list(x)
        if isinstance(x, dict):
            return immutable_dict(x)
        return x

    @functools.wraps(f)
    def wrapped(*args):
        args = [freeze_value(x) for x in args]
        return f(*args)

    wrapped.cache_info = f.cache_info
    return wrapped


def precompute_method(obj: Any, method: str):
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: List[str]):
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a, b):
    return int(a > b) - int(a < b)


def cache_on_self(fn):
    key = f"__{fn.__name__}_cache"

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)

    return wrapper


def sympy_str(expr: sympy.Expr):
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """
    if isinstance(expr, sympy.Symbol):
        return expr.name
    if isinstance(expr, sympy.Add):
        return " + ".join(map(sympy_str, expr.args))
    if isinstance(expr, sympy.Mul):
        return " * ".join(map(sympy_str, expr.args))

    from .ir import CleanDiv
    from .ir import IndexingDiv
    from .ir import ModularIndexing

    if isinstance(expr, (ModularIndexing, CleanDiv, IndexingDiv)):
        return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
    return str(expr)


def sympy_subs(expr: sympy.Expr, replacements: Dict[Any, Any]):
    """
    xreplace is faster than subs, but is way more picky
    """

    def promote_strings(key):
        if isinstance(key, str):
            return sympy.Symbol(key)
        return key

    return expr.xreplace(
        {promote_strings(k): promote_strings(v) for k, v in replacements.items()}
    )


def free_symbol_startswith(index: sympy.Expr, prefix: str):
    return any(v.name.startswith(prefix) for v in index.free_symbols)


def has_incompatible_cudagraph_ops(gm):
    forbidden_list = set(
        [
            "aten._fused_moving_avg_obs_fq_helper.default",
            "aten._fused_moving_avg_obs_fq_helper_functional.default",
            "fbgemm.dense_to_jagged.default",
            "fbgemm.jagged_to_padded_dense.default",
        ]
    )
    for node in gm.graph.nodes:
        if str(node.target) in forbidden_list:
            return True
    return False
