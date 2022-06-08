import collections
import contextlib
import functools
import itertools
import operator
import re
import textwrap
import typing
from io import StringIO
from itertools import chain

import sympy
from sympy.printing.printer import Printer

from ..virtualized import V
from ..virtualized import ops


def product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


class ExprPrinter(Printer):
    @staticmethod
    def paren(string):
        if (
            re.match(r"^[a-z0-9_.]+$", string, re.I)
            or re.match(r"^\([^)]*\)$", string, re.I)
            or string == ""
        ):
            return string
        return f"({string})"

    def _print_Pow(self, expr):
        # Pow() confuses triton
        base, exp = expr.args
        base = self._print(base)
        assert exp.is_integer
        exp = int(exp)
        return "*".join([self.paren(base)] * exp)

    def _print_Mul(self, expr):
        return "*".join(map(self.paren, map(self._print, expr.args)))

    def _print_Add(self, expr):
        return " + ".join(map(self.paren, map(self._print, expr.args)))

    def _print_Mod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_CleanDiv(self, expr):
        return self._print_IndexingDiv(expr)


class OpOverrides:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def identity(value):
        # used to trigger cse
        return value

    @staticmethod
    def constant(value, dtype):
        return repr(value)

    @staticmethod
    def sigmoid(x):
        x = ops.exp(f"-{x}")
        return f"1 / (1 + {x})"

    @staticmethod
    def silu(x):
        return f"{x} * {ops.sigmoid(x)}"

    @staticmethod
    def reciprocal(x):
        return ops.div("1", x)

    @staticmethod
    def square(x):
        return ops.mul(x, x)

    @staticmethod
    def sign(x):
        return ops.where(f"{x} < 0", "-1", "1")

    @staticmethod
    def bitwise_not(x):
        return f"~{ExprPrinter.paren(x)}"

    @staticmethod
    def logical_not(a):
        return f"{ExprPrinter.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x, y):
        return f"{ExprPrinter.paren(x)} & {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_or(x, y):
        return f"{ExprPrinter.paren(x)} | {ExprPrinter.paren(y)}"

    @staticmethod
    def bitwise_xor(x, y):
        return f"{ExprPrinter.paren(x)} ^ {ExprPrinter.paren(y)}"

    @staticmethod
    def remainder(a, b):
        r = ops.mod(a, b)
        return ops.where(f"(({r} != 0) & (({r} < 0) != ({b} < 0)))", ops.add(r, b), r)


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self.contents = StringIO()
        self._indent = initial_indent
        self.getvalue = self.contents.getvalue

    def __bool__(self):
        return len(self.getvalue()) > 0

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def writeline(self, line):
        self.contents.write(self.prefix())
        self.contents.write(line)
        self.contents.write("\n")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            yield
            self._indent -= offset

        return ctx()

    def splice(self, other_code, strip=False):
        if isinstance(other_code, IndentedBuffer):
            other_code = other_code.getvalue()
        other_code = textwrap.dedent(other_code)
        if strip:
            other_code = other_code.lstrip()
        if not other_code:
            return
        assert other_code.endswith("\n")
        self.contents.write(textwrap.indent(other_code, self.prefix()))


class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


class InplacedBuffer(typing.NamedTuple):
    inner_name: str
    other_names: typing.List[str]


def unique(it):
    return {id(x): x for x in it}.values()


class KernelArgs:
    @staticmethod
    def _lookup(prefix, odict, name):
        assert isinstance(name, (str, sympy.Symbol))
        name = str(name)
        if name not in odict:
            odict[name] = f"{prefix}{len(odict)}"
        return odict[name]

    def __init__(self, sizevars=None):
        self.input_buffers = collections.OrderedDict()
        self.output_buffers = collections.OrderedDict()
        self.inplace_buffers = collections.OrderedDict()
        self.sizevars = sizevars or collections.OrderedDict()

    def input(self, name):
        name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        if name in self.output_buffers:
            return self.output_buffers[name]
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name):
        name = V.graph.scheduler.mutation_real_name.get(name, name)
        assert name not in V.graph.removed_buffers, name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name, output_name):
        buf = InplacedBuffer(
            f"in_out_ptr{len(self.inplace_buffers)}", [input_name, output_name]
        )
        self.inplace_buffers[input_name] = buf
        self.inplace_buffers[output_name] = buf

    def size(self, name):
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def cpp_argdefs(self):
        from .cpp import DTYPE_TO_CPP
        from .cpp import INDEX_TYPE

        # TODO(jansel): replace this with data from scheduler
        buffer_types = {x.get_name(): x.get_dtype() for x in V.graph.buffers}
        buffer_types.update(
            {name: val.get_dtype() for name, val in V.graph.graph_inputs.items()}
        )
        buffer_types.update(
            {name: val.dtype for name, val in V.graph.constants.items()}
        )

        call_args = []
        arg_defs = []
        for inplaced in unique(self.inplace_buffers.values()):
            outer = inplaced.other_names[0]
            inner = inplaced.inner_name
            dtype = buffer_types[outer]
            arg_defs.append(f"{DTYPE_TO_CPP[dtype]}* __restrict__ {inner}")
            name = inplaced.other_names[-1]
            call_args.append(f"c_void_p({name}.data_ptr())")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = buffer_types[outer]
            arg_defs.append(f"const {DTYPE_TO_CPP[dtype]}* __restrict__ {inner}")
            call_args.append(f"c_void_p({outer}.data_ptr())")
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = buffer_types[outer]
            arg_defs.append(f"{DTYPE_TO_CPP[dtype]}* __restrict__ {inner}")
            call_args.append(f"c_void_p({outer}.data_ptr())")
        for outer, inner in self.sizevars.items():
            arg_defs.append(f"const {INDEX_TYPE} {inner}")
            call_args.append(f"c_long({outer})")
        return arg_defs, call_args

    def python_argdefs(self):
        arg_defs = []
        call_args = []
        for inplaced in unique(self.inplace_buffers.values()):
            arg_defs.append(inplaced.inner_name)
            call_args.append(inplaced.other_names[-1])
        for outer, inner in chain(
            self.input_buffers.items(), self.output_buffers.items()
        ):
            if outer in self.inplace_buffers:
                continue
            arg_defs.append(inner)
            call_args.append(outer)
        for outer, inner in self.sizevars.items():
            arg_defs.append(inner)
            call_args.append(outer)
        return arg_defs, call_args

    def aliases(self):
        for inplaced in unique(self.inplace_buffers.values()):
            for other in inplaced.other_names:
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield self.output_buffers[other], inplaced.inner_name


class CSE:
    """Common subexpression elimination"""

    def __init__(
        self,
        prefix="",
        suffix="",
        name_prefix="tmp",
        iter_buffers=None,
        store_cache=None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self.cache = {}
        self.name_prefix = name_prefix
        self.store_cache = store_cache or {}
        self.iter_buffer_ids = iter_buffers or itertools.count()

    def clone(self):
        return CSE(
            self.prefix,
            self.suffix,
            self.name_prefix,
            self.iter_buffer_ids,
            self.store_cache,
        )

    def generate(self, buffer: IndentedBuffer, expr: str, write=True):
        assert isinstance(expr, str), expr
        if expr.startswith(self.name_prefix) and re.match(r"^[a-z0-9]+$", expr):
            return expr
        if expr not in self.cache:
            var = self.newvar()
            self.cache[expr] = var
            if write:
                buffer.writeline(f"{self.prefix}{var} = {expr}{self.suffix}")
        return self.cache[expr]

    def newvar(self):
        return f"{self.name_prefix}{next(self.iter_buffer_ids)}"


class CodeGen:
    def __init__(self):
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class Kernel(CodeGen):
    newvar_prefix = ""
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, args=None):
        super().__init__()
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = CSE(self.newvar_prefix, self.suffix)

    @contextlib.contextmanager
    def swap_buffers(self, lb, cb=None, sb=None):
        if cb is None:
            cb = lb
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.clone()
        yield
        self.loads = loads
        self.compute = compute
        self.stores = stores
        self.cse = cse

    def load(self, name: str, index: sympy.Expr, upcast: bool = False, store_cache_key: sympy.Expr = None):
        raise NotImplementedError()

    def indirect_load(self, name: str, index: sympy.Expr, upcast: bool = False, store_cache_key: sympy.Expr = None):
        """A load the depends on an index we have read"""
        prior = self.loads
        try:
            # put the load in the compute section as it might have deps
            self.loads = self.compute
            return self.load(name, index, upcast, store_cache_key)
        finally:
            self.loads = prior

    def store(self, name, index, value, store_cache_key = None):
        raise NotImplementedError()

    def reduction(self, name, dtype, reduction_type, index, value):
        raise NotImplementedError()

    def __enter__(self):
        class CSEProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    return self.cse.generate(
                        self.compute, getattr(parent_handler, name)(*args, **kwargs)
                    )

                return inner

            @staticmethod
            def indirect_indexing(index_var):
                return sympy.Symbol(str(index_var))

            @staticmethod
            def load(name: str, index: sympy.Expr, upcast: bool = False, store_cache_key: sympy.Expr = None):
                if "tmp" in str(index):
                    return self.indirect_load(name, index, upcast, store_cache_key)
                store_cache = self.cse.store_cache
                key = store_cache_key if store_cache_key else index
                if (name, key) in store_cache:
                    return store_cache[(name, key)]
                key = self.rename_indexing(index)
                if (name, key) in store_cache:
                    # TODO(jansel): figure out why we need this second case
                    return store_cache[(name, key)]
                return self.load(name, index, upcast, store_cache_key)

            @staticmethod
            def store(name, index, value, store_cache_key = None):
                self.cse.store_cache[(name, store_cache_key if store_cache_key else index)] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, store_cache_key)

            @staticmethod
            def reduction(name, dtype, reduction_type, index, value):
                return self.reduction(name, dtype, reduction_type, index, value)

        super().__enter__()
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def rename_indexing(self, index) -> sympy.Expr:
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = sympy.simplify(index.subs(V.graph.sizevars.replacements))
        subs = {
            x: self.args.size(x) for x in index.free_symbols if str(x).startswith("s")
        }
        return index.subs(subs)
