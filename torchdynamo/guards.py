import collections
import dataclasses
import enum
import logging
import os
import re
import textwrap
import types
import weakref
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
import torch

from . import config
from . import mutation_guard
from ._guards import TensorGuards
from ._guards import check_obj_id
from ._guards import check_type_id
from .eval_frame import set_guard_error_hook
from .eval_frame import set_guard_fail_hook
from .exc import unimplemented
from .utils import guard_failures
from .utils import istype
from .utils import orig_code_map
from .utils import rename_implicit
from .utils import tuple_iterator_getitem
from .utils import tuple_iterator_len

log = logging.getLogger(__name__)


CLOSURE_VARS = collections.OrderedDict(
    [
        ("___check_type_id", check_type_id),
        ("___check_obj_id", check_obj_id),
        ("___is_grad_enabled", torch.is_grad_enabled),
        ("___odict_getitem", collections.OrderedDict.__getitem__),
        ("___tuple_iterator_len", tuple_iterator_len),
        ("___tuple_iterator_getitem", tuple_iterator_getitem),
        ("inf", float("inf")),
    ]
)


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3

    def select(self, locals_, globals_):
        if self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE):
            return locals_
        if self in (GuardSource.GLOBAL, GuardSource.GLOBAL_NN_MODULE):
            return globals_
        raise NotImplementedError()

    def is_nn_module(self):
        return self in (GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_NN_MODULE)

    def is_local(self):
        return self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE)


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    create_fn: Callable

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def sort_key(self):
        return (
            self.source.value,
            len(self.name),
            self.name,
            self.create_fn.__code__.co_firstlineno,
        )

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def __str__(self):
        return f"{self.source.name.lower()} {repr(self.name)} {self.create_fn.__name__}"

    def create(self, local_builder: "GuardBuilder", global_builder: "GuardBuilder"):
        return self.create_fn(self.source.select(local_builder, global_builder), self)

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_local(self):
        return self.source.is_local()


def strip_function_call(name):
    """
    "___odict_getitem(a, 1)" => "a"
    """
    m = re.search(r"[a-z0-9_]+\(([^(),]+)[^()]*\)", name)
    if m:
        return strip_function_call(m.group(1))
    return strip_getattr_getitem(name)


def strip_getattr_getitem(name):
    """
    "a[1]" => "a"
    "a.foo" => "a"
    """
    return re.split(r"[.\[]", name)[0]


class GuardBuilder:
    def __init__(
        self, id_ref: Callable, scope: Dict[str, Any], guarded_code, renames=True
    ):
        self.id_ref = id_ref
        if scope:
            if renames:
                scope = {rename_implicit(k): v for k, v in scope.items()}
        else:
            scope = dict()
        self.scope = scope
        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[str] = []
        self.tensor_check_names = []
        self.tensor_check_examples = []
        self.guarded_code = guarded_code

    def get(self, name: str):
        return eval(name, self.scope, CLOSURE_VARS)

    def arg_ref(self, guard: Guard):
        if isinstance(guard, str):
            name = guard
        else:
            name = guard.name
        base = strip_getattr_getitem(strip_function_call(name))
        if base not in self.argnames:
            self.argnames.append(base)

        return name

    def TYPE_MATCH(self, guard: Guard):
        # ___check_type_id is same as `id(type(x)) == y`
        self.code.append(
            f"___check_type_id({self.arg_ref(guard)}, {self.id_ref(type(self.get(guard.name)))})"
        )

    def ID_MATCH(self, guard: Guard):
        # ___check_obj_id is same as `id(x) == y`
        m = re.match(r"^type\((.+)\)$", guard.name)
        if m:
            # optional optimization to produce cleaner/faster guard code
            return self.TYPE_MATCH(Guard(m.group(1), guard.source, None))
        self.code.append(
            f"___check_obj_id({self.arg_ref(guard)}, {self.id_ref(self.get(guard.name))})"
        )

    def HASATTR(self, guard: Guard):
        m = re.match(r"^(.*)[.]([a-zA-Z0-9_]+)$", guard.name)
        assert m, f"invalid hasattr check {guard.name}"
        base, attr = m.group(1, 2)
        ref = self.arg_ref(base)
        val = hasattr(self.get(base), attr)
        if val:
            self.code.append(f"hasattr({ref}, {attr!r})")
        else:
            self.code.append(f"not hasattr({ref}, {attr!r})")

    def EQUALS_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        assert istype(
            val,
            (
                int,
                float,
                bool,
                type(None),
                str,
                type,
                list,
                tuple,
                set,
                slice,
                frozenset,
                range,
                torch.Size,
                torch.device,
                torch.dtype,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ), type(val).__name__
        if istype(val, (torch.device, torch.dtype)):
            # TODO(jansel): is this slow? perhaps optimize it
            self.code.append(f"str({ref}) == {str(val)!r}")
            return

        # Add type check to prevent equality check between tensor and non-tensor.
        if istype(val, (list, tuple)):
            self.LIST_LENGTH(guard)
            for idx, elem in enumerate(val):
                self.code.append(
                    f"___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})"
                )
        elif not istype(val, torch.Size):
            self.code.append(f"___check_type_id({ref}, {self.id_ref(type(val))})")

        if istype(val, torch.Size):
            val = tuple(val)
        self.code.append(f"{ref} == {val!r}")

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, (bool, type(None))):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        self.ID_MATCH(guard)
        ref = self.arg_ref(guard)
        val = self.get(guard.name)

        def setup_guard():
            assert istype(val.training, bool)
            self.code.append(f"{ref}.training == {val.training}")

        if hasattr(val, "training"):
            # There are cases where a monkeypatched object has a guard made between __new__ and __init__
            setup_guard()
        else:
            unimplemented(f"Guard setup for uninitialized class {type(val)}")

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
        if guard.is_local():
            return self.ID_MATCH(guard)

    def BUILTIN_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def PYMODULE_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def LIST_LENGTH(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"len({ref}) == {len(value)}")

    def TUPLE_ITERATOR_LEN(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}")

    def DICT_KEYS(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"{ref}.keys() == {set(value.keys())!r}")

    def NN_MODULE_PARAM_NAMES(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        keys = {k for k, v in value.named_parameters()}
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"{{k for k, v in {ref}.named_parameters()}} == {keys!r}")

    def ODICT_KEYS(self, guard):
        """OrderedDict keys match"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        self.code.append(f"___check_type_id({ref}, {self.id_ref(type(value))})")
        self.code.append(f"str({ref}.keys()) == {str(value.keys())!r}")

    def OBJECT_MUTATION(self, guard: Guard):
        mutation_guard.watch(self.get(guard.name), self.guarded_code)

    def GRAD_MODE(self, guard: Guard):
        """Guard on the current value of torch.is_grad_enabled()"""
        assert guard.name == ""
        assert guard.source is GuardSource.GLOBAL
        if torch.is_grad_enabled():
            self.code.append("___is_grad_enabled()")
        else:
            self.code.append("not ___is_grad_enabled()")

    def TENSOR_MATCH(self, guard: Guard):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            self.tensor_check_names.append(self.arg_ref(guard))
            self.tensor_check_examples.append(self.get(guard.name))


class GuardedCode:
    def __init__(
        self,
        code: types.CodeType,
        guards: Optional[Set[Guard]] = None,
        f_locals: Optional[Dict] = None,
        f_globals: Optional[Dict] = None,
    ):
        self.code = code
        self.valid = True
        self._weakrefs = []
        self._seen_ids = set()

        local_builder = GuardBuilder(self.id_ref, f_locals, self, renames=True)
        global_builder = GuardBuilder(self.id_ref, f_globals, self, renames=False)
        for guard in sorted(guards or [], key=Guard.sort_key):
            if not config.guard_nn_modules and guard.is_nn_module():
                continue
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(local_builder, global_builder)
        self._seen_ids.clear()

    def compile_check_fn(self, local_builder, global_builder):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        args = [a for a in local_builder.scope.keys() if a == "___implicit0"]
        args += [a for a in local_builder.argnames if a != "___implicit0"]
        args += ["**___kwargs_ignored"]
        args = ",".join(args)

        code_parts = (
            ["___guarded_code.valid"] + local_builder.code + global_builder.code
        )
        # TODO(whc) maybe only the 'check_tensors' one is ambiguous? if so we can be less general..
        verbose_code_parts = (
            ["___guarded_code.valid"] + local_builder.code + global_builder.code
        )

        tensor_check_names = (
            local_builder.tensor_check_names + global_builder.tensor_check_names
        )
        check_tensors_fn = None
        check_tensors_verbose_fn = None
        if tensor_check_names:
            tensor_check_examples = (
                local_builder.tensor_check_examples
                + global_builder.tensor_check_examples
            )
            tensor_guards = TensorGuards(
                *tensor_check_examples, dynamic_shapes=config.dynamic_shapes
            )
            check_tensors_fn = tensor_guards.check
            check_tensors_verbose_fn = tensor_guards.check_verbose
            code_parts.append(f"___check_tensors({', '.join(tensor_check_names)})")
            verbose_args = ", ".join(
                tensor_check_names + ["tensor_check_names=tensor_check_names"]
            )
            verbose_code_parts.append(f"___check_tensors_verbose({verbose_args})")

        code = " and ".join(unique(code_parts))

        closure_vars = collections.OrderedDict(
            [
                ("___guarded_code", self),
                ("___check_tensors", check_tensors_fn),
                ("___check_tensors_verbose", check_tensors_verbose_fn),
                ("tensor_check_names", tensor_check_names),
            ]
        )
        closure_vars.update(CLOSURE_VARS)
        py_code = textwrap.dedent(
            f"""
            def ___make_guard_fn({','.join(closure_vars.keys())}):
                return lambda {args}: {code}
            """
        )
        if os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1":
            print("GUARDS", code)
        set_guard_fail_hook(guard_fail_hook)
        out = dict()
        exec(py_code, global_builder.scope, out)
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        guard_fn.closure_vars = closure_vars
        # TODO(whc) maybe '.code_parts' was only kept around for the guard callback? so we don't need both
        guard_fn.code_parts = code_parts
        guard_fn.verbose_code_parts = verbose_code_parts
        guard_fn.global_scope = global_builder.scope
        return guard_fn

    def invalidate(self, ref):
        # A weakref is no longer valid, self.check_fn should return false
        self.valid = False

    def id_ref(self, obj):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._seen_ids:
                self._weakrefs.append(weakref.ref(obj, self.invalidate))
                self._seen_ids.add(id(obj))
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)


def guard_fail_hook(
    guard_fn: Callable, code: types.CodeType, f_locals: Dict[str, Any], last: bool
):
    """
    called whenever a guard fails.
    """
    if not last:
        return
    scope = {rename_implicit(k): v for k, v in f_locals.items()}
    scope.update(guard_fn.closure_vars)
    reasons = []
    for part in guard_fn.verbose_code_parts:
        fail_reason = eval(part, guard_fn.global_scope, scope)
        # TODO(whc) hacky for now as not every 'part' in guard_fn.verbose_code_parts
        # is updated to return a string explaining the failure.
        if isinstance(fail_reason, str):
            reasons.append(fail_reason)
            break
        elif isinstance(fail_reason, bool) and not fail_reason:
            reasons.append(part)
            break
    guard_failures[orig_code_map[code]].append(reasons)


def guard_error_hook(
    guard_fn: Callable, code: types.CodeType, f_locals: Dict[str, Any], last: bool
):
    print(
        f"ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}"
    )
    print(" ", " and\n  ".join(guard_fn.code_parts))


set_guard_error_hook(guard_error_hook)


def unique(seq):
    seen = set()
    for x in seq:
        if x not in seen:
            yield x
            seen.add(x)
