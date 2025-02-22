from typing import Dict
from typing import List

import torch

from .. import variables
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..source import GetItemSource
from ..utils import namedtuple_fields
from .base import MutableLocal
from .base import VariableTracker
from .constant import ConstantVariable


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for(obj):
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
        }[obj]

    def __init__(self, items: List[VariableTracker], **kwargs):
        super(BaseListVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
        index = arg.as_python_constant()
        if isinstance(index, slice):
            if self.source is not None:
                return self.clone(
                    items=self.items[index],
                    source=GetItemSource(self.source, index),
                    mutable_local=None,
                ).add_options(arg, self)
            else:
                return self.clone(
                    items=self.items[index], mutable_local=None
                ).add_options(arg, self)
        else:
            assert isinstance(index, int)
            return self.items[index].add_options(arg, self)

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            return self.getitem_const(args[0])
        elif name == "__add__":
            assert not kwargs and len(args) == 1
            return type(self)(self.items + args[0].items, **options)
        elif (
            name == "__contains__"
            and len(args) == 1
            and args[0].is_python_constant()
            and all(x.is_python_constant() for x in self.items)
        ):
            assert not kwargs
            search = args[0].as_python_constant()
            result = any(x.as_python_constant() == search for x in self.items)
            return variables.ConstantVariable(result, **options)

        return super(BaseListVariable, self).call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    def __init__(self, value, items=None, guards=None, **kwargs):
        if items is None:
            items = [variables.ConstantVariable(x, guards=guards) for x in value]
        super().__init__(items, guards=guards, **kwargs)
        self.value = value

    def python_type(self):
        return range

    def as_python_constant(self):
        return self.value

    def reconstruct(self, codegen):
        assert "range" not in codegen.tx.f_globals
        range_fn = codegen.create_load_global("range", add=True)
        if self.value.step == 1:
            if self.value.start == 0:
                return [
                    range_fn,
                    codegen.create_load_const(self.value.stop),
                    create_instruction("CALL_FUNCTION", 1),
                ]
            return [
                range_fn,
                codegen.create_load_const(self.value.start),
                codegen.create_load_const(self.value.stop),
                create_instruction("CALL_FUNCTION", 2),
            ]
        return [
            range_fn,
            codegen.create_load_const(self.value.start),
            codegen.create_load_const(self.value.stop),
            codegen.create_load_const(self.value.step),
            create_instruction("CALL_FUNCTION", 3),
        ]


class ListVariable(BaseListVariable):
    def python_type(self):
        return list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_LIST", len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "append" and self.mutable_local:
            assert not kwargs
            (arg,) = args
            tx.replace_all(
                self,
                ListVariable(self.items + [arg], **options),
            )
            return ConstantVariable(None)
        elif (
            name in ("extend", "__iadd__")
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs
            (arg,) = args
            return tx.replace_all(
                self,
                ListVariable(
                    list(self.items) + list(arg.unpack_var_sequence(tx)),
                    **options,
                ),
            )
        elif name == "insert" and self.mutable_local:
            assert not kwargs
            idx, value = args
            items = list(self.items)
            items.insert(idx.as_python_constant(), value)
            return tx.replace_all(
                self,
                ListVariable(items, **options),
            )
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(
                self,
                ListVariable(items, **options),
            )
            return result
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args
            return tx.replace_all(
                self,
                ListVariable([], **options),
            )
        elif (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            items = list(self.items)
            items[key.as_python_constant()] = value
            result = ListVariable(items, **options)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_TUPLE", len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], TupleVariable)
        ):
            assert not kwargs
            return TupleVariable(self.items + args[0].items, **options)
        elif (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], variables.ConstantVariable)
        ):
            assert not kwargs
            return TupleVariable(
                self.items + list(args[0].unpack_var_sequence(self)), **options
            )
        return super().call_method(tx, name, args, kwargs)


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    def python_type(self):
        return torch.Size

    def reconstruct(self, codegen):
        codegen.load_import_from("torch", "Size")
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]
        return build_torch_size


class NamedTupleVariable(TupleVariable):
    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]

    def var_getattr(self, tx, name):
        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            unimplemented(f"NamedTupleVariable.{name}")
        return self.items[fields.index(name)].add_options(self)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        options = VariableTracker.propagate(self)
        fields = namedtuple_fields(self.tuple_cls)
        return variables.ConstantVariable(name in fields, **options)


class SliceVariable(BaseListVariable):
    def __init__(self, items, **kwargs):
        start, stop, step = [variables.ConstantVariable(None)] * 3
        if len(items) == 1:
            (stop,) = items
        elif len(items) == 2:
            start, stop = items
        elif len(items) == 3:
            start, stop, step = items
        else:
            assert False
        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_SLICE", len(self.items))]

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)].add_options(self)


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, **kwargs):
        super(ListIteratorVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def next_variables(self):
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        return self.items[self.index].add_options(self), ListIteratorVariable(
            self.items,
            self.index + 1,
            mutable_local=MutableLocal(),
            **VariableTracker.propagate([self]),
        )

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError()
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items[self.index :]]

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index :]
        codegen.foreach(remaining_items)
        return [
            create_instruction("BUILD_TUPLE", len(remaining_items)),
            create_instruction("GET_ITER"),
        ]
