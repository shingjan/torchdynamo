import collections
import contextlib
import dataclasses
import functools
import itertools
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import torch

from . import config
from . import dependencies
from . import ir
from .dependencies import StarDep
from .sizevars import SimplifyIndexing
from .virtualized import V


def cmp(a, b):
    return int(a > b) - int(a < b)


class OutputNode:
    def __init__(self, dep):
        self.unmet_dependencies = {dep}
        self.inverse_users = []

    def is_reduction(self):
        return False

    def get_alias_names(self):
        return ()

    def get_name(self):
        return "OUTPUT"

    __repr__ = get_name


class BaseSchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.Buffer):
        self.scheduler = scheduler
        self.node = node
        self.users = None
        self.inverse_users = []
        self.set_read_writes(node.get_read_writes())

    def __repr__(self):
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def update_mutated_names(self, renames: Dict[str, str]):
        self.set_read_writes(self.read_writes.rename(renames))

    def add_mutation_dep(self, name):
        self.set_read_writes(self.read_writes.with_read(name))

    def set_users(self, users: List["NodeUser"]):
        # deduplicate
        result = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = NodeUser(
                    use.node, result[id(use.node)].can_inplace and use.can_inplace
                )
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self):
        return self.node.get_alias_names()

    def get_mutations(self):
        return self.node.get_mutation_names()

    def set_read_writes(self, rw):
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def get_name(self):
        return self.node.get_name()

    def get_device(self):
        return self.node.get_device()

    def is_reduction(self):
        return False

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def allocate(self):
        if self.node.should_allocate():
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
            name = use.get_name()
            if name not in self.scheduler.available_buffer_names:
                return False
        return True

    def get_priority(self):
        """Controls the order this node will be executed in, higher runs first"""
        raise NotImplementedError()


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def can_remove_buffer(self, **kwargs):
        return False

    def run(self, codegen_extern_call):
        self.allocate()
        self.scheduler.run_count += 1
        self.scheduler.pending_buffer_names.add(self.get_name())
        self.scheduler.kernels.append(self.node)
        codegen_extern_call(self.node)

    def get_priority(self):
        return 100


class NopKernelSchedulerNode(BaseSchedulerNode):
    def can_remove_buffer(self, **kwargs):
        return False

    def run(self):
        self.allocate()
        self.scheduler.run_count += 1
        self.scheduler.pending_buffer_names.add(self.get_name())

    def get_priority(self):
        return 200


def pick_loop_order(stride_lengths, sizes):
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a, b):
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        a_first = np.logical_or(
            stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]
        ).all()
        b_first = np.logical_or(
            stride_lengths[:, a] == 0, stride_lengths[:, a] > stride_lengths[:, b]
        ).all()

        if a_first and not b_first:
            return -1
        if b_first and not a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(stride_lengths.shape[1])))
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


class SchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group_fn):
        super().__init__(scheduler, node)
        (
            self._sizes,
            self._body,
        ) = node.simplify_reorder_and_tile()

        self.group = (node.get_device(), group_fn(self._sizes))

        self.set_read_writes(dependencies.extract_read_writes(self._body, *self._sizes))

    def can_remove_buffer(self, check_group):
        if (
            self.is_reduction()
            and len(self.users) == 1
            and isinstance(self.users[0].node, SchedulerNode)
            and len(self.users[0].node.unmet_dependencies) == 1
        ):
            user = self.users[0].node
            if not check_group(user):
                return False
            dep = next(iter(user.unmet_dependencies))
            writes = self.read_writes.writes
            if self._sizes[-1] != 1:
                writes = set(writes)
                writes.update(
                    [w.broadcast_extend_sizes(self._sizes[-1]) for w in writes]
                )
            # this will get fused into us, so we don't need to keep the buffer
            return not user.is_reduction() and dep in writes
        return False

    def mark_fusable(self, broadcast_after_reduce=False):
        self.scheduler.fusable_deps.update(self.read_writes.writes)
        if broadcast_after_reduce and self.is_reduction():
            self.scheduler.fusable_deps.update(
                w.broadcast_extend_sizes(self._sizes[-1])
                for w in self.read_writes.writes
            )

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        return bool(self.node.data.get_reduction_type())

    def allocate(self):
        if (
            not self.node.should_allocate()
            or self.node.get_alias_names()
            or self.node.get_mutation_names()
        ):
            return super().allocate()

        if config.inplace_buffers:
            for read in self.read_writes.reads:
                input_node: BaseSchedulerNode = self.scheduler.name_to_node.get(
                    read.name
                )
                if input_node and V.graph.wrapper_code.can_reuse(input_node):
                    remaining_uses = [
                        x
                        for x in input_node.users
                        if x.node.get_name()
                        not in self.scheduler.available_buffer_names
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                    ):
                        V.graph.wrapper_code.codegen_inplace_reuse(
                            input_node.node, self.node
                        )
                        V.kernel.args.make_inplace(
                            input_node.get_name(), self.get_name()
                        )
                        return
        super().allocate()

    def run(self, *index_vars):
        self.allocate()
        self.scheduler.run_count += 1
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        with V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)):
            self._body(*index_vars)
        self.scheduler.pending_buffer_names.add(self.get_name())

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.node.get_alias_names():
            return False
        if len(self.read_writes.writes) == 1 and hasattr(read_dep, "index"):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    def get_priority(self):
        if self.is_reduction():
            return len(self.group)
        else:
            return len(self.group) - 1


@dataclasses.dataclass
class SchedulerNodeBox:
    """Allow us to invalidate a blocked node"""

    value: SchedulerNode

    def __bool__(self):
        return self.value is not None

    def pop(self) -> SchedulerNode:
        assert self
        val = self.value
        self.value = None
        return val

    def peek(self) -> SchedulerNode:
        return self.value


class BlockedNodes:
    def __init__(self):
        super().__init__()
        self.name_to_nodes = collections.defaultdict(list)
        self.dep_to_nodes = collections.defaultdict(list)

    def add(self, node: SchedulerNode):
        box = SchedulerNodeBox(node)
        for dep in node.unmet_dependencies:
            self.name_to_nodes[dep.name].append(box)
            self.dep_to_nodes[dep].append(box)

    def pop_name(self, name):
        return [x.pop() for x in self.name_to_nodes.pop(name, []) if x]

    def pop_fusable(self, deps, group):
        assert isinstance(deps, set)
        result = []
        for dep in deps:
            self.dep_to_nodes[dep] = [x for x in self.dep_to_nodes[dep] if x]
            for box in self.dep_to_nodes[dep]:
                if (
                    len(box.peek().unmet_dependencies - deps) == 0
                    and box.peek().group == group
                ):
                    result.append(box.pop())
        return result


@dataclasses.dataclass
class NodeUser:
    node: BaseSchedulerNode
    can_inplace: bool = False

    def get_name(self):
        return self.node.get_name()


class Scheduler:
    def __init__(self, nodes):
        super(Scheduler, self).__init__()
        self.backends = {}
        self.current_device = None
        # runable_groups maps node group to priority
        # we use self.runable_groups.most_common() to implement a priority queue
        self.runable_groups = collections.Counter()
        # runable_nodes  maps node group to nodes
        self.runable_nodes: Dict[Any, SchedulerNode] = collections.defaultdict(list)
        self.runable_extern_kernels = collections.deque()
        self.blocked_nodes = BlockedNodes()
        self.run_count = 0
        self.nodes = []
        self.kernels = []
        self.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }
        self.pending_buffer_names = set()
        self.check_can_free = set()
        self.fusable_deps = set()
        for node in nodes:
            if node.is_no_op():
                self.nodes.append(NopKernelSchedulerNode(self, node))
            elif isinstance(node, ir.ComputedBuffer):
                group_fn = self.get_backend(node.get_device()).group_fn
                self.nodes.append(SchedulerNode(self, node, group_fn))
            elif isinstance(node, ir.ExternKernel):
                self.nodes.append(ExternKernelSchedulerNode(self, node))
            else:
                assert False, node
        self.name_to_node = {node.get_name(): node for node in self.nodes}

        # we handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        self.mutation_real_name = {}
        # mutation_real_name: maps back to the original name for codegen
        self.mutation_renames = {}

        self.compute_users()
        self.dead_node_elimination()
        self.enqueue(self.nodes)

    def compute_users(self):
        name_to_users = collections.defaultdict(list)

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node1 in self.nodes:
            node1_name = node1.get_name()
            for node2_name in node1.get_aliases():
                if node1_name in name_to_users and node2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[node1_name]
                    list2 = name_to_users[node2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if name_to_users[key] is list1 or name_to_users[key] is list2:
                            name_to_users[key] = combined
                elif node1_name in name_to_users:
                    name_to_users[node2_name] = name_to_users[node1_name]
                else:
                    name_to_users[node1_name] = name_to_users[node2_name]

        def rename(n):
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def add_user(used_by_name, user_node, can_inplace=False):
            name_to_users[rename(used_by_name)].append(NodeUser(user_node, can_inplace))

        for node in self.nodes:
            # a node will mutate either 0 or 1 buffers
            for alt_name in node.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_mutation_dep(alt_name)
                for other_node in name_to_users[alt_name]:
                    # this node must run after all prior readers
                    other_name = rename(other_node.get_name())
                    if other_name != node.get_name():
                        node.add_mutation_dep(other_name)
                        add_user(other_name, node)

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for alt_name in node.get_mutations():
                self.mutation_renames[rename(alt_name)] = node.get_name()
                self.mutation_renames[alt_name] = node.get_name()
                self.mutation_real_name[node.get_name()] = self.mutation_real_name.get(
                    alt_name, alt_name
                )

        # make sure outputs aren't dead-code-eliminated
        for node in V.graph.graph_outputs:
            name = node.get_name()
            add_user(node.get_name(), OutputNode(StarDep(name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)

        # copy users information onto the nodes
        for node in self.nodes:
            node.set_users(name_to_users[node.get_name()])

        # populate inverse_users
        for node in self.nodes:
            for user in node.users:
                user.node.inverse_users.append(node)

    def dead_node_elimination(self):
        updated_nodes = []
        for node in self.nodes:
            if node.users:
                updated_nodes.append(node)
            else:
                # dead code
                V.graph.removed_buffers.add(node.get_name())
        self.nodes = updated_nodes

    def maybe_remove_buffer(self, node: SchedulerNode, check_group: Callable):
        name = node.get_name()
        if name in self.mutation_renames:
            return
        if node.can_remove_buffer(check_group=check_group):
            print("REMOVING", name)
            V.graph.removed_buffers.add(name)

    def enqueue(self, node):
        if isinstance(node, (tuple, list)):
            for n in node:
                self.enqueue(n)
            return

        assert isinstance(node, BaseSchedulerNode)
        if node.unmet_dependencies:
            self.blocked_nodes.add(node)
        else:
            if isinstance(node, ExternKernelSchedulerNode):
                self.runable_extern_kernels.append(node)
            elif isinstance(node, NopKernelSchedulerNode):
                node.run()  # just schedule nop kernels eagerly
            else:
                self.runable_nodes[node.group].append(node)
                old_priority, old_count = self.runable_groups.get(node.group, (0, 0))
                self.runable_groups[node.group] = (
                    max(old_priority, node.get_priority()),
                    old_count + 1,
                )

    def barrier(self):
        """
        Mark all pending_buffer_names as available and enqueue any nodes
        that became runable.
        """
        while self.pending_buffer_names:
            self.available_buffer_names.update(self.pending_buffer_names)
            nodes_to_add = []
            for name in self.pending_buffer_names:
                self.check_can_free.update(self.pending_buffer_names)
                for node in self.blocked_nodes.pop_name(name):
                    node.prune_deps()
                    nodes_to_add.append(node)
            self.pending_buffer_names.clear()
            self.enqueue(nodes_to_add)

    def maybe_free_buffers(self):
        # perhaps there are some unused buffers we can free
        for done_name in self.check_can_free:
            done_node = self.name_to_node[done_name]
            for node in done_node.inverse_users:
                name = node.get_name()
                if node.can_free() and name:
                    if name in self.mutation_renames:
                        continue
                    if name in self.mutation_real_name:
                        name = self.mutation_real_name[name]
                        if name in self.name_to_node:
                            V.graph.wrapper_code.codegen_free(
                                self.name_to_node[name].node
                            )
                    else:
                        V.graph.wrapper_code.codegen_free(node.node)
        self.check_can_free.clear()

    def kernel(self, kernel):
        self.fusable_deps.clear()
        self.kernels.append(kernel)

        @contextlib.contextmanager
        def ctx():
            with kernel:
                yield kernel

        return ctx()

    def iter_runable_groups(self):
        while self.runable_groups or self.runable_extern_kernels:
            if self.runable_extern_kernels:
                self.runable_extern_kernels.popleft().run(self.codegen_extern_call)
            else:
                group, priority = self.runable_groups.most_common(1)[0]
                del self.runable_groups[group]
                yield group
        assert not self.runable_nodes
        assert len(self.nodes) == self.run_count

    def iter_fixed_point(self):
        """
        Keep yielding until self.run_count converges
        """
        prior_run_count = -1
        while prior_run_count != self.run_count:
            prior_run_count = self.run_count
            yield

    def pop_group(self, group_without_device):
        group = (self.current_device, tuple(group_without_device))
        while group in self.runable_nodes:
            if group in self.runable_groups:
                del self.runable_groups[group]
            yield from self.runable_nodes.pop(group)
        if self.fusable_deps:
            fusable = True
            while fusable:
                fusable = self.blocked_nodes.pop_fusable(self.fusable_deps, group)
                yield from fusable

    def flush(self):
        for backend in self.backends.values():
            backend.flush()

    def codegen_extern_call(self, node: ir.ExternKernel):
        assert isinstance(node, ir.ExternKernel)
        self.flush()
        node.codegen(V.graph.wrapper_code)
        self.barrier()
        self.maybe_free_buffers()

    def create_backend(self, device: torch.device):
        V.graph.device_types.add(device.type)
        if device.type == "cpu":
            from .codegen.cpp import CppScheduling

            return CppScheduling(self)
        else:
            from .codegen.triton import TritonScheduling

            return TritonScheduling(self)

    def get_backend(self, device: torch.device):
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def codegen(self):
        for device, group in self.iter_runable_groups():
            if device != self.current_device:
                self.flush()
                self.current_device = device
            self.get_backend(device).codegen(*group)
        self.flush()
