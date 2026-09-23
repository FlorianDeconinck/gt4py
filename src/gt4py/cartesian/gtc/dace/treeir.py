# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cmp_to_key, lru_cache
from types import TracebackType
from typing import Any, Generator, TypeAlias

from dace import Memlet, data, dtypes, nodes
from sympy import S, Symbol, parse_expr

from gt4py import eve
from gt4py.cartesian.gtc import common, definitions
from gt4py.cartesian.gtc.dace import utils


SymbolDict: TypeAlias = dict[str, dtypes.typeclass]


@dataclass
class Context:
    root: TreeRoot
    current_scope: TreeScope

    field_extents: dict[str, definitions.Extent]  # field_name -> Extent
    block_extents: dict[int, definitions.Extent]  # id(horizontal execution) -> Extent


class ContextPushPop:
    """Append the node to the scope, then push/pop the scope."""

    def __init__(self, ctx: Context, node: TreeScope) -> None:
        self._ctx = ctx
        self._parent_scope = ctx.current_scope
        self._node = node

    def __enter__(self) -> None:
        self._node.parent = self._parent_scope
        self._parent_scope.children.append(self._node)
        self._ctx.current_scope = self._node

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._ctx.current_scope = self._parent_scope


class Axis(eve.StrEnum):
    I = "I"  # noqa: E741 [ambiguous-variable-name]
    J = "J"
    K = "K"

    def domain_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef(f"__{self.upper()}")

    def iteration_symbol(self) -> eve.SymbolRef:
        return eve.SymbolRef(f"__{self.lower()}")

    @staticmethod
    def dims_3d() -> Generator[Axis, None, None]:
        yield from [Axis.I, Axis.J, Axis.K]

    @staticmethod
    def dims_horizontal() -> Generator[Axis, None, None]:
        yield from [Axis.I, Axis.J]

    def to_idx(self) -> int:
        return [Axis.I, Axis.J, Axis.K].index(self)

    def domain_dace_symbol(self):
        return utils.get_dace_symbol(self.domain_symbol())

    def iteration_dace_symbol(self):
        return utils.get_dace_symbol(self.iteration_symbol())


__I_sym = Symbol("__I", integer=True, positive=True)
__J_sym = Symbol("__J", integer=True, positive=True)
_HORIZONTAL_AXIS_SYMBOLS = {"__I": __I_sym, "__J": __J_sym}


@lru_cache(maxsize=None)
def _compare_symbolic(a: Any, b: Any) -> int:
    """Evaluate the difference while forcing __I -> ∞ and __J -> ∞"""

    sign = (a - b).limit(__I_sym, S.Infinity).limit(__J_sym, S.Infinity)

    if sign > 0:
        return 1
    elif sign < 0:
        return -1
    return 0


@lru_cache(maxsize=None)
def _is_strictly_less_symbolic(a: Any, b: Any):
    """Returns True if a < b as __I -> ∞"""
    if a == b:
        return False
    return (a - b).limit(__I_sym, S.Infinity).limit(__J_sym, S.Infinity) < 0


class Bounds(eve.Node):
    start: str
    end: str

    def __eq__(self, other):
        if not isinstance(other, Bounds):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    def as_tuple(self) -> tuple[str, str]:
        return (self.start, self.end)

    def do_bounds_overlap(self, bound_B: "Bounds") -> bool:
        """Solve for half-open intervals [a1, b1) and [a2, b2) which overlap if a1 < b2 AND a2 < b1"""

        start_A = parse_expr(self.start, local_dict=_HORIZONTAL_AXIS_SYMBOLS)
        end_A = parse_expr(self.end, local_dict=_HORIZONTAL_AXIS_SYMBOLS)
        start_B = parse_expr(bound_B.start, local_dict=_HORIZONTAL_AXIS_SYMBOLS)
        end_B = parse_expr(bound_B.end, local_dict=_HORIZONTAL_AXIS_SYMBOLS)

        return _is_strictly_less_symbolic(start_A, end_B) and _is_strictly_less_symbolic(
            start_B, end_A
        )

    @classmethod
    def get_disjoint_intervals(cls, *bounds) -> list["Bounds"]:
        """Return disjointed intervals of all the given intervals"""

        # Make sympy expression, turning axis to symbols
        parsed_bounds = [
            (
                parse_expr(b.start, local_dict=_HORIZONTAL_AXIS_SYMBOLS),
                parse_expr(b.end, local_dict=_HORIZONTAL_AXIS_SYMBOLS),
            )
            for b in bounds
        ]

        # Collect and sort all distinct boundaries
        endpoints = sorted(
            [pt for b in parsed_bounds for pt in b], key=cmp_to_key(_compare_symbolic)
        )

        # Create disjoint adjacent intervals from the ordered endpoints
        disjoint = []
        for start, end in itertools.pairwise(endpoints):
            if start != end:
                disjoint.append(cls(start=str(start), end=str(end)))

        return disjoint


class TreeNode(eve.Node):
    parent: TreeScope | None


class TreeScope(TreeNode):
    children: list[TreeScope | TreeNode]

    def scope(self, ctx: Context) -> ContextPushPop:
        return ContextPushPop(ctx, self)


class Tasklet(TreeNode):
    tasklet: nodes.Tasklet

    inputs: dict[str, Memlet]
    """Mapping tasklet.in_connectors to Memlets"""
    outputs: dict[str, Memlet]
    """Mapping tasklet.out_connectors to Memlets"""


class IfElse(TreeScope):
    # This should become an if/else, someday, so I am naming it if/else in hope
    # to see it before my bodily demise
    if_condition_code: str
    """Condition as ScheduleTree worthy code"""


class While(TreeScope):
    condition_code: str
    """Condition as ScheduleTree worthy code"""


class HorizontalLoop(TreeScope):
    bounds_i: Bounds
    bounds_j: Bounds

    schedule: dtypes.ScheduleType
    groups: list[Any]


class HorizontalRestriction(TreeScope):
    """HorizontalRestriction is only a temporary node in the IR.
    See visit_HorizontalExection to see how it is immediately transformed to an
    HorizontalLoop during oir->treeir transformation"""

    oir_hr: common.HorizontalRestriction
    groups: list[Any]


class SequentialVerticalLoop(TreeScope):
    iteration_variable: eve.SymbolRef
    bounds_k: Bounds
    loop_order: common.LoopOrder


class ParallelVerticalLoop(TreeScope):
    iteration_variable: eve.SymbolRef
    bounds_k: Bounds
    schedule: dtypes.ScheduleType


class TreeRoot(TreeScope):
    name: str

    containers: dict[str, data.Data]
    """Mapping field/scalar names to data descriptors."""

    dimensions: dict[str, tuple[bool, bool, bool]]
    """Mapping field names to shape-axis."""

    shift: dict[str, dict[Axis, int]]
    """Mapping field names to dict[axis] -> shift."""

    symbols: SymbolDict
    """Mapping between type and symbol name."""
