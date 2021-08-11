# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import textwrap
from typing import Any, Collection, Tuple, Union

from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator
from gt4py.definitions import Extent
from gtc import common
from gtc.passes.gtir_legacy_extents import FIELD_EXT_T
from gtc.python import npir


__all__ = ["NpirGen"]


def op_delta_from_int(value: int) -> Tuple[str, str]:
    operator = ""
    delta = str(value)
    if value == 0:
        delta = ""
    elif value < 0:
        operator = " - "
        delta = str(abs(value))
    else:
        operator = " + "
    return operator, delta


ORIGIN_CORRECTED_VIEW_CLASS = textwrap.dedent(
    """\
    class ShimmedView:
        def __init__(self, field, offsets):
            self.field = field
            self.offsets = offsets

        def shim_key(self, key):
            new_args = []
            if not isinstance(key, tuple):
                key = (key, )
            for dim, idx in enumerate(key):
                if self.offsets[dim] == 0:
                    new_args.append(idx)
                elif isinstance(idx, slice):
                    start = 0 if idx.start is None else idx.start
                    stop = -1 if idx.stop is None else idx.stop
                    new_args.append(
                        slice(start + self.offsets[dim], stop + self.offsets[dim], idx.step)
                    )
                else:
                    new_args.append(idx + self.offsets[dim])
            if isinstance(new_args[-1], np.ndarray):
                new_args[0:2] = np.broadcast_arrays(
                    np.arange(new_args[0].start, new_args[0].stop)[:, None, None],
                    np.arange(new_args[1].start, new_args[1].stop)[None, :, None],
                )
            return tuple(new_args)

        def __getitem__(self, key):
            return self.field.__getitem__(self.shim_key(key))

        def __setitem__(self, key, value):
            return self.field.__setitem__(self.shim_key(key), value)
    """
)


class NpirGen(TemplatedGenerator):
    def visit_DataType(self, node: common.DataType, **kwargs: Any) -> Union[str, Collection[str]]:
        return f"np.{node.name.lower()}"

    def visit_BuiltInLiteral(self, node: common.BuiltInLiteral, **kwargs) -> str:
        if node is common.BuiltInLiteral.TRUE:
            return "True"
        elif node is common.BuiltInLiteral.FALSE:
            return "False"
        else:
            return self.generic_visit(node, **kwargs)

    Literal = FormatTemplate("{dtype}({value})")

    BroadCast = FormatTemplate("{expr}")

    Cast = FormatTemplate("np.array({expr}, dtype={dtype})")

    def visit_NumericalOffset(
        self, node: npir.NumericalOffset, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        operator, delta = op_delta_from_int(node.value)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    NumericalOffset = FormatTemplate("{op}{delta}")

    def visit_VariableOffset(self, node: npir.VariableOffset, **kwargs: Any) -> str:
        return self.visit(node.value)

    def visit_AxisOffset(self, node: npir.AxisOffset, **kwargs: Any) -> Union[str, Collection[str]]:
        offset = self.visit(node.offset)
        axis_name = self.visit(node.axis_name)
        lpar, rpar = "()" if offset else ("", "")
        if isinstance(node.offset, npir.VariableOffset):
            rendered = f"{axis_name.lower()}_ + np.asarray({offset}[:])"
        else:
            variant = self.AxisOffset_parallel if node.parallel else self.AxisOffset_serial
            rendered = variant.render(lpar=lpar, rpar=rpar, axis_name=axis_name, offset=offset)
        return self.generic_visit(node, parallel_or_serial_variant=rendered, **kwargs)

    AxisOffset_parallel = JinjaTemplate(
        "{{ lpar }}{{ axis_name | lower }}{{ offset }}{{ rpar }}:{{ lpar }}{{ axis_name | upper }}{{ offset }}{{ rpar }}"
    )

    AxisOffset_serial = JinjaTemplate(
        "{{ lpar }}{{ axis_name | lower }}_{{ offset }}{{ rpar }}:({{ axis_name | lower }}_{{ offset}} + 1)"
    )

    AxisOffset = FormatTemplate("{parallel_or_serial_variant}")

    def visit_FieldDecl(self, node: npir.FieldDecl, **kwargs) -> Union[str, Collection[str]]:
        if all(node.dimensions):
            return ""
        shape_idx = iter(range(3))
        origin_idx = iter(range(3))
        shape = ", ".join(
            [f"{node.name}.shape[{next(shape_idx)}]" if dim else "1" for dim in node.dimensions]
        )
        origin = ", ".join(
            [
                f"_origin_['{node.name}'][{next(origin_idx)}]" if dim else "0"
                for dim in node.dimensions
            ]
        )
        return self.generic_visit(node, shape=shape, origin=origin, **kwargs)

    FieldDecl = FormatTemplate(
        "{name} = np.reshape({name}, ({shape}))\n_origin_['{name}'] = [{origin}]"
    )

    def visit_FieldSlice(
        self, node: npir.FieldSlice, mask_acc="", *, is_serial=False, is_rhs=False, **kwargs: Any
    ) -> Union[str, Collection[str]]:

        offset = [node.i_offset, node.j_offset, node.k_offset]

        offset_str = ", ".join(self.visit(off, **kwargs) if off else ":" for off in offset)

        if is_rhs and mask_acc and any(off is None for off in offset):
            k_size = 1 if is_serial else "K - k"
            arr_expr = f"np.broadcast_to({node.name}_[{offset_str}], (I - i, J - j, {k_size}))"
        else:
            arr_expr = f"{node.name}_[{offset_str}]"

        return self.generic_visit(node, arr_expr=arr_expr, mask_acc=mask_acc, **kwargs)

    FieldSlice = FormatTemplate("{arr_expr}{mask_acc}")

    def visit_EmptyTemp(
        self, node: npir.EmptyTemp, *, temp_name: str, **kwargs
    ) -> Union[str, Collection[str]]:
        shape = "_domain_"
        origin = [0, 0, 0]
        if extents := kwargs.get("field_extents", {}).get(temp_name):
            boundary = extents.to_boundary()
            i_total = sum(boundary[0])
            j_total = sum(boundary[1])
            shape = f"(_dI_ + {i_total}, _dJ_ + {j_total}, _dK_)"
            origin[:2] = boundary[0][0], boundary[1][0]
        return self.generic_visit(node, shape=shape, origin=origin, **kwargs)

    EmptyTemp = FormatTemplate("ShimmedView(np.zeros({shape}, dtype={dtype}), {origin})")

    NamedScalar = FormatTemplate("{name}")

    VectorTemp = FormatTemplate("{name}_")

    def visit_MaskBlock(self, node: npir.MaskBlock, **kwargs) -> Union[str, Collection[str]]:
        if isinstance(node.mask, npir.FieldSlice):
            mask_def = ""
        elif isinstance(node.mask, npir.BroadCast):
            mask_name = node.mask_name
            mask = self.visit(node.mask)
            mask_def = f"{mask_name}_ = np.full((I - i, J - j, K - k), {mask})\n"
        else:
            mask_name = node.mask_name
            mask = self.visit(node.mask)
            mask_def = f"{mask_name}_ = {mask}\n"
        return self.generic_visit(node, mask_def=mask_def, **kwargs)

    MaskBlock = JinjaTemplate(
        textwrap.dedent(
            """\
                {{ mask_def }}{% for stmt in body %}{{ stmt }}
                {% endfor %}
            """
        )
    )

    def visit_VectorAssign(
        self, node: npir.VectorAssign, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        mask_acc = ""
        if node.mask:
            mask_acc = f"[{self.visit(node.mask, **kwargs)}]"
        if isinstance(node.right, npir.EmptyTemp):
            kwargs["temp_name"] = node.left.name
        right = self.visit(node.right, mask_acc=mask_acc, is_rhs=True, **kwargs)
        left = self.visit(node.left, mask_acc=mask_acc, is_rhs=False, **kwargs)
        return f"{left} = {right}"

    VectorArithmetic = FormatTemplate("({left} {op} {right})")

    VectorLogic = FormatTemplate("np.bitwise_{op}({left}, {right})")

    def visit_UnaryOperator(
        self, node: common.UnaryOperator, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node is common.UnaryOperator.NOT:
            return "np.bitwise_not"
        return self.generic_visit(node, **kwargs)

    VectorUnaryOp = FormatTemplate("({op}({expr}))")

    VectorTernaryOp = FormatTemplate("np.where({cond}, {true_expr}, {false_expr})")

    def visit_LevelMarker(
        self, node: common.LevelMarker, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node: common.AxisBound, **kwargs: Any) -> Union[str, Collection[str]]:
        operator, delta = op_delta_from_int(node.offset)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    AxisBound = FormatTemplate("_d{level}_{op}{delta}")

    def visit_LoopOrder(self, node: common.LoopOrder, **kwargs) -> Union[str, Collection[str]]:
        if node is common.LoopOrder.FORWARD:
            return "for k_ in range(k, K):"
        elif node is common.LoopOrder.BACKWARD:
            return "for k_ in range(K-1, k-1, -1):"
        return ""

    def visit_VerticalPass(self, node: npir.VerticalPass, **kwargs):
        return self.generic_visit(
            node, is_serial=(node.direction != common.LoopOrder.PARALLEL), **kwargs
        )

    VerticalPass = JinjaTemplate(
        textwrap.dedent(
            """\
            # -- begin vertical block --{% set body_indent = 0 %}
            {% for assign in temp_defs %}{{ assign }}
            {% endfor %}k, K = {{ lower }}, {{ upper }}{% if direction %}
            {{ direction }}{% set body_indent = 4 %}{% endif %}
            {% for hblock in body %}{{ hblock | indent(body_indent, first=True) }}
            {% endfor %}# -- end vertical block --
            """
        )
    )

    def visit_HorizontalBlock(
        self, node: npir.HorizontalBlock, **kwargs
    ) -> Union[str, Collection[str]]:
        lower, upper = [0, 0], [0, 0]

        if extents := kwargs.get("field_extents"):
            fields = set(node.iter_tree().if_isinstance(npir.FieldSlice).getattr("name"))
            for field in fields:
                # The extent of masks has not yet been collected but is always zero.
                extents.setdefault(field, Extent.zeros())
            lower[0] = min(extents[field].to_boundary()[0][0] for field in fields)
            lower[1] = min(extents[field].to_boundary()[1][0] for field in fields)
            upper[0] = min(extents[field].to_boundary()[0][1] for field in fields)
            upper[1] = min(extents[field].to_boundary()[1][1] for field in fields)
        return self.generic_visit(node, h_lower=lower, h_upper=upper, **kwargs)

    HorizontalBlock = JinjaTemplate(
        textwrap.dedent(
            """\
            # --- begin horizontal block --
            i, I = _di_ - {{ h_lower[0] }}, _dI_ + {{ h_upper[0] }}
            j, J = _dj_ - {{ h_lower[1] }}, _dJ_ + {{ h_upper[1] }}
            {% for assign in body %}{{ assign }}
            {% endfor %}# --- end horizontal block --

            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, *, field_extents: FIELD_EXT_T, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.params, "_domain_", "_origin_"]
        kwargs["field_extents"] = field_extents
        return self.generic_visit(
            node,
            signature=", ".join(signature),
            data_view_class=ORIGIN_CORRECTED_VIEW_CLASS,
            **kwargs,
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            import numpy as np


            def run({{ signature }}):

                # -- begin domain boundary shortcuts --
                _di_, _dj_, _dk_ = 0, 0, 0
                _dI_, _dJ_, _dK_ = _domain_
                # -- end domain padding --

                {% for decl in field_decls %}{{ decl | indent(4) }}
                {% endfor %}
                # -- begin data views --
                {{ data_view_class | indent(4) }}
                {% for name in field_params %}{{ name }}_ = ShimmedView({{ name }}, _origin_["{{ name }}"])
                {% endfor %}# -- end data views --

                {% for pass in vertical_passes %}
                {{ pass | indent(4) }}
                {% endfor %}
            """
        )
    )

    def visit_NativeFunction(
        self, node: common.NativeFunction, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node == common.NativeFunction.MIN:
            return "minimum"
        elif node == common.NativeFunction.MAX:
            return "maximum"
        elif node == common.NativeFunction.POW:
            return "power"
        return self.generic_visit(node, **kwargs)

    NativeFuncCall = FormatTemplate("np.{func}({', '.join(arg for arg in args)})")

    While = JinjaTemplate(
        textwrap.dedent(
            """\
            _while_result = {{ cond }}
            while np.any(_while_result):
                {% for stmt in body %}{{ stmt }}
                {% endfor %}
                _while_result = {{ cond }}
            """
        )
    )

    def visit_While(self, node: npir.While, **kwargs: Any) -> str:
        cond = self.visit(node.cond, **kwargs)
        body = []
        for stmt in self.visit(node.body, **kwargs):
            body.extend(stmt.split("\n"))
        return self.While.render(cond=cond, body=body)
