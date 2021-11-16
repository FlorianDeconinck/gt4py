# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import abc
import collections.abc
import copy
import inspect
import os
import re
import sys
import time
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import dace
import numpy as np
from dace.frontend.python.common import SDFGClosure, SDFGConvertible

import gt4py.backend as gt_backend
import gt4py.storage as gt_storage
import gt4py.utils as gt_utils
from gt4py.definitions import DomainInfo, FieldInfo, Index, ParameterInfo, Shape
from gt4py.utils import shash


FieldType = Union[gt_storage.storage.Storage, np.ndarray]
OriginType = Union[Tuple[int, int, int], Dict[str, Tuple[int, ...]]]

_loaded_sdfgs: Dict[str, dace.SDFG] = dict()


@dataclass(frozen=True)
class FrozenStencil(SDFGConvertible):
    """Stencil with pre-computed domain and origin for each field argument."""

    stencil_object: "StencilObject"
    origin: Dict[str, Tuple[int, ...]]
    domain: Tuple[int, ...]

    def __post_init__(self):
        for name, field_info in self.stencil_object.field_info.items():
            if field_info is not None and (
                name not in self.origin or len(self.origin[name]) != field_info.ndim
            ):
                raise ValueError(
                    f"'{name}' origin {self.origin.get(name)} is not a {field_info.ndim}-dimensional integer tuple"
                )

    def __call__(self, **kwargs) -> None:
        assert "origin" not in kwargs and "domain" not in kwargs
        exec_info = kwargs.get("exec_info")

        if exec_info is not None:
            exec_info["call_run_start_time"] = time.perf_counter()

        field_args = {name: kwargs[name] for name in self.stencil_object.field_info.keys()}
        parameter_args = {name: kwargs[name] for name in self.stencil_object.parameter_info.keys()}

        self.stencil_object.run(
            _domain_=self.domain,
            _origin_=self.origin,
            exec_info=exec_info,
            **field_args,
            **parameter_args,
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()

    def _sdfg_add_arrays_and_edges(self, wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs):
        device = gt_backend.from_name(self.stencil_object.backend).storage_info["device"]
        for name, array in inner_sdfg.arrays.items():
            if isinstance(array, dace.data.Array) and not array.transient:
                axes = self.stencil_object.field_info[name].axes

                shape = [f"__{name}_{axis}_size" for axis in axes] + [
                    str(d) for d in self.stencil_object.field_info[name].data_dims
                ]

                wrapper_sdfg.add_array(
                    name,
                    dtype=array.dtype,
                    strides=array.strides,
                    shape=shape,
                    storage=dace.StorageType.GPU_Global
                    if device == "gpu"
                    else dace.StorageType.Default,
                )
                if isinstance(self.origin, tuple):
                    origin = [o for a, o in zip("IJK", self.origin) if a in axes]
                else:
                    origin = self.origin.get(name, self.origin.get("_all_", None))
                    if len(origin) == 3:
                        origin = [o for a, o in zip("IJK", origin) if a in axes]

                subset_strs = [
                    f"{o - e}:{o - e + s}"
                    for o, e, s in zip(
                        origin,
                        self.stencil_object.field_info[name].boundary.lower_indices,
                        inner_sdfg.arrays[name].shape,
                    )
                ]
                subset_strs += [f"0:{d}" for d in self.stencil_object.field_info[name].data_dims]

                if name in inputs:
                    state.add_edge(
                        state.add_read(name),
                        None,
                        nsdfg,
                        name,
                        dace.Memlet.simple(name, ",".join(subset_strs)),
                    )
                if name in outputs:
                    state.add_edge(
                        nsdfg,
                        name,
                        state.add_write(name),
                        None,
                        dace.Memlet.simple(name, ",".join(subset_strs)),
                    )

    def _sdfg_specialize_symbols(self, wrapper_sdfg):
        ival, jval, kval = self.domain[0], self.domain[1], self.domain[2]
        for sdfg in wrapper_sdfg.all_sdfgs_recursive():
            if sdfg.parent_nsdfg_node is not None:
                symmap = sdfg.parent_nsdfg_node.symbol_mapping

                if "__I" in symmap:
                    ival = symmap["__I"]
                    del symmap["__I"]
                if "__J" in symmap:
                    jval = symmap["__J"]
                    del symmap["__J"]
                if "__K" in symmap:
                    kval = symmap["__K"]
                    del symmap["__K"]

            sdfg.replace("__I", ival)
            if "__I" in sdfg.symbols:
                sdfg.remove_symbol("__I")
            sdfg.replace("__J", jval)
            if "__J" in sdfg.symbols:
                sdfg.remove_symbol("__J")
            sdfg.replace("__K", kval)
            if "__K" in sdfg.symbols:
                sdfg.remove_symbol("__K")

            for val in ival, jval, kval:
                sym = dace.symbolic.pystr_to_symbolic(val)
                for fsym in sym.free_symbols:
                    if sdfg.parent_nsdfg_node is not None:
                        sdfg.parent_nsdfg_node.symbol_mapping[str(fsym)] = fsym
                    if fsym not in sdfg.symbols:
                        if fsym in sdfg.parent_sdfg.symbols:
                            sdfg.add_symbol(str(fsym), stype=sdfg.parent_sdfg.symbols[str(fsym)])
                        else:
                            sdfg.add_symbol(str(fsym), stype=dace.dtypes.int32)

    def _sdfg_freeze_domain_and_origin(self, inner_sdfg: dace.SDFG):
        wrapper_sdfg = dace.SDFG("frozen_" + inner_sdfg.name)
        state = wrapper_sdfg.add_state("frozen_" + inner_sdfg.name + "_state")

        inputs = set()
        outputs = set()
        for inner_state in inner_sdfg.nodes():
            for node in inner_state.nodes():
                if (
                    not isinstance(node, dace.nodes.AccessNode)
                    or inner_sdfg.arrays[node.data].transient
                ):
                    continue
                if node.access != dace.dtypes.AccessType.WriteOnly:
                    inputs.add(node.data)
                if node.access != dace.dtypes.AccessType.ReadOnly:
                    outputs.add(node.data)

        nsdfg = state.add_nested_sdfg(inner_sdfg, None, inputs, outputs)

        self._sdfg_add_arrays_and_edges(wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs)

        # in special case of empty domain, remove entire SDFG.
        if any(d == 0 for d in self.domain):
            states = wrapper_sdfg.states()
            assert len(states) == 1
            for node in states[0].nodes():
                state.remove_node(node)

        # make sure that symbols are passed throught o inner sdfg
        for symbol in nsdfg.sdfg.free_symbols:
            if symbol not in wrapper_sdfg.symbols:
                wrapper_sdfg.add_symbol(symbol, nsdfg.sdfg.symbols[symbol])

        self._sdfg_specialize_symbols(wrapper_sdfg)

        for _, _, array in wrapper_sdfg.arrays_recursive():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.SDFG

        true_args = [
            arg
            for arg in wrapper_sdfg.signature_arglist(with_types=False)
            if not re.match("__.*_._stride", arg) and not re.match("__.*_._size", arg)
        ]
        signature = self.__sdfg_signature__()
        wrapper_sdfg.arg_names = [a for a in signature[0] if a not in signature[1]]
        assert len(wrapper_sdfg.arg_names) == len(true_args)

        return wrapper_sdfg

    def _assert_dace_backend(self):
        if not hasattr(self.stencil_object, "sdfg"):
            raise TypeError(
                f"Only dace backends are supported in DaCe-orchestrated programs."
                f' (found "{self.stencil_object.backend}")'
            )

    def __sdfg__(self, *args, **kwargs):
        self._assert_dace_backend()
        frozen_hash = shash(type(self.stencil_object)._gt_id_, self.origin, self.domain)

        # check if same sdfg already cached in memory
        if frozen_hash in _loaded_sdfgs:
            return copy.deepcopy(_loaded_sdfgs[frozen_hash])

        # check if same sdfg already cached on disk
        basename = os.path.splitext(self.stencil_object._file_name)[0]
        filename = f"{basename}_wrapped.sdfg"
        try:
            _loaded_sdfgs[frozen_hash] = dace.SDFG.from_file(filename)
            return copy.deepcopy(_loaded_sdfgs[frozen_hash])
        except FileNotFoundError:
            pass

        # otherwise, wrap and save sdfg from scratch
        inner_sdfg = self.stencil_object.sdfg

        _loaded_sdfgs[frozen_hash] = self._sdfg_freeze_domain_and_origin(inner_sdfg)
        _loaded_sdfgs[frozen_hash].save(filename)

        return copy.deepcopy(_loaded_sdfgs[frozen_hash])

    def __sdfg_signature__(self):
        self._assert_dace_backend()

        special_args = {"self", "domain", "origin", "validate_args", "exec_info"}
        args = []
        consts = []
        for arg in (
            inspect.getfullargspec(self.stencil_object.__call__).args
            + inspect.getfullargspec(self.stencil_object.__call__).kwonlyargs
        ):
            if arg in special_args:
                continue
            if (
                arg in self.stencil_object.field_info
                and self.stencil_object.field_info[arg] is None
            ):
                consts.append(arg)
            if (
                arg in self.stencil_object.parameter_info
                and self.stencil_object.parameter_info[arg] is None
            ):
                consts.append(arg)
            args.append(arg)
        return (args, consts)

    def __sdfg_closure__(self, *args, **kwargs):
        self._assert_dace_backend()
        return {}

    def closure_resolver(self, constant_args, parent_closure=None):
        self._assert_dace_backend()
        return SDFGClosure()


class StencilObject(abc.ABC):
    """Generic singleton implementation of a stencil callable.

    This class is used as base class for specific subclass generated
    at run-time for any stencil definition and a unique set of external symbols.
    Instances of this class do not contain state and thus it is
    implemented as a singleton: only one instance per subclass is actually
    allocated (and it is immutable).

    The callable interface is the same of the stencil definition function,
    with some extra keyword arguments.

    Keyword Arguments
    ------------------
    domain : `Sequence` of `int`, optional
        Shape of the computation domain. If `None`, it will be used the
        largest feasible domain according to the provided input fields
        and origin values (`None` by default).

    origin :  `[int * ndims]` or `{'field_name': [int * ndims]}`, optional
        If a single offset is passed, it will be used for all fields.
        If a `dict` is passed, there could be an entry for each field.
        A special key *'_all_'* will represent the value to be used for all
        the fields not explicitly defined. If `None` is passed or it is
        not possible to assign a value to some field according to the
        previous rule, the value will be inferred from the `boundary` attribute
        of the `field_info` dict. Note that the function checks if the origin values
        are at least equal to the `boundary` attribute of that field,
        so a 0-based origin will only be acceptable for fields with
        a 0-area support region.

    exec_info : `dict`, optional
        Dictionary used to store information about the stencil execution.
        (`None` by default). If the dictionary contains the magic key
        '__aggregate_data' and it evaluates to `True`, the dictionary is
        populated with a nested dictionary per class containing different
        performance statistics. These include the stencil calls count, the
        cumulative time spent in all stencil calls, and the actual time spent
        in carrying out the computations.

    """

    # Those attributes are added to the class at loading time:
    _gt_id_: str
    definition_func: Callable[..., Any]

    def __new__(cls, *args, **kwargs):
        if getattr(cls, "_instance", None) is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __setattr__(self, key, value) -> None:
        raise AttributeError("Attempting a modification of an attribute in a frozen class")

    def __delattr__(self, item) -> None:
        raise AttributeError("Attempting a deletion of an attribute in a frozen class")

    def __eq__(self, other) -> bool:
        return type(self) == type(other)

    def __str__(self) -> str:
        result = """
<StencilObject: {name}> [backend="{backend}"]
    - I/O fields: {fields}
    - Parameters: {params}
    - Constants: {constants}
    - Version: {version}
    - Definition ({func}):
{source}
        """.format(
            name=self.options["module"] + "." + self.options["name"],
            version=self._gt_id_,
            backend=self.backend,
            fields=self.field_info,
            params=self.parameter_info,
            constants=self.constants,
            func=self.definition_func,
            source=self.source,
        )

        return result

    def __hash__(self) -> int:
        return int.from_bytes(type(self)._gt_id_.encode(), byteorder="little")

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def source(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def domain_info(self) -> DomainInfo:
        pass

    @property
    @abc.abstractmethod
    def field_info(self) -> Dict[str, FieldInfo]:
        pass

    @property
    @abc.abstractmethod
    def parameter_info(self) -> Dict[str, ParameterInfo]:
        pass

    @property
    @abc.abstractmethod
    def constants(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def options(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def _make_origin_dict(origin: Any) -> Dict[str, Index]:
        try:
            if isinstance(origin, dict):
                return dict(origin)
            if origin is None:
                return {}
            if isinstance(origin, collections.abc.Iterable):
                return {"_all_": Index.from_value(origin)}
            if isinstance(origin, int):
                return {"_all_": Index.from_k(origin)}
        except Exception:
            pass

        raise ValueError("Invalid 'origin' value ({})".format(origin))

    def _get_max_domain(
        self,
        field_args: Dict[str, Any],
        origin: Dict[str, Tuple[int, ...]],
        *,
        squeeze: bool = True,
    ) -> Shape:
        """Return the maximum domain size possible.

        Parameters
        ----------
            field_args:
                Mapping from field names to actually passed data arrays.
            origin:
                The origin for each field.
            squeeze:
                Convert non-used domain dimensions to singleton dimensions.

        Returns
        -------
            `Shape`: the maximum domain size.
        """
        domain_ndim = self.domain_info.ndim
        max_size = sys.maxsize
        max_domain = Shape([max_size] * domain_ndim)

        for name, field_info in self.field_info.items():
            if field_info is not None:
                assert field_args.get(name, None) is not None, f"Invalid value for '{name}' field."
                field = field_args[name]
                api_domain_mask = field_info.domain_mask
                api_domain_ndim = field_info.domain_ndim
                upper_indices = field_info.boundary.upper_indices.filter_mask(api_domain_mask)
                field_origin = Index.from_value(origin[name])
                field_domain = tuple(
                    field.shape[i] - (field_origin[i] + upper_indices[i])
                    for i in range(api_domain_ndim)
                )
                max_domain &= Shape.from_mask(field_domain, api_domain_mask, default=max_size)

        if squeeze:
            return Shape([i if i != max_size else 1 for i in max_domain])
        else:
            return max_domain

    def _validate_args(  # noqa: C901  # Function is too complex
        self,
        field_args: Dict[str, FieldType],
        param_args: Dict[str, Any],
        domain: Tuple[int, ...],
        origin: Dict[str, Tuple[int, ...]],
    ) -> None:
        """
        Validate input arguments to _call_run.

        Raises
        -------
            ValueError
                If invalid data or inconsistent options are specified.

            TypeError
                If an incorrect field or parameter data type is passed.
        """
        assert isinstance(field_args, dict) and isinstance(param_args, dict)

        # validate domain sizes
        domain_ndim = self.domain_info.ndim
        if len(domain) != domain_ndim:
            raise ValueError(f"Invalid 'domain' value '{domain}'")

        try:
            domain = Shape(domain)
        except Exception:
            raise ValueError("Invalid 'domain' value ({})".format(domain))

        if not domain > Shape.zeros(domain_ndim):
            raise ValueError(f"Compute domain contains zero sizes '{domain}')")

        if not domain <= (max_domain := self._get_max_domain(field_args, origin, squeeze=False)):
            raise ValueError(
                f"Compute domain too large (provided: {domain}, maximum: {max_domain})"
            )

        # assert compatibility of fields with stencil
        for name, field_info in self.field_info.items():
            if field_info is not None:
                if name not in field_args:
                    raise ValueError(f"Missing value for '{name}' field.")
                field = field_args[name]

                if not gt_backend.from_name(self.backend).storage_info["is_compatible_layout"](
                    field
                ):
                    raise ValueError(
                        f"The layout of the field {name} is not compatible with the backend."
                    )

                if not gt_backend.from_name(self.backend).storage_info["is_compatible_type"](field):
                    raise ValueError(
                        f"Field '{name}' has type '{type(field)}', which is not compatible with the '{self.backend}' backend."
                    )
                elif type(field) is np.ndarray:
                    warnings.warn(
                        "NumPy ndarray passed as field. This is discouraged and only works with constraints and only for certain backends.",
                        RuntimeWarning,
                    )

                field_dtype = self.field_info[name].dtype
                if not field.dtype == field_dtype:
                    raise TypeError(
                        f"The dtype of field '{name}' is '{field.dtype}' instead of '{field_dtype}'"
                    )

                if isinstance(field, gt_storage.storage.Storage) and not field.is_stencil_view:
                    raise ValueError(
                        f"An incompatible view was passed for field {name} to the stencil. "
                    )

                # Check: domain + halo vs field size
                field_info = self.field_info[name]
                field_domain_mask = field_info.domain_mask
                field_domain_ndim = field_info.domain_ndim
                field_domain_origin = Index.from_mask(origin[name], field_domain_mask[:domain_ndim])

                if field.ndim != field_domain_ndim + len(field_info.data_dims):
                    raise ValueError(
                        f"Storage for '{name}' has {field.ndim} dimensions but the API signature "
                        f"expects {field_domain_ndim + len(field_info.data_dims)} ('{field_info.axes}[{field_info.data_dims}]')"
                    )

                if (
                    isinstance(field, gt_storage.storage.Storage)
                    and tuple(field.mask)[:domain_ndim] != field_domain_mask
                ):
                    raise ValueError(
                        f"Storage for '{name}' has domain mask '{field.mask}' but the API signature "
                        f"expects '[{', '.join(field_info.axes)}]'"
                    )

                # Check: data dimensions shape
                if field.shape[field_domain_ndim:] != field_info.data_dims:
                    raise ValueError(
                        f"Field '{name}' expects data dimensions {field_info.data_dims} but got {field.shape[field_domain_ndim:]}"
                    )

                min_origin = gt_utils.interpolate_mask(
                    field_info.boundary.lower_indices.filter_mask(field_domain_mask),
                    field_domain_mask,
                    default=0,
                )
                if field_domain_origin < min_origin:
                    raise ValueError(
                        f"Origin for field {name} too small. Must be at least {min_origin}, is {field_domain_origin}"
                    )

                spatial_domain = typing.cast(Shape, domain).filter_mask(field_domain_mask)
                upper_indices = field_info.boundary.upper_indices.filter_mask(field_domain_mask)
                min_shape = tuple(
                    o + d + h for o, d, h in zip(field_domain_origin, spatial_domain, upper_indices)
                )
                if min_shape > field.shape:
                    raise ValueError(
                        f"Shape of field {name} is {field.shape} but must be at least {min_shape} for given domain and origin."
                    )

        # assert compatibility of parameters with stencil
        for name, parameter_info in self.parameter_info.items():
            if parameter_info is not None:
                if name not in param_args:
                    raise ValueError(f"Missing value for '{name}' parameter.")
                if type(parameter := param_args[name]) != parameter_info.dtype:
                    raise TypeError(
                        f"The type of parameter '{name}' is '{type(parameter)}' instead of '{parameter_info.dtype}'"
                    )

    def _normalize_origins(
        self, field_args: Dict[str, FieldType], origin: Optional[OriginType]
    ) -> Dict[str, Tuple[int, ...]]:
        origin = self._make_origin_dict(origin)
        all_origin = origin.get("_all_", None)

        # Set an appropriate origin for all fields
        for name, field_info in self.field_info.items():
            if field_info is not None:
                assert name in field_args, f"Missing value for '{name}' field."
                field_origin = origin.get(name, None)

                if field_origin is not None:
                    field_origin_ndim = len(field_origin)
                    if field_origin_ndim != field_info.ndim:
                        assert (
                            field_origin_ndim == field_info.domain_ndim
                        ), f"Invalid origin specification ({field_origin}) for '{name}' field."
                        origin[name] = (*field_origin, *((0,) * len(field_info.data_dims)))

                elif all_origin is not None:
                    origin[name] = (
                        *gt_utils.filter_mask(all_origin, field_info.domain_mask),
                        *((0,) * len(field_info.data_dims)),
                    )

                elif isinstance(field_arg := field_args.get(name), gt_storage.storage.Storage):
                    origin[name] = field_arg.default_origin

                else:
                    origin[name] = (0,) * field_info.ndim

        return origin

    def _call_run(
        self,
        field_args: Dict[str, FieldType],
        parameter_args: Dict[str, Any],
        domain: Optional[Tuple[int, ...]],
        origin: Optional[OriginType],
        *,
        validate_args: bool = True,
        exec_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Check and preprocess the provided arguments (called by :class:`StencilObject` subclasses).

        Note that this function will always try to expand simple parameter values to complete
        data structures by repeating the same value as many times as needed.

        Parameters
        ----------
            field_args: `dict`
                Mapping from field names to actually passed data arrays.
                This parameter encapsulates `*args` in the actual stencil subclass
                by doing: `{input_name[i]: arg for i, arg in enumerate(args)}`

            parameter_args: `dict`
                Mapping from parameter names to actually passed parameter values.
                This parameter encapsulates `**kwargs` in the actual stencil subclass
                by doing: `{name: value for name, value in kwargs.items()}`

        Check :class:`StencilObject` for a full specification of the `domain`,
        `origin` and `exec_info` keyword arguments.

        Returns
        -------
            `None`

        Raises
        -------
            ValueError
                If invalid data or inconsistent options are specified.
        """
        if exec_info is not None:
            exec_info["call_run_start_time"] = time.perf_counter()

        origin = self._normalize_origins(field_args, origin)

        if domain is None:
            domain = self._get_max_domain(field_args, origin)

        if validate_args:
            self._validate_args(field_args, parameter_args, domain, origin)

        self.run(
            _domain_=domain, _origin_=origin, exec_info=exec_info, **field_args, **parameter_args
        )

        if exec_info is not None:
            exec_info["call_run_end_time"] = time.perf_counter()

    def freeze(
        self: "StencilObject", *, origin: Dict[str, Tuple[int, ...]], domain: Tuple[int, ...]
    ) -> FrozenStencil:
        """Return a StencilObject wrapper with a fixed domain and origin for each argument.

        Parameters
        ----------
            origin: `dict`
                The origin for each Field argument.

            domain: `Sequence` of `int`
                The compute domain shape for the frozen stencil.

        Notes
        ------
        Both `origin` and `domain` arguments should be compatible with the domain and origin
        specification defined in :class:`StencilObject`.

        Returns
        -------
            `FrozenStencil`
                The stencil wrapper. This should be called with the regular stencil arguments,
                but the field origins and domain cannot be changed. Note, no checking of origin
                or domain occurs at call time so it is the users responsibility to ensure
                correct usage.
        """
        return FrozenStencil(self, origin, domain)
