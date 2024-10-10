# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np

from gt4py import storage as gt_storage
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import BACKWARD, PARALLEL, computation, interval


def test_simple_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(BACKWARD):
            with interval(-2, -1):  # block 1
                field_out = field_in
            with interval(0, -2):  # block 2
                field_out = field_in
        with computation(BACKWARD):
            with interval(-1, None):  # block 3
                field_out = 2 * field_in
            with interval(0, -1):  # block 4
                field_out = 3 * field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0:-1], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, -1], 2)


def test_tmp_stencil():
    field_in = gt_storage.ones(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float64, backend="debug", shape=(6, 6, 6), aligned_index=(0, 0, 0)
    )

    @gtscript.stencil(backend="debug")
    def stencil(field_in: gtscript.Field[np.float64], field_out: gtscript.Field[np.float64]):
        with computation(PARALLEL):
            with interval(...):
                tmp = field_in + 1
        with computation(PARALLEL):
            with interval(...):
                field_out = tmp[-1, 0, 0] + tmp[1, 0, 0]

    stencil(field_in, field_out, origin=(1, 1, 0), domain=(4, 4, 6))

    # the inside of the domain is 4
    np.testing.assert_allclose(field_out.view(np.ndarray)[1:-1, 1:-1, :], 4)
    # the rest is 0
    np.testing.assert_allclose(field_out.view(np.ndarray)[0:1, :, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[-1:, :, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, 0:1, :], 0)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, -1:, :], 0)
