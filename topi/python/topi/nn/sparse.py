"""Elementwise operators"""
from __future__ import absolute_import
import tvm
from .. import tag

@tvm.target.generic_func
def sparse_dense(data, weight_data, weight_indices, weight_indptr):
    assert len(weight_data.shape) in (1, 3)
    if len(weight_data.shape) == 1:
        return sparse_dense_csrmv(data, weight_data, weight_indices, weight_indptr)
    if len(weight_data.shape) == 3:
        return sparse_dense_bsrmv(data, weight_data, weight_indices, weight_indptr)


def sparse_dense_csrmv(data, weight_data, weight_indices, weight_indptr):
    import topi
    # assert topi.util.get_const_tuple(data.shape)[0] == 1
    oshape = (
        topi.util.get_const_tuple(data.shape)[0],
        topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype

    def f(i, row):
        assert row.dtype == "int32"
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem].astype("float32")
        weight_val = data[i, weight_indices[elem]]
        return tvm.sum(a_val * weight_val, axis=elem_idx)
    return tvm.compute(
        oshape, f, name="sparse_dense", tag="sparse_dense_csrmv")

def tvm_from_bf16(x):
    return tvm.call_pure_intrin("float32", "reinterpret", x.astype("uint32") << 16)

def sparse_dense_bsrmv(data, weight_data, weight_indices, weight_indptr):
    import topi
    (M, K) = topi.util.get_const_tuple(data.shape)
    (_, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
    NB = NB_plus_1 - 1
    assert M == 1

    oshape = (M, NB, BS_R)

    def f(i, nb, r):
        row_start = weight_indptr[nb]
        row_end = weight_indptr[nb + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        jj = row_start + elem_idx
        c = tvm.reduce_axis((0, BS_C), name="c")
        j = weight_indices[jj]
        block_ij_val = weight_data[jj][r][c]
        assert weight_data.dtype in ("float32", "uint16")
        if weight_data.dtype == "uint16":
            block_ij_val = tvm_from_bf16(block_ij_val)

        x_val = data[0, BS_C * j + c]
        return tvm.sum(block_ij_val * x_val, axis=[elem_idx, c])

    Y = tvm.compute(
        oshape, f,
        name="sparse_dense_bsrmv_block",
        tag="sparse_dense_bsrmv_block")
    return tvm.compute((M, NB * BS_R), lambda m, n: Y[m, n // BS_R, n % BS_R],
                       name="sparse_dense_bsrmv",
                       tag="sparse_dense_bsrmv")

@tvm.target.generic_func
def sparse_dense2(data, weight_data, weight_indices, weight_indptr):
    import topi
    # assert topi.util.get_const_tuple(data.shape)[0] == 1
    # import pdb; pdb.set_trace()
    # oshape = (topi.util.get_const_tuple(data.shape)[0], topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    oshape = (topi.util.get_const_tuple(weight_indptr.shape)[0] - 1,
              topi.util.get_const_tuple(data.shape)[1])
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    def f(row, i):
        assert row.dtype == "int32"
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem].astype("float32")
        weight_val = data[weight_indices[elem], i]
        return tvm.sum(a_val * weight_val, axis=elem_idx)
    return tvm.compute(oshape, f, name="sparse_dense2", tag="sparse_dense2")


@tvm.target.generic_func
def sparse_dense_structure(data, weight_data, weight_indices, weight_indptr):
    import topi
    (K, M) = topi.util.get_const_tuple(data.shape)
    (_, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
    NB = NB_plus_1 - 1    # assert topi.util.get_const_tuple(data.shape)[0] == 1
    import pdb; pdb.set_trace()
    # oshape = (topi.util.get_const_tuple(data.shape)[0], topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    oshape = (NB * BS_R, M)

    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    # bs_r = tvm.reduce_axis((0, weight_data.shape[1]), name="bs_r")
    bs_c = tvm.reduce_axis((0, weight_data.shape[2]), name="bs_c")
    def f(row, i):
        # import pdb; pdb.set_trace()
        assert row.dtype == "int32"
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem, i % BS_R, bs_c].astype("float32")
        weight_val = data[weight_indices[elem] * BS_C + bs_c, i]
        a_val = weight_data[elem, 0, 0].astype("float32")
        weight_val = data[weight_indices[elem], i]

        return tvm.sum(a_val * weight_val, axis=[elem_idx, bs_c])
    return tvm.compute(oshape, f, name="sparse_dense_structure",
                       tag="sparse_dense_structure")


@tvm.target.generic_func
def sparse_dense_structure2(data, weight_data, weight_indices, weight_indptr):
    import topi
    # assert topi.util.get_const_tuple(data.shape)[0] == 1
    # import pdb; pdb.set_trace()
    # oshape = (topi.util.get_const_tuple(data.shape)[0], topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    oshape = (topi.util.get_const_tuple(weight_indptr.shape)[0] - 1,
              topi.util.get_const_tuple(data.shape)[2])
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    sidx = tvm.reduce_axis((0, weight_data.shape[1]), name="sidx")
    def f(row, i):
        assert row.dtype == "int32"
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem, sidx].astype("float32")
        weight_val = data[weight_indices[elem], sidx, i]
        return tvm.sum(a_val * weight_val, axis=[elem_idx, sidx])
    return tvm.compute(oshape, f, name="sparse_dense_structure2",
                       tag="sparse_dense_structure2")
