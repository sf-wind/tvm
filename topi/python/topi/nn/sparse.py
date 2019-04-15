"""Elementwise operators"""
from __future__ import absolute_import
import tvm
from .. import tag
from tvm import autotvm
from .. import generic, nn, tag

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
    # assert M == 1

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
        assert weight_data.dtype in ("float32", "uint16", "int8")
        if weight_data.dtype == "uint16":
            block_ij_val = tvm_from_bf16(block_ij_val)

        x_val = data[i, BS_C * j + c]
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
def sparse_dense_kmnk(data, weight_data, weight_indices, weight_indptr):
    import topi
    (K, M) = topi.util.get_const_tuple(data.shape)
    (NUM, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
    NB = NB_plus_1 - 1    # assert topi.util.get_const_tuple(data.shape)[0] == 1
    # import pdb; pdb.set_trace()
    # oshape = (topi.util.get_const_tuple(data.shape)[0], topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    oshape = (NB * BS_R, M)
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    # bs_r = tvm.reduce_axis((0, weight_data.shape[1]), name="bs_r")
    bs_c = tvm.reduce_axis((0, BS_C), name="bs_c")
    '''
        def f(row, i):
            assert row.dtype == "int32"
            row_start = weight_indptr[row // BS_R]
            row_end = weight_indptr[row //BS_R + 1]
            row_elems = row_end - row_start
            elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
            elem = row_start + elem_idx
            a_val = weight_data[elem, row % BS_R, bs_c].astype("float32")
            weight_val = data[weight_indices[elem] * BS_C + bs_c, i]

            return tvm.sum(a_val * weight_val, axis=[elem_idx, bs_c])
        return tvm.compute(oshape, f, name="sparse_dense_kmnk_vm",
                           tag="sparse_dense_kmnk_vm")
    '''
    def f(nb, r, i):
        row_start = weight_indptr[nb]
        row_end = weight_indptr[nb + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem, r, bs_c].astype("float32")
        weight_val = data[weight_indices[elem] * BS_C + bs_c, i]
        return tvm.sum(a_val * weight_val, axis=[elem_idx, bs_c])
    Y = tvm.compute((NB, BS_R, M), f,
        name="sparse_dense_kmnk_block",
        tag = "sparse_dense_kmnk_block")
    return tvm.compute(oshape, lambda n, m: Y[n // BS_R, n % BS_R, m],
        name="sparse_dense_kmnk", tag="sparse_dense_kmnk")

@tvm.target.generic_func
def sparse_dense_mknk(data, weight_data, weight_indices, weight_indptr):
    import topi
    (M, K) = topi.util.get_const_tuple(data.shape)
    (NUM, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
    NB = NB_plus_1 - 1
    oshape = (M, NB * BS_R)
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    assert K % BS_C == 0
    X = tvm.compute((M, K // BS_C, BS_C), lambda m, ko, ki: data[m, ko * BS_C + ki])
    # bs_r = tvm.reduce_axis((0, weight_data.shape[1]), name="bs_r")
    bs_c = tvm.reduce_axis((0, BS_C), name="bs_c")
    def f(i, nb, r):
        row_start = weight_indptr[nb]
        row_end = weight_indptr[nb + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem, r, bs_c].astype("float32")
        weight_val = X[i, weight_indices[elem], bs_c]
        return tvm.sum(a_val * weight_val, axis=[elem_idx, bs_c])
    Y = tvm.compute((M, NB, BS_R), f,
        name="sparse_dense_kmnk_block",
        tag = "sparse_dense_kmnk_block")
    return tvm.compute(oshape, lambda m, n: Y[m, n // BS_R, n % BS_R],
        name="sparse_dense_mknk", tag="sparse_dense_mknk")

@tvm.target.generic_func
def gru_gates(input_transform, hidden_transform):
    import topi

    def approx_sigmoid(v):
        x = tvm.abs(v)
        x2 = v * v
        e = 1.0 + x + x2 * 0.5658 + 0.143 * x2 * x2
        e_pos = e / (1.0 + e)
        e_neg = 1 / (1.0 + e)

        return tvm.if_then_else(v >= 0, e_pos, e_neg)

    def approx_tanh(v):
        x = tvm.abs(v)
        x2 = v * v
        e = 1.0 + x + x2 * 0.5658 + 0.143 * x2 * x2

        def sign(x):
            return tvm.if_then_else(v >= 0, 1, -1)

        return sign(v) * (e - 1 / e) / (e + 1 / e)

    dim3 = topi.util.get_const_int(input_transform.shape[1])
    assert dim3 % 3 == 0
    dim = dim3 // 3

    def gru_gate(n, d):
        u_t = approx_sigmoid(input_transform[n, d] + hidden_transform[n, d])
        r_t = approx_sigmoid(input_transform[n, d + dim] + hidden_transform[n, d + dim])
        e_t = approx_tanh(r_t * hidden_transform[n, d + 2 * dim] + input_transform[n, d + 2 * dim])
        return u_t * hidden_transform[n, d] + (1.0 - u_t) * e_t

    return tvm.compute((input_transform.shape[0], dim), gru_gate, name="gru_gates", tag=topi.tag.ELEMWISE)




def sdense_default(data, weight_data, weight_indices, weight_indptr):
    if len(weight_data.shape) == 1:
        import topi
        # import pdb; pdb.set_trace()
        # assert topi.util.get_const_tuple(data.shape)[0] == 1

        oshape = (
            topi.util.get_const_tuple(data.shape)[0],
            topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
        assert weight_indices.dtype == "int32", weight_indices.dtype
        assert weight_indptr.dtype == "int32", weight_indptr.dtype

        def f(i, row):
            # import pdb; pdb.set_trace()
            assert row.dtype == "int32"
            row_start = weight_indptr[row]
            row_end = weight_indptr[row + 1]
            row_elems = row_end - row_start
            elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
            elem = row_start + elem_idx
            offset = 0
            if len(weight_data.shape) == 3:
                # result incorrect
                a_val = weight_data[elem, 0, 0].astype("float32")
            else:
                a_val = weight_data[elem].astype("float32")
            weight_val = data[i, weight_indices[elem]]
            return tvm.sum(a_val * weight_val, axis=elem_idx)
        return tvm.compute(
            oshape, f, name="sdense", tag="sdense_default")
    elif len(weight_data.shape) == 3:
        import topi
        (M, K) = topi.util.get_const_tuple(data.shape)
        (NUM, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
        (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
        NB = NB_plus_1 - 1
        oshape = (M, NB * BS_R)
        assert weight_indices.dtype == "int32", weight_indices.dtype
        assert weight_indptr.dtype == "int32", weight_indptr.dtype
        assert K % BS_C == 0
        X = tvm.compute((M, K // BS_C, BS_C), lambda m, ko, ki: data[m, ko * BS_C + ki])
        # bs_r = tvm.reduce_axis((0, weight_data.shape[1]), name="bs_r")
        bs_c = tvm.reduce_axis((0, BS_C), name="bs_c")
        def f(i, nb, r):
            row_start = weight_indptr[nb]
            row_end = weight_indptr[nb + 1]
            row_elems = row_end - row_start
            elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
            elem = row_start + elem_idx
            a_val = weight_data[elem, r, bs_c].astype("float32")
            weight_val = X[i, weight_indices[elem], bs_c]
            return tvm.sum(a_val * weight_val, axis=[elem_idx, bs_c])
        Y = tvm.compute((M, NB, BS_R), f,
            name="sparse_dense_kmnk_block",
            tag = "sparse_dense_kmnk_block")
        return tvm.compute(oshape, lambda m, n: Y[m, n // BS_R, n % BS_R],
            name="sparse_dense_mknk", tag="sparse_dense_mknk")


@tvm.target.override_native_generic_func("sdense")
def sdense(data, weight_data, weight_indices, weight_indptr,
           data_layout="NI", kernel_layout="OI", out_layout=""):
    return sdense_default(data, weight_data, weight_indices, weight_indptr)


@tvm.target.override_native_generic_func("grucell")
def grucell(input, tw_x, tb_x, tw_z, tb_z, tw_in, tb_in, tw_hn, tb_hn,
           data_layout="NI", kernel_layout="OI", out_layout=""):
    assert False
    return None

@tvm.target.override_native_generic_func("sgrucell")
def sgrucell(input, w_x_data, w_x_indices, w_x_indptr, b_x,
    w_z_data, w_z_indices, w_z_indptr, b_z,
    w_in_data, w_in_indices, w_in_indptr, b_in,
    w_hn_data, w_hn_indices, w_hn_indptr, b_hn,
           data_layout="NI", kernel_layout="OI", out_layout=""):
    assert False
    return None
