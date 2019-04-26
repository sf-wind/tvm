from .. import generic, nn, tag
from ..util import traverse_inline, get_const_int
import tvm
from tvm import autotvm
from .util import get_fp32_len
from tvm.autotvm.task.space import OtherOptionEntity, ReorderEntity
import topi
import os

@autotvm.register_topi_compute(nn.sdense, 'cpu', ['direct'])
def sdense(cfg, data, weight_data, weight_indices, weight_indptr,
                 data_layout="NI", kernel_layout="OI", out_layout=""):
    return sdense_compute(cfg, data, weight_data, weight_indices, weight_indptr,
                          data_layout = data_layout, kernel_layout = kernel_layout)

def tvm_from_bf16(x):
    return tvm.call_pure_intrin("float32", "reinterpret", x.astype("uint32") << 16)


def specify_range(cfg, prefix, num):
    for i in range(num):
        cfg.define_knob(prefix + str(num) + "_" + str(i), range(i+1))




def sdense_compute(cfg, data, weight_data, weight_indices, weight_indptr,
                   data_layout, kernel_layout):
    import topi
    EVEN_ENTRIES = True if "TVM_SDENSE_EVEN_ENTRIES" in os.environ else False

    # import pdb; pdb.set_trace()
    (NUM, BS_R, BS_C) = topi.util.get_const_tuple(weight_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(weight_indptr.shape)
    NB = NB_plus_1 - 1
    K = None
    NK = None
    if len(data.shape) == 3:
        assert data_layout == "NI"
        (M, NK, BS_C_1) = topi.util.get_const_tuple(data.shape)
        assert BS_C == BS_C_1
    elif len(data.shape) == 2:
        if data_layout == "NI":
            (M, K) = topi.util.get_const_tuple(data.shape)
        else:
            (K, M) = topi.util.get_const_tuple(data.shape)
        assert K % BS_C == 0
    oshape = (M, NB * BS_R)
    # import pdb; pdb.set_trace()
    assert data.dtype in ("float32", "uint16"), data.dtype
    assert weight_indices.dtype in ("int32", "uint16"), weight_indices.dtype
    assert weight_indptr.dtype in ("int32", "uint16"), weight_indptr.dtype
    assert weight_data.dtype in ("float32", "uint16", "int8"), weight_data.dtype
    NUM_AXIS = 4
    specify_range(cfg, 'axis_', NUM_AXIS)
    cfg.define_knob('rfactor_bs_c', [False, True])
    cfg.define_knob('align_data', [False, True] if len(data.shape) == 2 and BS_C > 1 and K > BS_C else [False])
    cfg.define_knob('vectorize_axis', range(-1, NUM_AXIS, 1))
    cfg.define_knob('parallel_axis', [False, True])
    cfg.define_knob('unroll_axis', range(-1, NUM_AXIS, 1))
    '''
    cfg.define_knob('rfactor_bs_c', [False])
    cfg.define_knob('align_data', [False])
    cfg.define_knob('vectorize_axis', [-1])
    cfg.define_knob('parallel_axis', [-1])
    '''
    if cfg['align_data'].val and data_layout == "NI":
        X = tvm.compute((M, K // BS_C, BS_C), lambda m, ko, ki: data[m, ko * BS_C + ki])
    else:
        X = data
    cfg.add_flop(2 * NUM * BS_C * BS_R * M)
    # bs_r = tvm.reduce_axis((0, weight_data.shape[1]), name="bs_r")
    bs_c = tvm.reduce_axis((0, BS_C), name="bs_c")
    def f(i, nb, r):
        # import pdb; pdb.set_trace()
        mul_elem = 2 if EVEN_ENTRIES else 1
        row_start = mul_elem * weight_indptr[nb].astype("int32")
        row_end = mul_elem * weight_indptr[nb + 1].astype("int32")
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        weight_val = weight_data[elem, r, bs_c]
        if weight_data.dtype == "uint16":
            weight_val = tvm_from_bf16(weight_val)
        else:
            weight_val = weight_val.astype("float32")
        if data_layout == "IN":
            x_val = X[weight_indices[elem] * BS_C + bs_c, i]
        elif cfg['align_data'].val:
            x_val = X[i, weight_indices[elem], bs_c]
        else:
            x_val = X[i, weight_indices[elem] * BS_C + bs_c]
        if data.dtype == "uint16":
            x_val = tvm_from_bf16(x_val)
        else:
            x_val = x_val.astype("float32")
        return tvm.sum(weight_val * x_val, axis=[elem_idx, bs_c])
    Y = tvm.compute((M, NB, BS_R), f,
        name="sdense_block",
        tag = "sdense_block")
    O = tvm.compute(oshape, lambda m, n: Y[m, n // BS_R, n % BS_R],
        name="sdense", tag="sdense")
    if cfg.is_fallback:
        _default_sdense_config(cfg, M, K, NK, NUM, BS_R, BS_C, NB, NUM_AXIS, data_layout, kernel_layout)
    return O


def _default_sdense_config(cfg, M, K, NK, NUM, BS_R, BS_C, NB, NUM_AXIS, data_layout, kernel_layout):
    # import pdb; pdb.set_trace()
    cfg["align_data"] = OtherOptionEntity(False)
    cfg["rfactor_bs_c"] = OtherOptionEntity(False)
    for i in range(NUM_AXIS):
        cfg["axis_" + str(NUM_AXIS) + "_" + str(i)] = OtherOptionEntity(i)
    cfg["vectorize_axis"] = OtherOptionEntity(-1)
    cfg["parallel_axis"] = OtherOptionEntity(-1)
    cfg["unroll_axis"] = OtherOptionEntity(-1)
    if M == 1 and BS_R == 1 and BS_C >= 16:
        cfg["align_data"] = OtherOptionEntity(False)
        cfg["rfactor_bs_c"] = OtherOptionEntity(True)
        for i in range(NUM_AXIS):
            cfg["axis_" + str(NUM_AXIS) + "_" + str(i)] = OtherOptionEntity(i)
        cfg["vectorize_axis"] = OtherOptionEntity(-1)
        cfg["parallel_axis"] = OtherOptionEntity(-1)
        cfg["unroll_axis"] = OtherOptionEntity(-1)
    elif M == 1 and BS_R >= 16 and BS_C == 1:
        cfg["align_data"] = OtherOptionEntity(False)
        cfg["rfactor_bs_c"] = OtherOptionEntity(False)
        for i in range(NUM_AXIS):
            cfg["axis_" + str(NUM_AXIS) + "_" + str(i)] = OtherOptionEntity(i)
        cfg["vectorize_axis"] = OtherOptionEntity(2)
        cfg["parallel_axis"] = OtherOptionEntity(-1)
        cfg["unroll_axis"] = OtherOptionEntity(-1)
    elif M >= 8 and BS_R >= 16 and BS_C == 1:
        cfg["align_data"] = OtherOptionEntity(False)
        cfg["rfactor_bs_c"] = OtherOptionEntity(False)
        cfg["axis_4_0"] = OtherOptionEntity(0)
        cfg["axis_4_1"] = OtherOptionEntity(1)
        cfg["axis_4_2"] = OtherOptionEntity(2)
        cfg["axis_4_3"] = OtherOptionEntity(2)
        cfg["vectorize_axis"] = OtherOptionEntity(3)
        cfg["parallel_axis"] = OtherOptionEntity(-1)
        cfg["unroll_axis"] = OtherOptionEntity(-1)




@autotvm.register_topi_schedule(generic.schedule_sdense, 'cpu', ['direct'])
def schedule_sdense(cfg, outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'sdense':
            schedule_sdense_sch(s, cfg, op, outs[0].op)
    traverse_inline(s, outs[0].op, _callback)
    return s


def reorder_axes(cfg, prefix, axes):
    # import pdb; pdb.set_trace()
    num = len(axes)
    new_axes = [None] * num
    for i in range(num-1, -1, -1):
        name = prefix + str(num) + "_" + str(i)
        assert name in cfg, name + " doesn't exist in cfg"
        idx = cfg[name].val
        count = 0
        for j in range(num):
            if new_axes[j] is None:
                if count == idx:
                    new_axes[j] = axes[i]
                    break
                else:
                    count = count + 1
        assert count == idx
    for i in range(num):
        assert new_axes[i] is not None, "Reordered axes have None"

    # import pdb; pdb.set_trace()
    return new_axes


def schedule_sdense_sch(s, cfg, op, out):
    # import pdb; pdb.set_trace()
    num_threads = int(os.environ["TVM_NUM_THREADS"]) if "TVM_NUM_THREADS" in os.environ else 1
    (mo, no) = s[op].op.axis
    op_o = op.output(0)
    Y = op.input_tensors[0]
    Y_op = s[Y].op
    assert Y_op.tag == "sdense_block"
    (i, nb, r) = Y_op.axis
    (elem_idx, bs_c) = Y_op.reduce_axis
    I = get_const_int(op_o.shape[0])
    BS_C = get_const_int(bs_c.dom.extent)
    BS_R = get_const_int(Y.shape[2])
    BF = None
    '''
    s[Y].prefetch(Y_op.input_tensors[1], elem_idx, 2)
    s[Y].prefetch(Y_op.input_tensors[2], elem_idx, 2)
    s[Y].prefetch(Y_op.input_tensors[3], elem_idx, 2)
    '''
    if cfg['rfactor_bs_c'].val is True:
        # import pdb; pdb.set_trace()
        BF = s.rfactor(Y, bs_c, factor_axis=2)
        (fi, fnb, fbs_c, fr) = BF.op.axis
        (felem_idx,) = BF.op.reduce_axis
        axes = [felem_idx, fbs_c, fr, fi]
        new_axis = reorder_axes(cfg, "axis_", axes)
        s[BF].reorder(fnb, *new_axis)
        if cfg["vectorize_axis"].val >= 0 and \
                new_axis[cfg["vectorize_axis"].val] != felem_idx:
            s[BF].vectorize(new_axis[cfg["vectorize_axis"].val])
        if cfg["unroll_axis"].val >= 0 and \
                new_axis[cfg["unroll_axis"].val] != elem_idx:
            s[BF].unroll(new_axis[cfg["unroll_axis"].val])
    else:
        # import pdb; pdb.set_trace()
        io = None
        if I > 10240:
            (io, ii) = s[Y].split(i,32)
        else:
            ii = i
        axes = [elem_idx, bs_c, r, i]
        new_axis = reorder_axes(cfg, "axis_", axes)
        raxes = [nb] + new_axis
        if io is not None:
            raxes = [io] + raxes
        s[Y].reorder(*raxes)
        if cfg["vectorize_axis"].val >= 0 and \
                new_axis[cfg["vectorize_axis"].val] != elem_idx:
            s[Y].vectorize(new_axis[cfg["vectorize_axis"].val])
        if cfg["parallel_axis"].val >= 0 and \
                new_axis[cfg["parallel_axis"].val] != elem_idx:
            s[Y].parallel(new_axis[cfg["parallel_axis"].val])
        if cfg["unroll_axis"].val >= 0 and \
                new_axis[cfg["unroll_axis"].val] != elem_idx:
            s[Y].unroll(new_axis[cfg["unroll_axis"].val])

    if op != out:
        # fix it for now
        s[Y].compute_at(s[op], no)
        s[op].vectorize(no)
        d = s[out].op.axis
        split_axis = d[0]
        if len(d) > 1:
            d0 = get_const_int(d[0].dom.extent)
            d1 = get_const_int(d[1].dom.extent)
            if d0 == 1 or ((d0 < 32) and (d1 > 32)):
                split_axis = d[1]
        (yo, yi) = s[out].split(split_axis, 32)
        # s[op_o].compute_at(s[out], yo)
        # s[Y].compute_at(s[out], yo)
        s[out].vectorize(yi)
        # s[out].unroll(yo)
        if num_threads > 1:
            s[out].parallel(yo)
        # s[out].unroll(yi)
        s[op_o].compute_at(s[out], yo)
        s[Y].compute_at(s[out], yo)
    else:
        (noo, noi) = s[op].split(no, BS_R)
        s[Y].compute_at(s[op], noo)
        # s[out].unroll(mo)
        # s[op].reorder(no, mo)
        pass

@generic.schedule_sparse_dense.register(["cpu"])
def schedule_sparse_dense(outs):
    s = tvm.create_schedule([x.op for x in outs])
    def callback(op):
        if op.tag == "sparse_dense_csrmv" and op != outs[0].op:
            (_, vi) = s[op].op.axis
            s[op].vectorize(vi)
            (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
            s[op].compute_at(s[outs[0]], yo)
            s[outs[0].op].vectorize(yi)
        if op.tag == "sparse_dense_bsrmv":
            Y_bsrmv = op.input_tensors[0]
            assert Y_bsrmv.op.tag == "sparse_dense_bsrmv_block"
            Y_reshape = op
            (m, nb, br) = s[Y_bsrmv].op.axis
            BS_R = get_const_int(br.dom.extent)
            (elem_idx, c) = s[Y_bsrmv].op.reduce_axis
            s[Y_bsrmv].reorder(nb, m, elem_idx, br, c)
            s[Y_bsrmv].vectorize(br)
            (mo, no) = s[Y_reshape].op.axis
            (noo, noi) = s[Y_reshape].split(no, BS_R)
            s[Y_reshape].unroll(noo)
            s[Y_bsrmv].compute_at(s[Y_reshape], noi)
            s[Y_reshape].vectorize(noi)
            s[Y_reshape].unroll(mo)
            num_threads = int(os.environ["TVM_NUM_THREADS"]) if "TVM_NUM_THREADS" in os.environ else 1
            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[Y_reshape].compute_at(s[outs[0]], yo)
                if num_threads > 1:
                    s[outs[0].op].parallel(yo)
                else:
                    s[outs[0].op].unroll(yo)
                s[outs[0].op].vectorize(yi)
            else:
                if num_threads > 1:
                    s[Y_reshape].parallel(noo)
                else:
                    s[Y_reshape].unroll(noo)

    traverse_inline(s, outs[0].op, callback)
    return s


@generic.schedule_sparse_dense2.register(["cpu"])
def schedule_sparse_dense2(outs):
    s = tvm.create_schedule([x.op for x in outs])
    # import pdb; pdb.set_trace()
    if "sparse_dense2" not in outs[0].op.tag:
        # import pdb; pdb.set_trace()
        def callback(op):
            # import pdb; pdb.set_trace()
            if "sparse_dense2" in op.tag:
                (n, vi) = s[op].op.axis
                # k = s[op].op.reduce_axis
                s[op].vectorize(vi)
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 16)
                s[op].compute_at(s[outs[0]], yo)
                s[outs[0].op].vectorize(yi)
                # s[op].parallel(n)
        traverse_inline(s, outs[0].op, callback)
    # import pdb; pdb.set_trace()
    # C = outs[0]
    # A1, A2, A3, A4 = C.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, C], simple_mode=True))
    # C = outs[0]
    # A, B = outs[0].op.input_tensors
    # A1, A2, A3, A4 = A.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, B, C], simple_mode=True))
    return s


@generic.schedule_sparse_dense_kmnk.register(["cpu"])
def schedule_sparse_dense_kmnk(outs):
    s = tvm.create_schedule([x.op for x in outs])
    # import pdb; pdb.set_trace()
    def callback(op):
        # import pdb; pdb.set_trace()
        if "sparse_dense_kmnk" in op.tag:
            '''
            (n, vi) = s[op].op.axis
            (elem_idx, bs_c) = s[op].op.reduce_axis
            s[op].unroll(bs_c)
            s[op].vectorize(vi)
            # (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 8)
            # s[op].compute_at(s[outs[0]], yo)
            # s[outs[0].op].vectorize(yi)
            # s[op].parallel(n)
            '''

            #import pdb; pdb.set_trace()
            op_o = op.output(0)
            I = get_const_int(op_o.shape[1])
            Y = op.input_tensors[0]
            Y_op = s[Y].op
            assert Y_op.tag == "sparse_dense_kmnk_block"
            (nb, r, i) = Y_op.axis
            (elem_idx, bs_c) = Y_op.reduce_axis
            BS_R = get_const_int(Y.shape[1])
            # (nbo, nbi) = s[Y_op].split(nb, 8)
            # CC = s.cache_write(Y, 'global')
            # s[CC].compute_at(s[Y], nb)
            s[Y_op].reorder(nb, elem_idx, bs_c, r, i)
            BS_C = get_const_int(bs_c.dom.extent)
            BF = None
            if (I >= BS_C) and (I >= BS_R):
                # bsci = s[Y_op].fuse(bs_c, i)
                s[Y_op].vectorize(i)
            else:
                bsc_axis = 1
                if BS_C > BS_R and BS_C > I:
                    bsc_axis = 3
                elif BS_C > BS_R:
                    bsc_axis = 2
                else:
                    bsc_axis = 1

                BF = s.rfactor(Y, bs_c, factor_axis=bsc_axis)
                (fnb, x2, x1, x0) = BF.op.axis
                (felem_idx,) = BF.op.reduce_axis
                lx0 = get_const_int(x0.dom.extent)
                lx1 = get_const_int(x1.dom.extent)
                lx2 = get_const_int(x2.dom.extent)
                if lx0 >= lx1 and lx0 >= lx2:
                    xx0 = x0
                    if lx1 >= lx2:
                        xx1 = x1
                        xx2 = x2
                    else:
                        xx1 = x2
                        xx2 = x1
                elif lx1 >= lx2:
                    xx0 = x1
                    if lx0 >= lx2:
                        xx1 = x0
                        xx2 = x2
                    else:
                        xx1 = x2
                        xx2 = x0
                else:
                    xx0 = x2
                    if lx0 >= lx1:
                        xx1 = x0
                        xx2 = x1
                    else:
                        xx1 = x1
                        xx2 = x0
                lxx0 = get_const_int(xx0.dom.extent)
                if lxx0 >= 16:
                    # xx0 has enough parallelism
                    # for m = 2, bs_r = 16, bs_c = 1
                    s[BF].reorder(fnb, xx2, xx1, felem_idx, xx0)
                    s[BF].vectorize(xx0)
                else:
                    # for m = 2, bs_r = 8, bs_c = 1
                    s[BF].reorder(fnb, xx2, felem_idx, xx1, xx0)
                # s[BF].vectorize(xx0)

                '''
                do not fuse explicitly, rely on compiler to do it.
                fri = s[BF].fuse(fbsc, fi)
                frri = s[BF].fuse(fri, fr)
                s[BF].vectorize(frri)
                '''
                # s[BF].vectorize(fri)
                # BS_R = get_const_int(r.dom.extent)
                # (n, m) = op.axis
                # (no, ni) = s[op].split(n, 8)
                # M = get_const_int(m.dom.extent)
                ### s[BF].compute_at(s[Y], s[Y].op.axis[0])
                # import pdb; pdb.set_trace()
                '''
                FR = get_const_int(fr.dom.extent)
                FBSC = get_const_int(fbsc.dom.extent)
                if FR >= FBSC:
                    s[BF].vectorize(fr)
                else:
                    s[BF].vectorize(fbsc)
                '''
            # (n, m) = op.axis
            op_o = op.output(0)
            # import pdb; pdb.set_trace()
            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[0], 32)
                s[op_o].compute_at(s[outs[0]], yo)
                s[Y].compute_at(s[outs[0]], yo)
                if BF is not None:
                    s[BF].compute_at(s[outs[0]], yo)
                s[outs[0].op].vectorize(yi)

    traverse_inline(s, outs[0].op, callback)
    # import pdb; pdb.set_trace()
    # C = outs[0]
    # A1, A2, A3, A4 = C.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, C], simple_mode=True))
    # C = outs[0]
    # A, B = outs[0].op.input_tensors
    # A1, A2, A3, A4 = A.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, B, C], simple_mode=True))
    return s




@generic.schedule_sparse_dense_mknk.register(["cpu"])
def schedule_sparse_dense_mknk(outs):
    s = tvm.create_schedule([x.op for x in outs])
    def callback(op):
        if "sparse_dense_mknk" in op.tag:
            # import pdb; pdb.set_trace()
            op_o = op.output(0)
            Y = op.input_tensors[0]
            Y_op = s[Y].op
            assert Y_op.tag == "sparse_dense_kmnk_block"
            (i, nb, r) = Y_op.axis
            (elem_idx, bs_c) = Y_op.reduce_axis
            I = get_const_int(op_o.shape[0])
            BS_C = get_const_int(bs_c.dom.extent)
            BS_R = get_const_int(Y.shape[2])
            s[Y_op].reorder(nb, elem_idx, bs_c, r, i)

            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[0], 32)
                s[op_o].compute_at(s[outs[0]], yo)
                s[Y].compute_at(s[outs[0]], yo)
                s[outs[0].op].vectorize(yi)

    traverse_inline(s, outs[0].op, callback)
    # import pdb; pdb.set_trace()
    # C = outs[0]
    # A1, A2, A3, A4 = C.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, C], simple_mode=True))
    # C = outs[0]
    # A, B = outs[0].op.input_tensors
    # A1, A2, A3, A4 = A.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, B, C], simple_mode=True))

    return s


@autotvm.register_topi_compute(nn.grucell, 'cpu', ['direct'])
def grucell(cfg, input, tw_x, tb_x, tw_z, tb_z, tw_in, tb_in, tw_hn, tb_hn,
                 data_layout="NI", kernel_layout="OI", out_layout=""):
    (BATCH, XH) = topi.util.get_const_tuple(input.shape)
    (H, X) = topi.util.get_const_tuple(tw_in.shape)
    assert XH == X + H
    (tmp,) = topi.util.get_const_tuple(tb_in.shape)
    assert tmp == H
    (htmp, xtmp) = topi.util.get_const_tuple(tw_hn.shape)
    assert xtmp == X
    assert htmp == H
    (tmp,) = topi.util.get_const_tuple(tb_hn.shape)
    assert tmp == H

    (htmp, xhtmp) = topi.util.get_const_tuple(tw_x.shape)
    assert xhtmp == XH
    assert htmp == H
    (tmp,) = topi.util.get_const_tuple(tb_x.shape)
    assert tmp == H
    (htmp, xhtmp) = topi.util.get_const_tuple(tw_z.shape)
    assert xhtmp == XH
    assert htmp == H
    (tmp,) = topi.util.get_const_tuple(tb_z.shape)
    assert tmp == H

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
        ee = 1.0 + x + x2 * 0.5658 + 0.143 * x2 * x2
        e = tvm.if_then_else(tvm.abs(ee) < 1e-8, 1e-8, ee)

        def sign(x):
            return tvm.if_then_else(v >= 0, 1, -1)

        return sign(v) * (e - 1 / e) / (e + 1 / e)

    oshape = (BATCH, H)
    def mul_x(b, h):
        r = tvm.reduce_axis((0, XH), name="xr")
        return tvm.sum(input[b, r] * tw_x[h, r], axis=[r])

    def mul_z(b, h):
        r = tvm.reduce_axis((0, XH), name="zr")
        return tvm.sum(input[b, r] * tw_z[h, r], axis=[r])

    def mul_in(b, h):
        r = tvm.reduce_axis((0, X), name="inr")
        return tvm.sum(input[b, r] * tw_in[h, r], axis=[r])

    def mul_hn(b, h):
        r = tvm.reduce_axis((0, H), name="hnr")
        return tvm.sum(input[b, X + r] * tw_hn[h, r], axis=[r])

    # import pdb; pdb.set_trace()
    R = tvm.compute(oshape, mul_x,
        name="grucell_r", tag="grucell_r")
    Z = tvm.compute(oshape, mul_z,
        name="grucell_z", tag="grucell_z")
    IN = tvm.compute(oshape, mul_in,
        name="grucell_in", tag="grucell_in")
    HN = tvm.compute(oshape, mul_hn,
        name="grucell_hn", tag="grucell_hn")

    def gru(b, h):
        rr = R[b, h]
        zz = Z[b, h]
        hhn = HN[b, h]
        iin = IN[b, h]
        r = approx_sigmoid(rr + tb_x[h])
        z = approx_sigmoid(zz + tb_z[h])
        hn = hhn + tb_hn[h]
        _in = iin + tb_in[h]
        n = approx_tanh(r * hn + _in)
        h_prime = (1 - z) * n + z * input[b, X + h]
        return h_prime

    return tvm.compute(oshape, gru,
        name="grucell", tag="grucell")

def schedule_grucell_default(s, cfg, op, out):
    # import pdb; pdb.set_trace()
    (b, h) = s[op].op.axis
    Z = op.input_tensors[0]
    R = op.input_tensors[2]
    HN = op.input_tensors[4]
    IN = op.input_tensors[6]
    (ho, hi) = s[op].split(h, 1)
    s[R].compute_at(s[op], hi)
    s[Z].compute_at(s[op], hi)
    s[IN].compute_at(s[op], hi)
    s[HN].compute_at(s[op], hi)

    def vec_reduction(Z):
        (zb, zh) = s[Z].op.axis
        (zr,) = s[Z].op.reduce_axis
        # import pdb; pdb.set_trace()
        (zro, zri) = s[Z].split(zr, 8)
        ZF = s.rfactor(Z, zri, factor_axis=2)
        (zfb, zfh, zfr) = s[ZF].op.axis
        # (zfro, zfri) = s[ZF].split(zfr, 8)
        s[ZF].vectorize(zfr)
        # axis is modified
        (zb, zh) = s[Z].op.axis
        s[ZF].compute_at(s[Z], zh)

    vec_reduction(Z)
    vec_reduction(R)
    vec_reduction(IN)
    vec_reduction(HN)
    # s[R].vectorize(zh)
    # s[R].reorder(zh, zb)


@autotvm.register_topi_schedule(generic.schedule_grucell, 'cpu', ['direct'])
def schedule_grucell(cfg, outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == 'grucell':
            schedule_grucell_default(s, cfg, op, outs[0].op)
    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_compute(nn.sgrucell, 'cpu', ['direct'])
def sgrucell(cfg, input, w_x_data, w_x_indices, w_x_indptr, b_x,
             w_z_data, w_z_indices, w_z_indptr, b_z,
             w_in_data, w_in_indices, w_in_indptr, b_in,
             w_hn_data, w_hn_indices, w_hn_indptr, b_hn,
             data_layout="NI", kernel_layout="OI", out_layout=""):
    # import pdb; pdb.set_trace()
    (BATCH, XH) = topi.util.get_const_tuple(input.shape)
    (H,) = topi.util.get_const_tuple(b_x.shape)
    (tmp, BS_R, BS_C) = topi.util.get_const_tuple(w_x_data.shape)
    (NB_plus_1, ) = topi.util.get_const_tuple(w_x_indptr.shape)
    NB = NB_plus_1 - 1
    X = XH - H
    assert H == NB * BS_R
    oshape = (BATCH, H)


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
        ee = 1.0 + x + x2 * 0.5658 + 0.143 * x2 * x2
        e = tvm.if_then_else(tvm.abs(ee) < 1e-8, 1e-8, ee)

        def sign(x):
            return tvm.if_then_else(v >= 0, 1, -1)

        return sign(v) * (e - 1 / e) / (e + 1 / e)

    def f(i, nb, r, **kwargs):
        bs_c = tvm.reduce_axis((0, BS_C), name="bs_c")
        weight_indptr = kwargs["weight_indptr"]
        weight_data = kwargs["weight_data"]
        weight_indices = kwargs["weight_indices"]
        input = kwargs["input"]
        input_offset = kwargs["input_offset"] if "input_offset" in kwargs else 0
        row_start = weight_indptr[nb]
        row_end = weight_indptr[nb + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        weight_val = weight_data[elem, r, bs_c]
        if weight_data.dtype == "uint16":
            weight_val = tvm_from_bf16(weight_val)
        else:
            weight_val = weight_val.astype("float32")
        x_val = input[i, input_offset + weight_indices[elem] * BS_C + bs_c]
        x_val = x_val.astype("float32")
        return tvm.sum(weight_val * x_val, axis=[elem_idx, bs_c])

    RB = tvm.compute((BATCH, NB, BS_R), f,
        name="sgrucell_r_block", tag="sgrucell_r_block",
        fcompute_kwargs={"input": input, "weight_data": w_x_data, "weight_indices": w_x_indices, "weight_indptr": w_x_indptr}
        )
    R = tvm.compute(oshape, lambda m, n: RB[m, n // BS_R, n % BS_R],
        name="sgrucell_r", tag="sgrucell_r")

    ZB = tvm.compute((BATCH, NB, BS_R), f,
        name="sgrucell_z_block", tag="sgrucell_z_block",
        fcompute_kwargs={"input": input, "weight_data": w_z_data, "weight_indices": w_z_indices, "weight_indptr": w_z_indptr}
        )
    Z = tvm.compute(oshape, lambda m, n: ZB[m, n // BS_R, n % BS_R],
        name="sgrucell_z", tag="sgrucell_z")

    INB = tvm.compute((BATCH, NB, BS_R), f,
        name="sgrucell_in_block", tag="sgrucell_in_block",
        fcompute_kwargs={"input": input, "weight_data": w_in_data, "weight_indices": w_in_indices, "weight_indptr": w_in_indptr}
        )
    IN = tvm.compute(oshape, lambda m, n: INB[m, n // BS_R, n % BS_R],
        name="sgrucell_in", tag="sgrucell_in")

    HNB = tvm.compute((BATCH, NB, BS_R), f,
        name="sgrucell_hn_block", tag="sgrucell_hn_block",
        fcompute_kwargs={"input": input, "input_offset": X, "weight_data": w_hn_data, "weight_indices": w_hn_indices, "weight_indptr": w_hn_indptr}
        )
    HN = tvm.compute(oshape, lambda m, n: HNB[m, n // BS_R, n % BS_R],
        name="sgrucell_hn", tag="sgrucell_hn")

    def gru(b, h):
        rr = R[b, h]
        zz = Z[b, h]
        hhn = HN[b, h]
        iin = IN[b, h]
        r = approx_sigmoid(rr + b_x[h])
        z = approx_sigmoid(zz + b_z[h])
        hn = hhn + b_hn[h]
        _in = iin + b_in[h]
        n = approx_tanh(r * hn + _in)
        h_prime = (1 - z) * n + z * input[b, X + h]
        return h_prime

    return tvm.compute(oshape, gru,
        name="sgrucell", tag="sgrucell")


def schedule_sgrucell_default(s, cfg, op, out):
    # import pdb; pdb.set_trace()
    (b, h) = s[op].op.axis
    (ho, hi) = s[op].split(h, 16)
    def schedule_block(Z):
        BLOCK = Z.op.input_tensors[0]
        (i, nb, r) = s[BLOCK].op.axis
        (elem_idx, c) = s[BLOCK].op.reduce_axis
        BS_R = get_const_int(r.dom.extent)
        BS_C = get_const_int(c.dom.extent)
        if BS_R > 1:
            s[Z].compute_at(s[op], ho)
            s[BLOCK].compute_at(s[op], ho)
            s[BLOCK].vectorize(r)
        elif BS_C > 0:
            s[Z].compute_at(s[op], hi)
            s[BLOCK].compute_at(s[op], hi)
            BF = s.rfactor(BLOCK, c, factor_axis=3)
            (i, nb, r) = s[BLOCK].op.axis
            s[BF].compute_at(s[BLOCK], r)
            (bi, bnb, br, bc) = s[BF].op.axis
            (belem_idx,) = s[BF].op.reduce_axis
            s[BF].reorder(bi, bnb, belem_idx, br, bc)
            s[BF].vectorize(bc)

    Z = op.input_tensors[0]
    R = op.input_tensors[2]
    HN = op.input_tensors[4]
    IN = op.input_tensors[6]
    # import pdb; pdb.set_trace()
    schedule_block(Z)
    schedule_block(R)
    schedule_block(HN)
    schedule_block(IN)


@autotvm.register_topi_schedule(generic.schedule_sgrucell, 'cpu', ['direct'])
def schedule_sgrucell(cfg, outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == 'sgrucell':
            schedule_sgrucell_default(s, cfg, op, outs[0].op)
    traverse_inline(s, outs[0].op, _callback)
    return s
