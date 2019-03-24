from .. import generic, nn, tag
from ..util import traverse_inline, get_const_int
import tvm

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
            s[Y_bsrmv].compute_at(s[Y_reshape], noi)
            s[Y_reshape].vectorize(noi)
            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[Y_reshape].compute_at(s[outs[0]], yo)
                s[outs[0].op].vectorize(yi)
    traverse_inline(s, outs[0].op, callback)
    # import pdb; pdb.set_trace()
    # C = outs[0]
    # A, B = outs[0].op.input_tensors
    # A1, A2, A3, A4 = A.op.input_tensors
    # print(tvm.lower(s, [A1, A2, A3, A4, B, C], simple_mode=True))

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
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 8)
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


@generic.schedule_sparse_dense_structure.register(["cpu"])
def schedule_sparse_dense_structure(outs):
    s = tvm.create_schedule([x.op for x in outs])
    # import pdb; pdb.set_trace()
    if "sparse_dense_structure" not in outs[0].op.tag:
        # import pdb; pdb.set_trace()
        def callback(op):
            # import pdb; pdb.set_trace()
            if "sparse_dense_structure" in op.tag:
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

                # import pdb; pdb.set_trace()
                Y = op.input_tensors[0]
                Y_op = s[Y].op
                assert Y_op.tag == "sparse_dense_structure_block"
                (nb, r, i) = Y_op.axis
                (elem_idx, bs_c) = Y_op.reduce_axis
                BS_R = get_const_int(Y.shape[1])
                # (nbo, nbi) = s[Y_op].split(nb, 8)
                # CC = s.cache_write(Y, 'global')
                # s[CC].compute_at(s[Y], nb)
                s[Y_op].reorder(nb, elem_idx, r, bs_c, i)
                BS_C = get_const_int(bs_c.dom.extent)
                I = get_const_int(i.dom.extent)
                if (I >= BS_C) and (I >= BS_R):
                    # bsci = s[Y_op].fuse(bs_c, i)
                    s[Y_op].vectorize(i)
                else:
                    BF = s.rfactor(Y, bs_c, factor_axis=2)
                    (fnb, fr, fbsc, fi) = BF.op.axis
                    (felem_idx,) = BF.op.reduce_axis
                    s[BF].reorder(fnb, felem_idx, fr, fbsc, fi)
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
                    s[BF].compute_at(s[Y], s[Y].op.axis[0])
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




@generic.schedule_sparse_dense_structure2.register(["cpu"])
def schedule_sparse_dense_structure2(outs):
    s = tvm.create_schedule([x.op for x in outs])
    # import pdb; pdb.set_trace()
    if "sparse_dense_structure2" not in outs[0].op.tag:
        # import pdb; pdb.set_trace()
        def callback(op):
            # import pdb; pdb.set_trace()
            if "sparse_dense_structure2" in op.tag:
                (n, vi) = s[op].op.axis
                (elem_idx, sidx) = s[op].op.reduce_axis
                # s[op].unroll(sidx)
                s[op].vectorize(vi)
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 8)
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

    traverse_inline(s, outs[0].op, callback)
    return s
