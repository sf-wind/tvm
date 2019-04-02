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
            s[Y_reshape].unroll(noo)
            s[Y_bsrmv].compute_at(s[Y_reshape], noi)
            s[Y_reshape].vectorize(noi)
            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[Y_reshape].compute_at(s[outs[0]], yo)
                s[outs[0].op].parallel(yo)
                s[outs[0].op].vectorize(yi)
            else:
                s[Y_reshape].parallel(noo)

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
    if "sparse_dense_kmnk" not in outs[0].op.tag:
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
    # import pdb; pdb.set_trace()
    if "sparse_dense_mknk" not in outs[0].op.tag:
        # import pdb; pdb.set_trace()
        def callback(op):
            # import pdb; pdb.set_trace()
            if "sparse_dense_mknk" in op.tag:
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
