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
                (n, vi) = s[op].op.axis
                (elem_idx, bs_c) = s[op].op.reduce_axis
                # s[op].unroll(sidx)
                # s[op].vectorize(vi)
                # (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 8)
                # s[op].compute_at(s[outs[0]], yo)
                # s[outs[0].op].vectorize(yi)
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
