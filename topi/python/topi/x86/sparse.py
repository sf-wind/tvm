from .. import generic, nn, tag
from ..util import traverse_inline
import tvm

@generic.schedule_sparse_dense.register(["cpu"])
def schedule_sparse_dense(outs):
    s = tvm.create_schedule([x.op for x in outs])
    if "sparse_dense" not in outs[0].op.tag:
        # import pdb; pdb.set_trace()
        def callback(op):
            if "sparse_dense" in op.tag:
                (_, vi) = s[op].op.axis
                s[op].vectorize(vi)
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[op].compute_at(s[outs[0]], yo)
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
    return s
