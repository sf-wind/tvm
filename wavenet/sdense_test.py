import tvm
import topi
from tvm import relay
from tvm import autotvm
import numpy as np
from scipy.sparse import bsr_matrix

import tvm.contrib.debugger.debug_runtime as graph_runtime
# import tvm.contrib.graph_runtime as graph_runtime
import logging
import collections
import netron
import tempfile
import logging
import os
import argparse
import sys

sys.settrace

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--bs_r", type=int, default=0)
parser.add_argument("--bs_c", type=int, default=0)
parser.add_argument("--tuner", type=str, default="ga",
                    choices=["ga", "xgboost"])
args = parser.parse_args()

if args.num_threads > 0:
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

if args.debug:
    from tvm.contrib.debugger import debug_runtime as graph_runtime
else:
    import tvm.contrib.graph_runtime as graph_runtime

# logging.basicConfig(level=logging.DEBUG)
dtype = "float32"
itype = 'int32'

context = "llvm -mcpu=core-avx2"
skl_target = tvm.target.create(context)
ctx = tvm.context(context, 0)

M = args.m if args.m > 0 else 1
N = 3072
K = 1024
'''
N = 8
K = 8
'''
BS_R = args.bs_r if args.bs_r > 0 else 1
BS_C = args.bs_c if args.bs_c > 0 else 1

assert N % BS_R == 0
assert K % BS_C == 0

print("\n\nmatrix: [{}, {}] * [{}, {}], BS_R: {}, BS_C: {}".format(M, K, K, N, BS_R, BS_C))

density = 0.04
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
filename = str(N) + "_" + str(K) + "_" + str(BS_R) + "_" + str(BS_C) + ".npz"
if os.path.isfile("mask_data/" + filename):
    with open("mask_data/" + filename, "rb") as f:
        mask = np.load(f)
else:
    mask = np.random.choice([0, 1], size=(N //BS_R, K // BS_C), p=[1-density, density])
    with open("mask_data/" + filename, "wb") as f:
        np.save(f, mask)
mask = np.repeat(mask, BS_C, axis=1)
mask = np.repeat(mask, BS_R, axis=0)
bb = (np.random.rand(N, K).astype(dtype) * mask).astype(dtype)
bsr_m = bsr_matrix(bb, blocksize=(BS_R, BS_C))
csr_m = bsr_m.tocsr(True)
b = tvm.nd.array(bb, ctx)
num_nonzeros = np.count_nonzero(b.asnumpy())
print("non zeros: {}".format(num_nonzeros))

# baseline
answer = np.dot(a.asnumpy(), np.transpose(b.asnumpy()))

import collections
CSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'density'])
BSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'BS_R', 'BS_C', 'density'])


def to_csr(v, density=0.04):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = num_nonzeros
    v_data = relay.var(name + "_data", shape=(nnz,), dtype=dtype)
    v_indices = relay.var(name + "_indices", shape=(nnz,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N + 1,), dtype="int32")
    return CSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, density=density)

def to_bsr(v, density=0.04):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = num_nonzeros
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    v_data = relay.var(name + "_data", shape=(num_blocks, BS_R, BS_C), dtype=dtype)
    v_indices = relay.var(name + "_indices", shape=(num_blocks,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N // BS_R + 1,), dtype="int32")
    return BSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, BS_R=BS_R, BS_C=BS_C, density=density)


def instantiate(param):
    if isinstance(param, CSR):
        return [
            (param.data.name_hint, tvm.nd.array(csr_m.data.astype("float32"), ctx)),
            (param.indices.name_hint, tvm.nd.array(csr_m.indices.astype("int32"), ctx)),
            (param.indptr.name_hint, tvm.nd.array(csr_m.indptr.astype("int32"), ctx)),
        ]
    elif isinstance(param, BSR):
        return [
            (param.data.name_hint, tvm.nd.array(bsr_m.data.astype("float32"), ctx)),
            (param.indices.name_hint, tvm.nd.array(bsr_m.indices.astype("int32"), ctx)),
            (param.indptr.name_hint, tvm.nd.array(bsr_m.indptr.astype("int32"), ctx)),
        ]
    else:
        return [(param.name_hint, tvm.nd.array(np.zeros(param.type_annotation.concrete_shape).astype(param.type_annotation.dtype), ctx))
        ]


print("sdense")
x = relay.var("x", shape=[M, K], dtype=dtype)
fc_0_W = to_bsr(relay.var("fc_0_W", shape=(N, K), dtype=dtype))
zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = relay.add(relay.nn.sdense(x, fc_0_W), zero)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))

param_vars = [
    fc_0_W, zero
]
input_vars = [x]

params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param)])
inputs = collections.OrderedDict(
    [(
        param.name_hint,
        tvm.nd.array(a.asnumpy(), ctx)
    ) for param in input_vars])

def tune():
    global func
    with relay.build_config(opt_level=2):
        func = relay.optimize(func, target=skl_target, params=params)
        print(func.astext(show_meta_data=False))
        # import pdb; pdb.set_trace()
        tasks = autotvm.task.extract_from_program(
            func, target=skl_target, params=params, ops=(relay.op.nn.sdense,))
        for i, tsk in enumerate(tasks):
            print(tsk)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            if args.tuner == "xgboost":
                tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
            else:
                tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
            n_trial = 100
            early_stopping = 200
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            )
            log_filename = "synthesis_autotvm_skl.log"
            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename)
                ]
            )

if args.tune:
    tune()
    import sys
    sys.exit()


with autotvm.apply_history_best("synthesis_autotvm_skl.log"):
# if 1:
    with relay.build_config(opt_level=3):
        func = relay.optimize(func, target=skl_target, params=params)
        # print(func.astext(show_meta_data=False))
        func = relay.ir_pass.infer_type(func)
        graph, lib, new_params = relay.build_module.build(
            func, target=skl_target,  params=params)
        # print(func.astext(show_meta_data=False))
        # print(lib.get_source('asm'))
        # print(lib.get_source())
        '''
        print(lib.get_source('asm'))
        for (k, v) in params.items():
            print(k, v.shape)
        for (k, v) in new_params.items():
            print(k, v.shape)
        '''

r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty([M, N])
module.get_output(0, ro)
tvm.testing.assert_allclose(ro.asnumpy(), answer, rtol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 10)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
