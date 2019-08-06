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
import tempfile
import logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--bs_r", type=int, default=0)
parser.add_argument("--bs_c", type=int, default=0)
parser.add_argument("--default_schedule", action="store_true")
parser.add_argument("--tuner", type=str, default="xgboost",
                    choices=["ga", "xgboost"])
parser.add_argument("--n_parallel", type=int, default=4)
parser.add_argument("--target", type=str, default="core-avx2",
                    choices=["core-avx2", "skylake-avx512"])
args = parser.parse_args()

if args.num_threads > 0:
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
os.environ["TVM_NUM_THREADS"] = str(1)

if args.debug:
    from tvm.contrib.debugger import debug_runtime as graph_runtime
else:
    import tvm.contrib.graph_runtime as graph_runtime

logging.basicConfig(level=logging.DEBUG)
dtype = "float32"
itype = 'int32'

context = "llvm -mcpu=" + args.target
target_platform = tvm.target.create(context)
ctx = tvm.context(context, 0)

log_filename = "autotvm.best.log"

M = 32 # args.m if args.m > 0 else 1
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

density = 1
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
mask = np.random.choice([0, 1], size=(N //BS_R, K // BS_C), p=[1-density, density])
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


def to_csr(v, density):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = num_nonzeros
    v_data = relay.var(name + "_data", shape=(nnz,), dtype=dtype)
    v_indices = relay.var(name + "_indices", shape=(nnz,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N + 1,), dtype="int32")
    return CSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, density=density)

def to_bsr(v, density):
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

tune_ops = (relay.op.nn.dense, relay.op.nn.sparse_dense)

def tune(func, params, target_platform):
    with relay.build_config(opt_level=2):
        # func = relay.optimize(func, target=target_platform, params=params)
        # print(func.astext(show_meta_data=False))
        # import pdb; pdb.set_trace()
        tasks = autotvm.task.extract_from_program(
            func, target=target_platform, params=params, ops=tune_ops)
        for i, task in enumerate(tasks):
            print(task)
            tsk = autotvm.task.create(task.name,
                                      args=task.args,
                                      target=target_platform,
                                      template_key="direct")
            tsk.workload = task.workload

            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            if args.tuner == "xgboost":
                tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
            else:
                tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
            n_trial = 1000
            early_stopping = 200
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=100, n_parallel=args.n_parallel),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            )
            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename)
                ]
            )


def build_graph_base(func, params, target_platform):
    with relay.build_config(opt_level=3):
        # func = relay.optimize(func, target=target_platform, params=params)
        # print(func.astext(show_meta_data=False))
        # func = relay.analysis.infer_type(func)
        graph, lib, new_params = relay.build_module.build(
            func, target=target_platform,  params=params)
        # print(lib.get_source('asm'))
        # print(lib.get_source())
        print(func.astext(show_meta_data=False))
        '''
        print(lib.get_source())
        print(lib.get_source('asm'))
        for (k, v) in params.items():
            print(k, v.shape)
        for (k, v) in new_params.items():
            print(k, v.shape)
        '''
        return graph, lib, new_params


def build_graph(func, params, target_platform):
    if args.default_schedule:
        (graph, lib, new_params) = build_graph_base(func, params,
                                                    target_platform)
    else:
        with autotvm.apply_history_best(log_filename):
            (graph, lib, new_params) = build_graph_base(func, params,
                                                        target_platform)
    return graph, lib, new_params


print("\ndense: ")
x = relay.var("x", shape=[M, K], dtype=dtype)
fc_0_W = relay.var("fc_0_W", shape=(N, K), dtype=dtype)
zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = relay.add(relay.nn.dense(x, fc_0_W), zero)

func = relay.Function(relay.analysis.free_vars(outputs), outputs)
# relay.frontend.infer_type(func)

param_vars = [
    fc_0_W, zero
]
input_vars = [x]

params = collections.OrderedDict({
    "zero": tvm.nd.array(np.zeros(zero.type_annotation.concrete_shape).astype(zero.type_annotation.dtype), ctx),
    "fc_0_W": b,
})
inputs = collections.OrderedDict(
    {
        "x": a
    })

graph, lib, new_params = build_graph(func, params, target_platform)

r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty([M, N])
module.get_output(0, ro)
tvm.testing.assert_allclose(ro.asnumpy(), answer, rtol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 100)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
