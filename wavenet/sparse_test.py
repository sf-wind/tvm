import tvm
import topi
from tvm import relay
from tvm import autotvm
import numpy as np

import tvm.contrib.debugger.debug_runtime as graph_runtime
# import tvm.contrib.graph_runtime as graph_runtime
import torch
import logging
import collections
import netron
import tempfile
import logging
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--structure", type=int, default=0)
args = parser.parse_args()

if args.num_threads > 0:
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

if args.debug:
    from tvm.contrib.debugger import debug_runtime as graph_runtime
else:
    import tvm.contrib.graph_runtime as graph_runtime

logging.basicConfig(level=logging.DEBUG)
dtype = "float32"
itype = 'int32'

M = 8
N = 3072
K = 1024
structure = args.structure if args.structure > 0 else 1
ctx = tvm.context("llvm -mcpu=core-avx2", 0)

density = 0.04
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
mask = np.random.choice([0, 1], size=(N, K // structure), p=[1-density, density])
rmask = np.repeat(mask, structure, axis=1)
bb = (np.random.rand(N, K).astype(dtype) * rmask).astype(dtype)
b = tvm.nd.array(bb, ctx)
num_nonzeros = np.count_nonzero(b.asnumpy())
print("non zeros: {}".format(num_nonzeros))

# import pdb; pdb.set_trace()
# baseline
answer = np.dot(a.asnumpy(), np.transpose(b.asnumpy()))

import collections
CSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'density'])

def convert_sparse(v, mask, structure, use_structure):
    if not use_structure:
        mask = np.repeat(mask, structure, axis=1)
        structure = 1
    source_array = v.reshape(v.shape[:-1] + (v.shape[-1] // structure,) + (structure,))
    ridx, cidx = np.nonzero(mask)
    data = source_array[ridx, cidx]
    if not use_structure:
        assert structure == 1
        data = np.squeeze(data, axis=1)
    data = tvm.nd.array(data, ctx)
    indices = np.nonzero(mask)[1].astype(itype)
    indices = tvm.nd.array(indices, ctx)
    indptr = [0]+np.apply_along_axis(np.count_nonzero, axis=1, arr=mask).tolist()
    indptr = np.cumsum(np.array(indptr, itype)).astype(itype)
    indptr = tvm.nd.array(indptr, ctx)
    return (data, indices, indptr)

def to_sparse(v, density=0.04):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = num_nonzeros
    v_data = relay.var(name + "_data", shape=(nnz,), dtype=dtype)
    v_indices = relay.var(name + "_indices", shape=(nnz,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N + 1,), dtype="int32")
    return CSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, density=density)

def instantiate(param, use_sparse):
    if isinstance(param, CSR):
        data, indices, indptr = convert_sparse(b.asnumpy(), mask, structure, use_sparse)
        return [
            (param.data.name_hint, data),
            (param.indices.name_hint, indices),
            (param.indptr.name_hint, indptr),
        ]
    else:
        return [(param.name_hint, tvm.nd.array(np.zeros(param.type_annotation.concrete_shape).astype(param.type_annotation.dtype), ctx))
        ]

x = relay.var("x", shape=[M, K], dtype=dtype)
fc_0_W = to_sparse(relay.var("fc_0_W", shape=(N, K), dtype=dtype))
zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = relay.add(relay.nn.sparse_dense(x, fc_0_W), zero)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
relay.ir_pass.infer_type(func)

skl_target = tvm.target.create('llvm -mcpu=core-avx2 -target=x86_64-linux-gnu')

param_vars = [
    fc_0_W, zero
]
input_vars = [x]

params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param, False)])
inputs = collections.OrderedDict(
    {
        "x": a
    })

# import pdb; pdb.set_trace()

with relay.build_config(opt_level=3):
    func = relay.optimize(func, target=skl_target, params=params)
    # print(func.astext(show_meta_data=False))
    # import pdb; pdb.set_trace()
    func = relay.ir_pass.infer_type(func)
    graph, lib, new_params = relay.build_module.build(
        func, target=skl_target,  params=params)
    '''
    print(lib.get_source())
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

ftimer = module.module.time_evaluator("run", ctx, 100)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)


print("sparse_dense2")
# import pdb; pdb.set_trace()
x = relay.var("x", shape=[K, M], dtype=dtype)
fc_0_W = to_sparse(relay.var("fc_0_W", shape=(N, K), dtype=dtype))
zero = relay.var("zero", shape=(N, M), dtype=dtype)
outputs = relay.add(relay.nn.sparse_dense2(x, fc_0_W), zero)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))
# import pdb; pdb.set_trace()
skl_target = tvm.target.create('llvm -mcpu=core-avx2 -target=x86_64-linux-gnu')

param_vars = [
    fc_0_W, zero
]
input_vars = [x]

params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param, False)])
inputs = collections.OrderedDict(
    [(
        param.name_hint,
        tvm.nd.array(np.transpose(a.asnumpy()), ctx)
    ) for param in input_vars])
# import pdb; pdb.set_trace()
with relay.build_config(opt_level=3):
    func = relay.optimize(func, target=skl_target, params=params)
    # print(func.astext(show_meta_data=False))
    # import pdb; pdb.set_trace()
    func = relay.ir_pass.infer_type(func)
    graph, lib, new_params = relay.build_module.build(
        func, target=skl_target,  params=params)
    '''
    print(func.astext(show_meta_data=False))
    print(lib.get_source())
    print(lib.get_source('asm'))
    for (k, v) in params.items():
        print(k, v.shape)
    for (k, v) in new_params.items():
        print(k, v.shape)
    '''

# import pdb; pdb.set_trace()
r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty([N, M])
# import pdb; pdb.set_trace()
module.get_output(0, ro)
tvm.testing.assert_allclose(np.transpose(ro.asnumpy()), answer, rtol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 100)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)


print("sparse_dense_structure")
# import pdb; pdb.set_trace()
x = relay.var("x", shape=[K // structure, structure, M], dtype=dtype)
fc_0_W = to_sparse(relay.var("fc_0_W", shape=(N, K), dtype=dtype))
zero = relay.var("zero", shape=(N, M), dtype=dtype)
outputs = relay.add(relay.nn.sparse_dense_structure(x, fc_0_W), zero)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))
# import pdb; pdb.set_trace()
skl_target = tvm.target.create('llvm -mcpu=core-avx2 -target=x86_64-linux-gnu')

param_vars = [
    fc_0_W, zero
]
input_vars = [x]

params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param, True)])
inputs = collections.OrderedDict(
    [(
        param.name_hint,
        tvm.nd.array(np.transpose(a.asnumpy()).reshape((a.shape[1] // structure, structure, a.shape[0])), ctx)
    ) for param in input_vars])
# import pdb; pdb.set_trace()
with relay.build_config(opt_level=3):
    func = relay.optimize(func, target=skl_target, params=params)
    # print(func.astext(show_meta_data=False))
    # import pdb; pdb.set_trace()
    func = relay.ir_pass.infer_type(func)
    graph, lib, new_params = relay.build_module.build(
        func, target=skl_target,  params=params)
    print(func.astext(show_meta_data=False))

    print(lib.get_source())
    print(lib.get_source('asm'))
    for (k, v) in params.items():
        print(k, v.shape)
    for (k, v) in new_params.items():
        print(k, v.shape)


# import pdb; pdb.set_trace()
r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty([N, M])
# import pdb; pdb.set_trace()
module.get_output(0, ro)
tvm.testing.assert_allclose(np.transpose(ro.asnumpy()), answer, rtol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 100)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
