import tvm
import topi
from tvm import relay
from tvm import autotvm
import numpy as np
from scipy.sparse import bsr_matrix
import torch
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
import time

sys.settrace

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--input_size", type=int, default=0)
parser.add_argument("--hidden_size", type=int, default=0)
parser.add_argument("--bs_r", type=int, default=0)
parser.add_argument("--bs_c", type=int, default=0)
parser.add_argument("--tuner", type=str, default="xgboost",
                    choices=["ga", "xgboost"])
parser.add_argument("--target", type=str, default="core-avx2",
                    choices=["core-avx2", "skylake-avx512"])
parser.add_argument("--default_schedule", action="store_true")
parser.add_argument("--wdtype", type=str, default="float32",
                    choices=["float32", "bfloat16"])
parser.add_argument("--witype", type=str, default="int32",
                    choices=["int32", "uint16"])
args = parser.parse_args()

if args.num_threads > 0:
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

if args.debug:
    from tvm.contrib.debugger import debug_runtime as graph_runtime
else:
    import tvm.contrib.graph_runtime as graph_runtime

logging.basicConfig(level=logging.DEBUG)

np.random.seed(int(time.clock()))
dtype = "float32"
wdtype = "uint16" if args.wdtype == "bfloat16" else "float32"
witype = "uint16" if args.witype == "uint16" else "int32"
itype = 'int32'

context = "llvm -mcpu=" + args.target
skl_target = tvm.target.create(context)
ctx = tvm.context(context, 0)

BATCH = 1
IS = args.input_size if args.input_size > 0 else 1024
HS = args.hidden_size if args.hidden_size > 0 else 1024

BS_R = args.bs_r if args.bs_r > 0 else 1
BS_C = args.bs_c if args.bs_c > 0 else 1

assert HS % BS_R == 0
assert HS % BS_C == 0
assert IS % BS_C == 0

print("\n\ngrucell: [{}, {}, {}], BS_R: {}, BS_C: {}".format(BATCH, IS, HS, BS_R, BS_C))

density = 0.05

def get_matrix(IS, HS, BS_R, BS_C, density):
    filename = str(IS) + "_" + str(HS) + "_" + str(BS_R) + "_" + str(BS_C) + "_" + str(density) + ".npz"
    if os.path.isfile("mask_data/" + filename):
        with open("mask_data/" + filename, "rb") as f:
            mask = np.load(f)
    else:
        mask = np.random.choice([0, 1], size=(3 * HS // BS_R, IS // BS_C), p=[1-density, density])
        with open("mask_data/" + filename, "wb") as f:
            np.save(f, mask)
    mask = np.repeat(mask, BS_C, axis=1)
    mask = np.repeat(mask, BS_R, axis=0)
    bb = (np.random.rand(3 * HS, IS).astype(dtype) * mask).astype(dtype)
    if wdtype == "uint16":
        assert bb.dtype == np.float32
        bb = ((bb.view('<u4') + 2 ** 15) >> 16).astype("uint16")
    return bb

def get_bsr(m):
    return bsr_matrix(m, blocksize=(BS_R, BS_C))

w_i = get_matrix(IS, HS, BS_R, BS_C, density)
w_h = get_matrix(HS, HS, BS_R, BS_C, density)
b_i = np.random.rand(3 * HS)
b_h = np.random.rand(3 * HS)
x = np.random.rand(BATCH, IS)
h = np.random.rand(BATCH, HS)
num_nonzeros = np.count_nonzero(w_i) + np.count_nonzero(w_h)
print("non zeros: {}".format(num_nonzeros))

# baseline
ww_i = w_i
if wdtype == "uint16":
    ww_i = (w_i.astype("uint32") << 16).view(dtype="float32")
ww_h = w_h
if wdtype == "uint16":
    ww_h = (w_h.astype("uint32") << 16).view(dtype="float32")

gru_cell = torch.nn.GRUCell(IS, HS)
gru_cell.weight_hh.data = torch.tensor(w_h.astype("double"))
gru_cell.weight_ih.data = torch.tensor(w_i.astype("double"))
gru_cell.bias_hh.data = torch.tensor(b_h.astype("double"))
gru_cell.bias_ih.data = torch.tensor(b_i.astype("double"))
xx = torch.tensor(x.astype("double"))
hh = torch.tensor(h.astype("double"))
answer = gru_cell(xx, hh)

def build_graph(params):
    global func
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
        return graph, lib, new_params

def test(params, inputs):
    if args.default_schedule:
        (graph, lib, new_params) = build_graph(params)
    else:
        with autotvm.apply_history_best("synthesis_autotvm_skl.log"):
            (graph, lib, new_params) = build_graph(params)
    # mport pdb; pdb.set_trace()
    r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
    r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**r_new_params)
    module.set_input(**r_inputs)
    module.run()
    ro = tvm.nd.empty([BATCH, HS])
    module.get_output(0, ro)
    # import pdb; pdb.set_trace()
    # tvm.testing.assert_allclose(ro.asnumpy(), answer.detach().numpy(), atol=1e-5, rtol=1e-1)

    ftimer = module.module.time_evaluator("run", ctx, 100)
    for i in range(5):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)

def dense(X, W, B, **kwargs):
    return relay.nn.bias_add(relay.nn.dense(X, W), B)

def gru_cell(cell, x, h):
    xt = dense(x, cell.weight_ih, cell.bias_ih)
    ht = dense(h, cell.weight_hh, cell.bias_hh)
    xt_gates = relay.split(xt, indices_or_sections=3, axis=1)
    ht_gates = relay.split(ht, indices_or_sections=3, axis=1)
    reset_gate = relay.sigmoid(xt_gates[0] + ht_gates[0])
    input_gate = relay.sigmoid(xt_gates[1] + ht_gates[1])
    new_gate = relay.tanh(xt_gates[2] + reset_gate * ht_gates[2])
    return new_gate + input_gate * (h - new_gate)

Cell = collections.namedtuple('Cell', ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])

Rrnn1 = Cell(
    weight_ih=relay.var("rnn1_weight_ih", shape=(3 * HS, IS), dtype="float32"),
    weight_hh=relay.var("rnn1_weight_hh", shape=(3 * HS, HS), dtype="float32"),
    bias_ih=relay.var("rnn1_bias_ih", shape=(3 * HS, ), dtype="float32"),
    bias_hh=relay.var("rnn1_bias_hh", shape=(3 * HS, ), dtype="float32"),
)

tx = relay.var('x', shape=[BATCH, IS], dtype="float32")
th = relay.var('h', shape=[BATCH, HS], dtype="float32")
outputs = gru_cell(Rrnn1, tx, th)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)

params = collections.OrderedDict([
    ("rnn1_weight_ih", tvm.nd.array(w_i.astype(wdtype), ctx)),
    ("rnn1_weight_hh", tvm.nd.array(w_h.astype(wdtype), ctx)),
    ("rnn1_bias_ih", tvm.nd.array(b_i.astype(wdtype), ctx)),
    ("rnn1_bias_hh", tvm.nd.array(b_h.astype(wdtype), ctx)),
])

inputs = collections.OrderedDict([
    ("x", tvm.nd.array(x.astype("float32"), ctx)),
    ("h", tvm.nd.array(h.astype("float32"), ctx))
])

# test(params, inputs)

w_x = np.concatenate((w_i[:HS,:],w_h[:HS,:]), axis=1)
b_x = b_i[:HS] + b_h[:HS]
w_z = np.concatenate((w_i[HS:2*HS,:],w_h[HS:2*HS,:]), axis=1)
b_z = b_i[HS:2*HS] + b_h[HS:2*HS]
w_in = w_i[2*HS:,:]
b_in = b_i[2*HS:]
w_hn = w_h[2*HS:,:]
b_hn = b_h[2*HS:]

input = np.concatenate((x, h), axis=1)

import collections
CSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'density'])
BSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'BS_R', 'BS_C', 'density'])

print("grucell")
tinput = relay.var("tinput", shape=[BATCH, IS + HS], dtype=dtype)
tw_x = relay.var("tw_x", shape=(HS, HS + IS), dtype=wdtype)
tb_x = relay.var("tb_x", shape=(HS,), dtype=wdtype)
tw_z = relay.var("tw_z", shape=(HS, HS + IS), dtype=wdtype)
tb_z = relay.var("tb_z", shape=(HS,), dtype=wdtype)
tw_in = relay.var("tw_in", shape=(HS, IS), dtype=wdtype)
tb_in = relay.var("tb_in", shape=(HS,), dtype=wdtype)
tw_hn = relay.var("tw_hn", shape=(HS, HS), dtype=wdtype)
tb_hn = relay.var("tb_hn", shape=(HS,), dtype=wdtype)

outputs = relay.nn.grucell(tinput, tw_x, tb_x, tw_z, tb_z, tw_in, tb_in, tw_hn, tb_hn)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))

params = collections.OrderedDict([
    ("tw_x", tvm.nd.array(w_x.astype(wdtype), ctx)),
    ("tb_x", tvm.nd.array(b_x.astype(wdtype), ctx)),
    ("tw_z", tvm.nd.array(w_z.astype(wdtype), ctx)),
    ("tb_z", tvm.nd.array(b_z.astype(wdtype), ctx)),
    ("tw_in", tvm.nd.array(w_in.astype(wdtype), ctx)),
    ("tb_in", tvm.nd.array(b_in.astype(wdtype), ctx)),
    ("tw_hn", tvm.nd.array(w_hn.astype(wdtype), ctx)),
    ("tb_hn", tvm.nd.array(b_hn.astype(wdtype), ctx)),
])
inputs = collections.OrderedDict(
    [("tinput", tvm.nd.array(input.astype("float32"), ctx))])


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

# test(params, inputs)

w_x = np.concatenate((w_i[:HS,:],w_h[:HS,:]), axis=1)
b_x = b_i[:HS] + b_h[:HS]
w_z = np.concatenate((w_i[HS:2*HS,:],w_h[HS:2*HS,:]), axis=1)
b_z = b_i[HS:2*HS] + b_h[HS:2*HS]
w_in = w_i[2*HS:,:]
b_in = b_i[2*HS:]
w_hn = w_h[2*HS:,:]
b_hn = b_h[2*HS:]


def to_bsr(v, ref_v, density=0.04):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = np.count_nonzero(ref_v)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    v_data = relay.var(name + "_data", shape=(num_blocks, BS_R, BS_C), dtype=wdtype)
    v_indices = relay.var(name + "_indices", shape=(num_blocks,), dtype=witype)
    v_indptr = relay.var(name + "_indptr", shape=(N // BS_R + 1,), dtype="int32")
    return BSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, BS_R=BS_R, BS_C=BS_C, density=density)


tinput = relay.var("tinput", shape=[BATCH, IS + HS], dtype=dtype)
tw_x = to_bsr(relay.var("tw_x", shape=(HS, HS + IS), dtype=wdtype), w_x)
tb_x = relay.var("tb_x", shape=(HS,), dtype=wdtype)
tw_z = to_bsr(relay.var("tw_z", shape=(HS, HS + IS), dtype=wdtype), w_z)
tb_z = relay.var("tb_z", shape=(HS,), dtype=wdtype)
tw_in = to_bsr(relay.var("tw_in", shape=(HS, IS), dtype=wdtype), w_in)
tb_in = relay.var("tb_in", shape=(HS,), dtype=wdtype)
tw_hn = to_bsr(relay.var("tw_hn", shape=(HS, HS), dtype=wdtype), w_hn)
tb_hn = relay.var("tb_hn", shape=(HS,), dtype=wdtype)

outputs = relay.nn.sgrucell(tinput, tw_x, tb_x, tw_z, tb_z, tw_in, tb_in, tw_hn, tb_hn)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))

w_x_bsr = get_bsr(w_x)
w_z_bsr = get_bsr(w_z)
w_in_bsr = get_bsr(w_in)
w_hn_bsr = get_bsr(w_hn)

params = collections.OrderedDict([
    ("tw_x_data", tvm.nd.array(w_x_bsr.data.astype(wdtype), ctx)),
    ("tw_x_indices", tvm.nd.array(w_x_bsr.indices.astype(witype), ctx)),
    ("tw_x_indptr", tvm.nd.array(w_x_bsr.indptr.astype("int32"), ctx)),
    ("tb_x", tvm.nd.array(b_x.astype(wdtype), ctx)),
    ("tw_z_data", tvm.nd.array(w_z_bsr.data.astype(wdtype), ctx)),
    ("tw_z_indices", tvm.nd.array(w_z_bsr.indices.astype(witype), ctx)),
    ("tw_z_indptr", tvm.nd.array(w_z_bsr.indptr.astype("int32"), ctx)),
    ("tb_z", tvm.nd.array(b_z.astype(wdtype), ctx)),
    ("tw_in_data", tvm.nd.array(w_in_bsr.data.astype(wdtype), ctx)),
    ("tw_in_indices", tvm.nd.array(w_in_bsr.indices.astype(witype), ctx)),
    ("tw_in_indptr", tvm.nd.array(w_in_bsr.indptr.astype("int32"), ctx)),
    ("tb_in", tvm.nd.array(b_in.astype(wdtype), ctx)),
    ("tw_hn_data", tvm.nd.array(w_hn_bsr.data.astype(wdtype), ctx)),
    ("tw_hn_indices", tvm.nd.array(w_hn_bsr.indices.astype(witype), ctx)),
    ("tw_hn_indptr", tvm.nd.array(w_hn_bsr.indptr.astype("int32"), ctx)),
    ("tb_hn", tvm.nd.array(b_hn.astype(wdtype), ctx)),
])

inputs = collections.OrderedDict(
    [("tinput", tvm.nd.array(input.astype("float32"), ctx))])

test(params, inputs)
