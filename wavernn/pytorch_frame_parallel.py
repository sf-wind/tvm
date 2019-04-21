import logging
logging.basicConfig(level=logging.DEBUG)

import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import tvm
from tvm import relay
from tvm import autotvm
import itertools
import scipy.sparse as sp
import os
import time
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--bs_r", type=int, default=0)
parser.add_argument("--bs_c", type=int, default=0)
parser.add_argument("--tuner", type=str, default="xgboost",
                    choices=["ga", "xgboost"])
parser.add_argument("--target", type=str, default="core-avx2",
                    choices=["core-avx2", "skylake-avx512"])
parser.add_argument("--default_schedule", action="store_true")
parser.add_argument("--wtype", type=str, default="float32",
                    choices=["float32", "bfloat16"])
parser.add_argument("--sdense", action="store_true")
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

def sparsify(arr, BS_R, BS_C, density):
    (M, N) = arr.shape
    Y = np.zeros((M, N), dtype="float32")
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(
        list(itertools.product(range(0, M, BS_R),
                               range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = arr[r:r + BS_R, c:c + BS_C]
    return Y

'''
def sparsify(arr, BS_R, BS_C, density, dtype="float32", wdtype="float32"):
    (M, N) = arr.shape
    assert M % BS_R == 0
    assert N % BS_C == 0
    filename = str(M) + "_" + str(N) + "_" + str(BS_R) + "_" + str(BS_C) + "_" + str(density) + ".npz"
    if os.path.isfile("mask_data/" + filename):
        with open("mask_data/" + filename, "rb") as f:
            mask = np.load(f)
    else:
        mask = np.random.choice([0, 1], size=(M //BS_R, N // BS_C), p=[1-density, density])
        with open("mask_data/" + filename, "wb") as f:
            np.save(f, mask)
    mask = np.repeat(mask, BS_C, axis=1)
    mask = np.repeat(mask, BS_R, axis=0)

    bb = (np.random.rand(M, N).astype(dtype) * mask).astype(dtype)
    if wdtype == "uint16":
        bb = (bb.view(dtype="uint32") >> 16).astype("uint16")
    return bb
'''

BS_R = args.bs_r if args.bs_r > 0 else 1
BS_C = args.bs_c if args.bs_c > 0 else 1
BS_R = 16
BS_C = 1
rnn_dims = 1024
fc_dims = 1024

feat_dims = 19
aux_dims = 64
n_classes = 2 ** 8
num_parallel_samples = 4

T = 8
x_num = np.sum(range(num_parallel_samples))
x_0 = torch.randn(1, x_num)
outputs_0 = torch.ones(int(num_parallel_samples), int(num_parallel_samples-1))
outputs_0 = outputs_0 / 2
h1_0 = torch.randn(1, rnn_dims)
m = torch.randn(1, T, feat_dims)
a1 = torch.randn(1, T, aux_dims)
a2 = torch.randn(1, T, aux_dims)

I = nn.Linear(feat_dims + aux_dims + x_num, rnn_dims)
rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
fc1 = nn.Linear(rnn_dims + aux_dims, fc_dims)
fc2 = nn.Linear(fc_dims, n_classes * num_parallel_samples)


rnn1.weight_ih[:, :] = torch.tensor(sparsify(rnn1.weight_ih.detach().numpy(), BS_R=BS_R, BS_C=BS_C, density=0.05))
rnn1.weight_hh[:, :] = torch.tensor(sparsify(rnn1.weight_hh.detach().numpy(), BS_R=BS_R, BS_C=BS_C, density=0.05))

fc2.weight[:, :] = torch.tensor(sparsify(fc2.weight.detach().numpy(), BS_R=BS_R, BS_C=BS_C, density=0.2))

fc1.weight[:, :rnn_dims] = torch.tensor(sparsify(fc1.weight[:, :rnn_dims].detach().numpy(), BS_R=BS_R, BS_C=BS_C, density=0.05))

def tvm_random_seed(seed):
    tvm.get_global_func("tvm.contrib.wavernn.set_seed")(seed)

def tvm_random_uniform():
    return tvm.get_global_func("tvm.contrib.wavernn.random_uniform")()


def sample(x_prob):
    gumbel = -torch.log(-torch.log(torch.tensor(np.random.uniform(size=x_prob.shape).astype("float32"))))
    result = np.zeros((1, 1), dtype="float32")
    result[:] = np.argmax(x_prob - gumbel)
    return torch.tensor(result / n_classes)

def sample_proba(x_prob, i=0):
    rand_sample = 0
    prob_sum = x_prob[i][0]
    rand = tvm_random_uniform()

    while prob_sum < rand:
        rand_sample += 1
        prob_sum += x_prob[i][rand_sample]

    result = np.zeros((1, 1), dtype="float32")
    result[:] = rand_sample
    return torch.tensor(result / n_classes)

def reference_frame(a1, a2, m, outputs_0, h1_0):
    # import pdb; pdb.set_trace()
    tvm_random_seed(10)
    (outputs, h1) = (outputs_0, h1_0)
    T = a1.shape[1]
    for t in range(T):
        x = torch.tensor([])
        for shift in np.arange(1,num_parallel_samples):
            x = torch.cat([x, outputs[0:(num_parallel_samples-shift), -shift]])
        x = x.unsqueeze(0)
        xconcat = torch.cat([x, m[0, t:t+1], a1[0, t:t+1]], dim=1)
        xconcat_trns = I(xconcat)
        h1 = rnn1(xconcat_trns, h1)
        xres = xconcat_trns + h1
        xres_concat = torch.cat([xres, a2[0, t:t+1]], dim=1)
        x_fc = F.relu(fc1(xres_concat))

        x_prob = fc2(x_fc)
        output = torch.empty(num_parallel_samples)
        for i in range(num_parallel_samples):
            x_prob_sub = torch.softmax(x_prob[:, i * n_classes : (i+1) * n_classes], dim=1)
            x = sample_proba(x_prob_sub)
            output[i] = x
        output = output.unsqueeze(1)
        outputs=torch.cat([outputs, output], dim=1)
    return outputs, h1


def reference():
    return reference_frame(a1, a2, m, outputs_0, h1_0)


def test_pytorch_reference():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = reference_frame(a1, a2, m, outputs_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new)
        np.testing.assert_allclose(h1_ref, h1_new)


def factored_premul_frame(a1, a2, m, outputs_0, h1_0):
    tvm_random_seed(10)
    I_residual =  m[0] @ I.weight[:, x_num:x_num + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, x_num + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    Ifactored = nn.Linear(x_num, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :x_num]
    Ifactored.bias[:] = I.bias[:]

    (outputs, h1) = (outputs_0, h1_0)
    T = a1.shape[1]
    for t in range(T):
        x = torch.tensor([])
        for shift in np.arange(1,num_parallel_samples):
            x = torch.cat([x, outputs[0:(num_parallel_samples-shift), -shift]])
        xconcat_trns = Ifactored(x) + I_residual[t:t+1]

        def gru_cell(cell, x, h):
            xt = x @ cell.weight_ih.transpose(1, 0) + cell.bias_ih
            ht = h @ cell.weight_hh.transpose(1, 0) + cell.bias_hh
            assert xt.shape == (1, 3 * rnn_dims)
            assert ht.shape == (1, 3 * rnn_dims)

            reset_gate = torch.sigmoid(xt[:, 0:rnn_dims] + ht[:, 0:rnn_dims])
            input_gate = torch.sigmoid(xt[:, rnn_dims:2 * rnn_dims] + ht[:, rnn_dims:2 * rnn_dims])
            new_gate = torch.tanh(xt[:, 2 * rnn_dims:3 * rnn_dims] + reset_gate * ht[:, 2 * rnn_dims:3 * rnn_dims])
            return new_gate + input_gate * (h - new_gate)

        h1 = gru_cell(rnn1, xconcat_trns, h1)

        xres = xconcat_trns + h1

        fc1factored = nn.Linear(rnn_dims, fc_dims)
        fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
        fc1factored.bias[:] = fc1.bias[:]

        x_fc = F.relu(fc1factored(xres) + fc1_residual[t:t+1])
        x_prob = fc2(x_fc)
        # import pdb; pdb.set_trace()
        x_prob_sub = x_prob.view(num_parallel_samples, n_classes)
        x_softmax = torch.softmax(x_prob_sub, dim=-1)
        output = torch.empty(num_parallel_samples)
        for i in range(num_parallel_samples):
            # x_prob_sub = torch.softmax(x_prob[:, i * n_classes : (i+1) * n_classes], dim=1)
            x = sample_proba(x_softmax, i)
            output[i] = x
        output = output.unsqueeze(1)
        outputs=torch.cat([outputs, output], dim=1)
    return outputs, h1


def build_wavernn_module(target="llvm"):
    Ifactored = nn.Linear(x_num, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :x_num]
    Ifactored.bias[:] = I.bias[:]

    fc1factored = nn.Linear(rnn_dims, fc_dims)
    fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
    fc1factored.bias[:] = fc1.bias[:]

    params = {
        "I_W": tvm.ndarray.array(Ifactored.weight.detach().numpy()),
        "I_B": tvm.ndarray.array(Ifactored.bias.detach().numpy()),
        "rnn1_weight_ih": tvm.ndarray.array(rnn1.weight_ih.detach().numpy()),
        "rnn1_weight_hh": tvm.ndarray.array(rnn1.weight_hh.detach().numpy()),
        "rnn1_bias_ih": tvm.ndarray.array(rnn1.bias_ih.detach().numpy()),
        "rnn1_bias_hh": tvm.ndarray.array(rnn1.bias_hh.detach().numpy()),
        "fc1_W": tvm.ndarray.array(fc1factored.weight.detach().numpy()),
        "fc1_B": tvm.ndarray.array(fc1factored.bias.detach().numpy()),
        "fc2_W": tvm.ndarray.array(fc2.weight.detach().numpy()),
        "fc2_B": tvm.ndarray.array(fc2.bias.detach().numpy()),
    }

    Rx = relay.var("x", shape=[1, x_num], dtype="float32")
    Rh1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")
    RI_residual = relay.var("I_residual", shape=[1, rnn_dims], dtype="float32")
    Rfc1_residual = relay.var("fc1_residual", shape=[1, fc_dims], dtype="float32")
    RI_W = relay.var("I_W", shape=(rnn_dims, x_num), dtype="float32")
    RI_B = relay.var("I_B", shape=(rnn_dims,), dtype="float32")

    Cell = collections.namedtuple('Cell', ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])

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

    xconcat_trns = dense(Rx, RI_W, RI_B) + RI_residual

    Rrnn1 = Cell(
        weight_ih=relay.var("rnn1_weight_ih", shape=(3 * rnn_dims, rnn_dims), dtype="float32"),
        weight_hh=relay.var("rnn1_weight_hh", shape=(3 * rnn_dims, rnn_dims), dtype="float32"),
        bias_ih=relay.var("rnn1_bias_ih", shape=(3 * rnn_dims, ), dtype="float32"),
        bias_hh=relay.var("rnn1_bias_hh", shape=(3 * rnn_dims, ), dtype="float32"),
    )
    h1 = gru_cell(Rrnn1, xconcat_trns, Rh1)
    xres = xconcat_trns + h1

    Rfc1_W = relay.var("fc1_W", shape=(fc_dims, rnn_dims), dtype="float32")
    Rfc1_B = relay.var("fc1_B", shape=(fc_dims,), dtype="float32")

    x_fc = relay.nn.relu(dense(xres, Rfc1_W, Rfc1_B) + Rfc1_residual)

    Rfc2_W = relay.var("fc2_W", shape=(n_classes * num_parallel_samples, fc_dims), dtype="float32")
    Rfc2_B = relay.var("fc2_B", shape=(n_classes * num_parallel_samples,), dtype="float32")

    x_dense = dense(x_fc, Rfc2_W, Rfc2_B)
    x_prob_tuple = relay.split(x_dense, num_parallel_samples, axis=1)
    tuple = []
    for i in range(num_parallel_samples):
        tuple.append(relay.nn.softmax(x_prob_tuple[i]))

    outputs = relay.expr.Tuple( tuple + [h1])
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    func = relay.ir_pass.infer_type(func)
    graph, lib, params = relay.build_module.build(func, target=target, params=params)
    return (graph, lib, params)

def build_fast_wavernn_module(target="llvm", bfloat16=False, tune=False, profile=False):
    Ifactored = nn.Linear(x_num, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :x_num]
    Ifactored.bias[:] = I.bias[:]

    fc1factored = nn.Linear(rnn_dims, fc_dims)
    fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
    fc1factored.bias[:] = fc1.bias[:]

    params = {
        "I_W": tvm.ndarray.array(Ifactored.weight.detach().numpy()),
        "I_B": tvm.ndarray.array(Ifactored.bias.detach().numpy()),
        "rnn1_weight_ih": tvm.ndarray.array(rnn1.weight_ih.detach().numpy()),
        "rnn1_weight_hh": tvm.ndarray.array(rnn1.weight_hh.detach().numpy()),
        "rnn1_bias_ih": tvm.ndarray.array(rnn1.bias_ih.detach().numpy()),
        "rnn1_bias_hh": tvm.ndarray.array(rnn1.bias_hh.detach().numpy()),
        "fc1_W": tvm.ndarray.array(fc1factored.weight.detach().numpy()),
        "fc1_B": tvm.ndarray.array(fc1factored.bias.detach().numpy()),
        "fc2_W": tvm.ndarray.array(fc2.weight.detach().numpy()),
        "fc2_B": tvm.ndarray.array(fc2.bias.detach().numpy()),
    }

    Rx = relay.var("x", shape=[1, x_num], dtype="float32")
    Rh1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")
    RI_residual = relay.var("I_residual", shape=[1, rnn_dims], dtype="float32")
    Rfc1_residual = relay.var("fc1_residual", shape=[1, fc_dims], dtype="float32")
    RI_W = relay.var("I_W", shape=(rnn_dims, x_num), dtype="float32")
    RI_B = relay.var("I_B", shape=(rnn_dims,), dtype="float32")

    Cell = collections.namedtuple('Cell', ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])

    BSR = collections.namedtuple(
        'BSR',
        ['data', 'indices', 'indptr',])

    def approx_exp(x):
        x = relay.minimum(relay.maximum(x, C(-88.0)), C(88.0))
        x = C(127.0) + x * C(1.44269504)
        xf = relay.floor(x)
        i = relay.cast(xf, "int32")
        x = x - xf
        Y = C(0.99992522) + x * (C(0.69583354) + x * (C(0.22606716) + x * C(0.078024523)))
        exponent = relay.left_shift(i, relay.expr.const(23, "int32"))
        exponent = relay.reinterpret(exponent, "float32")
        return exponent * Y

    def approx_sigmoid(x):
        y = approx_exp(x)
        return y / (y + C(1.0))

    def approx_tanh(x):
        x = x * C(2.0)
        y = approx_exp(x)
        return (y - C(1.0)) / (y + C(1.0))

    def C(x):
        return relay.expr.const(x, "float32")

    def sparse_dense(X, W, B, **kwargs):
        if args.sdense:
            d = relay.nn.sdense(X, W)
        else:
            d = relay.nn.sparse_dense(X, W)
        return relay.nn.bias_add(d, B)

    def dense(X, W, B, **kwargs):
        return relay.nn.bias_add(relay.nn.dense(X, W), B)

    def gru_cell(cell, x, h):
        xt = sparse_dense(x, cell.weight_ih, cell.bias_ih)
        ht = sparse_dense(h, cell.weight_hh, cell.bias_hh)
        xt_gates = relay.split(xt, indices_or_sections=3, axis=1)
        ht_gates = relay.split(ht, indices_or_sections=3, axis=1)
        reset_gate = approx_sigmoid(xt_gates[0] + ht_gates[0])
        input_gate = approx_sigmoid(xt_gates[1] + ht_gates[1])
        new_gate = approx_tanh(xt_gates[2] + reset_gate * ht_gates[2])
        return new_gate + input_gate * (h - new_gate)

    def to_bf16(x):
        assert x.dtype == np.float32
        return ((x.view('<u4') + 2 ** 15) >> 16).astype("uint16")

    def to_sparse(v, arr, BS_R=BS_R, BS_C=BS_C):
        name = v.name_hint
        (N, K) = v.type_annotation.concrete_shape
        assert (N, K) == arr.shape
        sp_arr = sp.bsr_matrix(arr, blocksize=(BS_R, BS_C))
        print("Sparsity achieved: {:.2f}%".format((1.0 - float(sp_arr.data.size) / arr.size) * 100))
        v_data = relay.var(name + "_data", shape=sp_arr.data.shape, dtype="float32" if not bfloat16 else "uint16")
        v_indices = relay.var(name + "_indices", shape=sp_arr.indices.shape, dtype="int32")
        v_indptr = relay.var(name + "_indptr", shape=sp_arr.indptr.shape, dtype="int32")
        params[name + "_data"] = tvm.ndarray.array(sp_arr.data) if not bfloat16 else tvm.ndarray.array(to_bf16(sp_arr.data))
        params[name + "_indices"] = tvm.ndarray.array(sp_arr.indices)
        params[name + "_indptr"] = tvm.ndarray.array(sp_arr.indptr)
        return BSR(data=v_data, indices=v_indices, indptr=v_indptr)

    xconcat_trns = dense(Rx, RI_W, RI_B) + RI_residual

    Rrnn1 = Cell(
        weight_ih=to_sparse(relay.var("rnn1_weight_ih", shape=(3 * rnn_dims, rnn_dims), dtype="float32"), rnn1.weight_ih.detach().numpy()),
        weight_hh=to_sparse(relay.var("rnn1_weight_hh", shape=(3 * rnn_dims, rnn_dims), dtype="float32"), rnn1.weight_hh.detach().numpy()),
        bias_ih=relay.var("rnn1_bias_ih", shape=(3 * rnn_dims, ), dtype="float32"),
        bias_hh=relay.var("rnn1_bias_hh", shape=(3 * rnn_dims, ), dtype="float32"),
    )
    h1 = gru_cell(Rrnn1, xconcat_trns, Rh1)
    xres = xconcat_trns + h1

    Rfc1_W = to_sparse(relay.var("fc1_W", shape=(fc_dims, rnn_dims), dtype="float32"), fc1factored.weight.detach().numpy())
    Rfc1_B = relay.var("fc1_B", shape=(fc_dims,), dtype="float32")

    x_fc = relay.nn.relu(sparse_dense(xres, Rfc1_W, Rfc1_B) + Rfc1_residual)

    Rfc2_W = to_sparse(relay.var("fc2_W", shape=(n_classes * num_parallel_samples, fc_dims), dtype="float32"), fc2.weight.detach().numpy())
    Rfc2_B = relay.var("fc2_B", shape=(n_classes * num_parallel_samples,), dtype="float32")

    x_dense = sparse_dense(x_fc, Rfc2_W, Rfc2_B)
    x_prob_unnorm = approx_exp(x_dense)
    x_prob_tuple = relay.reshape(x_prob_unnorm, (num_parallel_samples, n_classes))
    x_prob_sum = relay.sum(x_prob_tuple, axis=-1)
    x_prob_sum = relay.expand_dims(x_prob_sum, -1)
    x_prob = x_prob_tuple / x_prob_sum
    outputs = relay.expr.Tuple([x_prob, h1])
    '''
    x_prob_tuple = relay.split(x_prob_unnorm, num_parallel_samples, axis=1)

    tuple = []
    for i in range(num_parallel_samples):
        x_prob_sum = relay.sum(x_prob_unnorm, axis=-1)
        x_prob = x_prob_unnorm / x_prob_sum
        tuple.append(x_prob)
    outputs = relay.expr.Tuple(tuple + [h1])
    '''
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    func = relay.ir_pass.infer_type(func)
    # print(func.astext())
    TARGET = tvm.target.create(target)
    log_filename = "lpcnet_no_bf16_autotvm_skl.log"

    if tune:
        with relay.build_config(opt_level=2):
            func = relay.optimize(func, target=TARGET, params=params)
            print(func.astext(show_meta_data=False))
            tasks = autotvm.task.extract_from_program(
                func, target=TARGET, params=params, ops=(relay.op.nn.dense,))
            for i, tsk in enumerate(tasks):
                print(tsk)
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
                n_trial = 100
                early_stopping = 200
                measure_option = autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(number=100, repeat=3,
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

    with autotvm.apply_history_best(log_filename):
        with relay.build_config(opt_level=3):
            func = relay.optimize(func, target=TARGET, params=params)
            func = relay.ir_pass.infer_type(func)
            graph, lib, new_params = relay.build_module.build(
                func, target=TARGET,  params=params)

        # profile
        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath('net.tar')
        with TARGET:
            lib.export_library(lib_fname)
        # tracker = tvm.rpc.connect_tracker('0.0.0.0', 9198)
        # remote = tracker.request('skl')

    if profile:
        # remote.upload(lib_fname)
        rlib = lib
        ctx = tvm.context(target, 0)
        # rlib = remote.load_module('net.tar')
        # ctx = remote.cpu(0)
        r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
        inputs = {
            "x": tvm.nd.array(np.random.randn(1, x_num).astype(np.float32)),
            "h1": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
            "I_residual": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
            "fc1_residual": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
        }
        r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
        r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
        module = graph_runtime.create(graph, rlib, ctx)
        module.set_input(**r_new_params)
        module.set_input(**r_inputs)
        ftimer = module.module.time_evaluator("run", ctx, number=100000)
        for i in range(5):
            prof_res = ftimer()
            print("TVM time: {:.2f}us".format(prof_res.mean * 10 ** 6))
        module.run()
        module.run()

    return (graph, lib, new_params)


def factored_relay_frame(a1, a2, m, outputs_0, h1_0):
    tvm_random_seed(10)
    h1 = tvm.ndarray.array(h1_0)
    outputs = outputs_0
    T = a1.shape[1]
    (graph, lib, params) = build_wavernn_module()
    module = graph_runtime.create(graph, lib, tvm.cpu(0))
    module.set_input(**params)

    I_residual = m[0] @ I.weight[:, x_num:x_num + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, x_num + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    for t in range(T):
        x = np.empty((1, x_num))
        # import pdb; pdb.set_trace()
        start = 0
        for shift in np.arange(1,num_parallel_samples):
            end = start + num_parallel_samples - shift
            x[0, start:end] = outputs[0:(num_parallel_samples-shift), -shift]
            start = end
        x = tvm.ndarray.array(x.astype("float32"))
        inputs = {
            "x": x,
            "h1": h1,
            "I_residual": tvm.ndarray.array(I_residual[t:t+1].detach().numpy()),
            "fc1_residual": tvm.ndarray.array(fc1_residual[t:t+1].detach().numpy()),
        }

        module.set_input(**inputs)
        module.run()
        num_outputs = module.get_num_outputs()
        h1 = module.get_output(num_outputs-1)
        output = torch.empty(num_outputs-1)
        for i in range(num_outputs-1):
            x_prob = module.get_output(i)
            x = sample_proba(torch.tensor(x_prob.asnumpy()))
            output[i] = x
        output = output.unsqueeze(1)
        outputs = torch.cat([outputs, output], dim=1)
    return outputs.numpy(), h1.asnumpy()

def factored_relay_cpp_frame(a1, a2, m, outputs_0, h1_0):
    tvm_random_seed(10)
    outputs = outputs_0
    h1 = tvm.ndarray.array(h1_0)
    T = a1.shape[1]
    (graph, lib, params) = build_wavernn_module()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, prefix="tvm_model_lib", suffix=".so") as lib_f:
        lib.export_library(lib_f.name)
    I_residual = m[0] @ I.weight[:, x_num:x_num + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, x_num + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    frame_func = tvm.get_global_func("tvm.contrib.wavernn.parallel_frame")
    np_outs = np.random.randn(num_parallel_samples, T + num_parallel_samples - 1).astype("float32")
    np_outs[:, :num_parallel_samples-1] = outputs
    outs = tvm.ndarray.array(np_outs)

    # h1 = tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32"))
    frame_func(
        # Inputs
        tvm.ndarray.array(I_residual),
        tvm.ndarray.array(fc1_residual),
        tvm.ndarray.array(h1_0),
        # inouts
        outs,
        h1,
        # Temporary storage to make entire frame_func allocation free.
        tvm.ndarray.array(np.random.randn(1, x_num).astype("float32")),
        tvm.ndarray.array(np.random.randn(num_parallel_samples, n_classes).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, fc_dims).astype("float32")),
        # Data for constructing the module.
        graph,  # the graph JSON.
        lib_f.name,  # the exported shared object.
        relay.save_param_dict(params)  # the serialized parameters.
    )
    return outs.asnumpy(), h1.asnumpy()

def factored_relay_cpp_frame_fast(a1, a2, m, outputs_0, h1_0):
    tvm_random_seed(10)
    outputs = outputs_0
    h1 = tvm.ndarray.array(h1_0)
    T = a1.shape[1]
    (graph, lib, params) = build_fast_wavernn_module(profile=False)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, prefix="tvm_model_lib", suffix=".so") as lib_f:
        lib.export_library(lib_f.name)
    I_residual = m[0] @ I.weight[:, x_num:x_num + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, x_num + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    frame_func = tvm.get_global_func("tvm.contrib.wavernn.parallel_frame")

    np_outs = np.random.randn(num_parallel_samples, T + num_parallel_samples - 1).astype("float32")
    np_outs[:, :num_parallel_samples-1] = outputs
    outs = tvm.ndarray.array(np_outs)
    # h1 = tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32"))
    frame_func(
        # Inputs
        tvm.ndarray.array(I_residual),
        tvm.ndarray.array(fc1_residual),
        tvm.ndarray.array(h1_0),
        # inouts
        outs,
        h1,
        # Temporary storage to make entire frame_func allocation free.
        tvm.ndarray.array(np.random.randn(1, x_num).astype("float32")),
        tvm.ndarray.array(np.random.randn(num_parallel_samples, n_classes).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, fc_dims).astype("float32")),
        # Data for constructing the module.
        graph,  # the graph JSON.
        lib_f.name,  # the exported shared object.
        relay.save_param_dict(params)  # the serialized parameters.
    )
    return outs.asnumpy(), h1.asnumpy()

def test_factored_premul_frame():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_premul_frame(a1, a2, m, outputs_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_frame():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_frame(a1, a2, m, outputs_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_cpp_frame():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_cpp_frame(a1, a2, m, outputs_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_cpp_frame_fast():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_cpp_frame_fast(a1, a2, m, outputs_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        # print(h1_ref, h1_new)
        # print(outs_ref, outs_new)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def skylake():
    (graph, lib, params) = build_fast_wavernn_module("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu", bfloat16=True, profile=True)
    with open(
            "skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_graph.json".format(**globals()),
            "w") as f:
        f.write(graph)

    with open(
            "skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_params.bin".format(**globals()),
            "wb") as f:
        f.write(relay.save_param_dict(params))

    lib.save("skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.o".format(**globals()))

def haswell():
    (graph, lib, params) = build_fast_wavernn_module("llvm -mcpu=core-avx2 -target=x86_64-linux-gnu", bfloat16=True, profile=True)
    with open(
            "hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_graph.json".format(**globals()),
            "w") as f:
        f.write(graph)

    with open(
            "hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_params.bin".format(**globals()),
            "wb") as f:
        f.write(relay.save_param_dict(params))

    lib.save("hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.o".format(**globals()))

test_factored_premul_frame()
# test_relay_frame()
# test_relay_cpp_frame()
test_relay_cpp_frame_fast()
skylake()
# haswell()
