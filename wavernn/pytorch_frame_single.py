import logging

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
import argparse
import os
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--align_entries", type=int, default=1)
parser.add_argument("--merged_gru", type=str, default="interleaving",
                    choices=["simple", "interleaving"])
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--m", type=int, default=0)
parser.add_argument("--bs_r", type=int, default=0)
parser.add_argument("--bs_c", type=int, default=0)
parser.add_argument("--tuner", type=str, default="xgboost",
                    choices=["ga", "xgboost"])
parser.add_argument("--target", type=str, default="core-avx2",
                    choices=["core-avx2", "skylake-avx512"])
parser.add_argument("--default_schedule", action="store_true")
parser.add_argument("--wdtype", type=str, default="uint16",
                    choices=["float32", "uint16", "int8", "compare"])
parser.add_argument("--witype", type=str, default="int32",
                    choices=["int32", "uint16", "compare"])
parser.add_argument("--sdense", type=str, default="False",
                    choices=["False", "True", "compare"])
parser.add_argument("--graph", type=str)
parser.add_argument("--lib", type=str)
parser.add_argument("--params", type=str)
parser.add_argument("--real_data", action="store_true")

args = parser.parse_args()

if args.align_entries:
    align_entries = args.align_entries
    os.environ["TVM_SDENSE_ALIGN_ENTRIES"] = str(args.align_entries)

if args.num_threads > 0:
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

if args.debug:
    from tvm.contrib.debugger import debug_runtime as graph_runtime
    logging.basicConfig(level=logging.DEBUG)
else:
    import tvm.contrib.graph_runtime as graph_runtime


wdtype = args.wdtype
witype = args.witype

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

rnn_dims = 1024
fc_dims = 1024

feat_dims = 19
aux_dims = 64
n_classes = 2 ** 8

T = 8

x_0 = torch.randn(1, 1)
h1_0 = torch.randn(1, rnn_dims)
m = torch.randn(1, T, feat_dims)
a1 = torch.randn(1, T, aux_dims)
a2 = torch.randn(1, T, aux_dims)

if args.real_data:
    import data_loader
    (I, rnn1, fc1, fc2) = data_loader.load_pytorch_params()
else:
    I = nn.Linear(feat_dims + aux_dims + 1, rnn_dims)
    rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
    fc1 = nn.Linear(rnn_dims + aux_dims, fc_dims)
    fc2 = nn.Linear(fc_dims, n_classes)

    rnn1.weight_ih[:, :] = torch.tensor(sparsify(rnn1.weight_ih.detach().numpy(), BS_R=16, BS_C=1, density=0.06))
    rnn1.weight_hh[:, :] = torch.tensor(sparsify(rnn1.weight_hh.detach().numpy(), BS_R=16, BS_C=1, density=0.06))

    fc2.weight[:, :] = torch.tensor(sparsify(fc2.weight.detach().numpy(), BS_R=16, BS_C=1, density=0.2))

    fc1.weight[:, :rnn_dims] = torch.tensor(sparsify(fc1.weight[:, :rnn_dims].detach().numpy(), BS_R=16, BS_C=1, density=0.06))

def tvm_random_seed(seed):
    tvm.get_global_func("tvm.contrib.wavernn.set_seed")(seed)

def tvm_random_uniform():
    return tvm.get_global_func("tvm.contrib.wavernn.random_uniform")()


def sample(x_prob):
    gumbel = -torch.log(-torch.log(torch.tensor(np.random.uniform(size=x_prob.shape).astype("float32"))))
    result = np.zeros((1, 1), dtype="float32")
    result[:] = np.argmax(x_prob - gumbel)
    return torch.tensor(result / n_classes)

def sample_proba(x_prob):
    rand_sample = 0
    prob_sum = x_prob[0][0]
    rand = tvm_random_uniform()

    while prob_sum < rand:
        rand_sample += 1
        prob_sum += x_prob[0][rand_sample]

    result = np.zeros((1, 1), dtype="float32")
    result[:] = rand_sample
    return torch.tensor(result / n_classes)

def reference_frame(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    (x, h1) = (x_0, h1_0)
    T = a1.shape[1]
    outs = []
    for t in range(T):
        xconcat = torch.cat([x, m[0, t:t+1], a1[0, t:t+1]], dim=1)
        xconcat_trns = I(xconcat)
        h1 = rnn1(xconcat_trns, h1)
        xres = xconcat_trns + h1
        xres_concat = torch.cat([xres, a2[0, t:t+1]], dim=1)
        x_fc = F.relu(fc1(xres_concat))
        x_prob = fc2(x_fc)
        x_prob = torch.softmax(x_prob, dim=1)
        x = sample_proba(x_prob)
        outs.append(x)
    return outs, h1


def reference():
    return reference_frame(a1, a2, m, x_0, h1_0)

def test_pytorch_reference():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = reference_frame(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new)
        np.testing.assert_allclose(h1_ref, h1_new)

def factored_premul_frame(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    I_residual =  m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)
    I_weight = I.weight[:]
    I_bias = I.bias[:]

    Ifactored = nn.Linear(1, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :1]
    Ifactored.bias[:] = I.bias[:]

    (x, h1) = (x_0, h1_0)
    outs = []
    T = a1.shape[1]
    for t in range(T):
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
        x_prob = torch.softmax(x_prob, dim=1)
        x = sample_proba(x_prob)
        outs.append(x)
    return outs, h1

def build_wavernn_module(target="llvm"):
    Ifactored = nn.Linear(1, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :1]
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

    Rx = relay.var("x", shape=[1, 1], dtype="float32")
    Rh1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")
    RI_residual = relay.var("I_residual", shape=[1, rnn_dims], dtype="float32")
    Rfc1_residual = relay.var("fc1_residual", shape=[1, fc_dims], dtype="float32")
    RI_W = relay.var("I_W", shape=(rnn_dims, 1), dtype="float32")
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

    Rfc2_W = relay.var("fc2_W", shape=(n_classes, fc_dims), dtype="float32")
    Rfc2_B = relay.var("fc2_B", shape=(n_classes,), dtype="float32")

    x_prob = relay.nn.softmax(dense(x_fc, Rfc2_W, Rfc2_B))

    outputs = relay.expr.Tuple([x_prob, h1])
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    func = relay.ir_pass.infer_type(func)
    graph, lib, params = relay.build_module.build(func, target=target, params=params)
    return (graph, lib, params)


def build_fast_wavernn_module(target="llvm", wdtype="uint16", witype="int32", sdense="False", tune=False, profile=False):
    Ifactored = nn.Linear(1, rnn_dims)
    Ifactored.weight[:, :] = I.weight[:, :1]
    Ifactored.bias[:] = I.bias[:]

    fc1factored = nn.Linear(rnn_dims, fc_dims)
    fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
    fc1factored.bias[:] = fc1.bias[:]

    # import pdb; pdb.set_trace()
    if args.merged_gru is not None:
        if args.merged_gru == "interleaving":
            rnn1_weight = torch.cat(
                            (torch.cat((rnn1.weight_ih[: 2 * rnn_dims, :], rnn1.weight_hh[: 2 * rnn_dims, :]), dim=1),
                             torch.cat((rnn1.weight_ih[2 * rnn_dims: , :], torch.zeros(rnn_dims, rnn_dims)), dim=1),
                             torch.cat((torch.zeros(rnn_dims, rnn_dims), rnn1.weight_hh[2 * rnn_dims: , :]), dim=1)),
                             dim=0).detach().numpy()
            rnn1_bias = torch.cat(
                            (torch.add(rnn1.bias_ih[:2 * rnn_dims], rnn1.bias_hh[:2 * rnn_dims]),
                             rnn1.bias_ih[2 * rnn_dims:],
                             rnn1.bias_hh[2 * rnn_dims:])).detach().numpy()
        else:
            rnn1_weight = torch.cat(
                            (torch.cat((rnn1.weight_ih[: rnn_dims, :], torch.zeros(rnn_dims, rnn_dims)), dim=1),
                             torch.cat((rnn1.weight_ih[2 * rnn_dims:, :], torch.zeros(rnn_dims, rnn_dims)), dim=1),
                             torch.cat((rnn1.weight_ih[rnn_dims: 2 * rnn_dims, :], torch.zeros(rnn_dims, rnn_dims)), dim=1),
                             torch.cat((torch.zeros(rnn_dims * 3, rnn_dims), rnn1.weight_hh), dim=1)),
                             dim=0).detach().numpy()
            rnn1_bias = torch.cat(
                            (rnn1.bias_ih[:rnn_dims], rnn1.bias_ih[2 * rnn_dims:], rnn1.bias_ih[rnn_dims: 2 * rnn_dims],
                             rnn1.bias_hh)).detach().numpy()
        params = {
            "I_W": tvm.ndarray.array(Ifactored.weight.detach().numpy()),
            "I_B": tvm.ndarray.array(Ifactored.bias.detach().numpy()),
            "fc1_B": tvm.ndarray.array(fc1factored.bias.detach().numpy()),
            "fc2_W": tvm.ndarray.array(fc2.weight.detach().numpy()),
            "fc2_B": tvm.ndarray.array(fc2.bias.detach().numpy()),
            "rnn1_weight": tvm.ndarray.array(rnn1_weight),
            "rnn1_bias": tvm.ndarray.array(rnn1_bias),
        }
    else:
        params = {
            "I_W": tvm.ndarray.array(Ifactored.weight.detach().numpy()),
            "I_B": tvm.ndarray.array(Ifactored.bias.detach().numpy()),
            "fc1_B": tvm.ndarray.array(fc1factored.bias.detach().numpy()),
            "fc2_W": tvm.ndarray.array(fc2.weight.detach().numpy()),
            "fc2_B": tvm.ndarray.array(fc2.bias.detach().numpy()),
            "rnn1_weight_ih": tvm.ndarray.array(rnn1.weight_ih.detach().numpy()),
            "rnn1_weight_hh": tvm.ndarray.array(rnn1.weight_hh.detach().numpy()),
            "rnn1_bias_ih": tvm.ndarray.array(rnn1.bias_ih.detach().numpy()),
            "rnn1_bias_hh": tvm.ndarray.array(rnn1.bias_hh.detach().numpy()),
        }

    Rx = relay.var("x", shape=[1, 1], dtype="float32")
    Rh1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")
    RI_residual = relay.var("I_residual", shape=[1, rnn_dims], dtype="float32")
    Rfc1_residual = relay.var("fc1_residual", shape=[1, fc_dims], dtype="float32")
    RI_W = relay.var("I_W", shape=(rnn_dims, 1), dtype="float32")
    RI_B = relay.var("I_B", shape=(rnn_dims,), dtype="float32")

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
        if sdense == "True":
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

    def gru_cell2(cell2, x, h):
        xh = relay.concatenate((x, h), axis=1)
        xht = sparse_dense(xh, cell2.weight, cell2.bias)
        xht_split = relay.split(xht, indices_or_sections=4, axis=1)
        reset_gate = approx_sigmoid(xht_split[0])
        input_gate = approx_sigmoid(xht_split[1])
        new_gate = approx_tanh(xht_split[2] + reset_gate * xht_split[3])
        return new_gate + input_gate * (h - new_gate)

    def gru_cell3(cell2, x, h):
        xh = relay.concatenate((x, h), axis=1)
        xht = sparse_dense(xh, cell2.weight, cell2.bias)
        xht_split = relay.split(xht, indices_or_sections=6, axis=1)
        reset_gate = approx_sigmoid(xht_split[0] + xht_split[3])
        input_gate = approx_sigmoid(xht_split[2] + xht_split[4])
        new_gate = approx_tanh(xht_split[1] + reset_gate * xht_split[5])
        return new_gate + input_gate * (h - new_gate)

    def to_bf16(x):
        assert x.dtype == np.float32
        return ((x.view('<u4') + 2 ** 15) >> 16).astype("uint16")

    def to_sparse(v, arr, BS_R=16, BS_C=1):
        def convert_sparse(sparse):
            data = np.array(sparse.data)
            indices = np.array(sparse.indices)
            indptr = np.array(sparse.indptr)
            if align_entries == 1 or sdense != "True":
                return (data, indices, indptr)
            for i in range(indptr.shape[0]):
                idx = indptr[i]
                while idx % align_entries != 0:
                    # not efficient
                    data = np.insert(data, idx, np.zeros(data[idx-1].shape), axis=0)
                    indices = np.insert(indices, idx, indices[idx-1], axis=0)
                    # not efficient
                    for j in range(i, indptr.shape[0], 1):
                        indptr[j] = indptr[j] + 1
                    idx = indptr[i]
            indptr = indptr / align_entries
            return (data, indices, indptr)

        name = v.name_hint
        (N, K) = v.type_annotation.concrete_shape
        assert (N, K) == arr.shape
        sp_arr = sp.bsr_matrix(arr, blocksize=(BS_R, BS_C))
        (data, indices, indptr) = convert_sparse(sp_arr)
        # import pdb; pdb.set_trace()
        nnz = sp_arr.getnnz()
        # import pdb; pdb.set_trace()
        indptr_type = "int32"
        if nnz < 2 ** 16:
            indptr_type = "uint16"
        # print("Sparsity achieved: {:.2f}%".format((1.0 - float(sp_arr.data.size) / arr.size) * 100))
        v_data = relay.var(name + "_data", shape=data.shape, dtype=wdtype)
        v_indices = relay.var(name + "_indices", shape=indices.shape, dtype=witype)
        v_indptr = relay.var(name + "_indptr", shape=indptr.shape, dtype=indptr_type)
        params[name + "_data"] = tvm.ndarray.array(data.astype(wdtype)) if wdtype != "uint16" else tvm.ndarray.array(to_bf16(data))
        params[name + "_indices"] = tvm.ndarray.array(indices.astype(witype))
        params[name + "_indptr"] = tvm.ndarray.array(indptr.astype(indptr_type))
        return BSR(data=v_data, indices=v_indices, indptr=v_indptr)

    xconcat_trns = dense(Rx, RI_W, RI_B) + RI_residual

    if args.merged_gru == "interleaving":
        Cell = collections.namedtuple('Cell', ['weight', 'bias'])
        Rrnn1 = Cell(
            weight=to_sparse(relay.var("rnn1_weight", shape=(4 * rnn_dims, 2 * rnn_dims), dtype="float32"), rnn1_weight),
            bias=relay.var("rnn1_bias", shape=(4 * rnn_dims, ), dtype="float32")
        )
        h1 = gru_cell2(Rrnn1, xconcat_trns, Rh1)
    elif args.merged_gru == "simple":
        Cell = collections.namedtuple('Cell', ['weight', 'bias'])
        Rrnn1 = Cell(
            weight=to_sparse(relay.var("rnn1_weight", shape=(6 * rnn_dims, 2 * rnn_dims), dtype="float32"), rnn1_weight),
            bias=relay.var("rnn1_bias", shape=(6 * rnn_dims, ), dtype="float32")
        )
        h1 = gru_cell3(Rrnn1, xconcat_trns, Rh1)
    else:
        Cell = collections.namedtuple('Cell', ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])
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

    Rfc2_W = to_sparse(relay.var("fc2_W", shape=(n_classes, fc_dims), dtype="float32"), fc2.weight.detach().numpy())
    Rfc2_B = relay.var("fc2_B", shape=(n_classes,), dtype="float32")

    x_prob_unnorm = approx_exp(sparse_dense(x_fc, Rfc2_W, Rfc2_B))

    x_prob_sum = relay.sum(x_prob_unnorm, axis=-1)
    x_prob = x_prob_unnorm / x_prob_sum
    outputs = relay.expr.Tuple([x_prob, h1])
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    func = relay.ir_pass.infer_type(func)
    # print(func.astext())
    TARGET = tvm.target.create(target)
    log_filename = "lpcnet_no_bf16_autotvm_skl.log"

    if tune:
        with relay.build_config(opt_level=2):
            func = relay.optimize(func, target=TARGET, params=params)
            # print(func.astext(show_meta_data=False))
            tasks = autotvm.task.extract_from_program(
                func, target=TARGET, params=params, ops=(relay.op.nn.dense,))
            for i, tsk in enumerate(tasks):
                # print(tsk)
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
            "x": tvm.nd.array(np.random.randn(1, 1).astype(np.float32)),
            "h1": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
            "I_residual": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
            "fc1_residual": tvm.nd.array(np.random.randn(1, rnn_dims).astype(np.float32)),
        }
        r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
        r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
        module = graph_runtime.create(graph, rlib, ctx)
        if args.debug:
            print(rlib.get_source())
            print(rlib.get_source('asm'))
        module.set_input(**r_new_params)
        module.set_input(**r_inputs)
        ftimer = module.module.time_evaluator("run", ctx, number=10000)
        for i in range(5):
            prof_res = ftimer()
            print("TVM time: {:.2f}us".format(prof_res.mean * 10 ** 6))
        module.run()
        module.run()

    return (graph, lib, new_params)

def factored_relay_frame(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    (x, h1) = (tvm.ndarray.array(x_0), tvm.ndarray.array(h1_0))
    outs = []
    T = a1.shape[1]
    (graph, lib, params) = build_wavernn_module()
    module = graph_runtime.create(graph, lib, tvm.cpu(0))
    module.set_input(**params)

    I_residual = m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    for t in range(T):
        inputs = {
            "x": x,
            "h1": h1,
            "I_residual": tvm.ndarray.array(I_residual[t:t+1].detach().numpy()),
            "fc1_residual": tvm.ndarray.array(fc1_residual[t:t+1].detach().numpy()),
        }

        module.set_input(**inputs)
        module.run()
        (x_prob, h1) = module.get_output(0), module.get_output(1)
        x = tvm.ndarray.array(sample_proba(torch.tensor(x_prob.asnumpy())))
        outs.append(x.asnumpy()[0][0])
    return outs, h1.asnumpy()

def factored_relay_frame_fast(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    (x, h1) = (tvm.ndarray.array(x_0), tvm.ndarray.array(h1_0))
    outs = []
    T = a1.shape[1]
    (graph, lib, params) = build_fast_wavernn_module()
    module = graph_runtime.create(graph, lib, tvm.cpu(0))
    module.set_input(**params)

    I_residual = m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    for t in range(T):
        inputs = {
            "x": x,
            "h1": h1,
            "I_residual": tvm.ndarray.array(I_residual[t:t+1].detach().numpy()),
            "fc1_residual": tvm.ndarray.array(fc1_residual[t:t+1].detach().numpy()),
        }

        module.set_input(**inputs)
        module.run()
        (x_prob, h1) = module.get_output(0), module.get_output(1)
        x = tvm.ndarray.array(sample_proba(torch.tensor(x_prob.asnumpy())))
        outs.append(x.asnumpy()[0][0])
    return outs, h1.asnumpy()

def factored_relay_cpp_frame(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    (x, h1) = (tvm.ndarray.array(x_0), tvm.ndarray.array(h1_0))
    T = a1.shape[1]
    (graph, lib, params) = build_wavernn_module()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, prefix="tvm_model_lib", suffix=".so") as lib_f:
        lib.export_library(lib_f.name)
    # import pdb; pdb.set_trace()
    I_residual = m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    frame_func = tvm.get_global_func("tvm.contrib.wavernn.frame")

    outs = tvm.ndarray.array(np.random.randn(T).astype("float32"))
    h1 = tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32"))
    frame_func(
        # Inputs
        tvm.ndarray.array(I_residual),
        tvm.ndarray.array(fc1_residual),
        tvm.ndarray.array(x_0),
        tvm.ndarray.array(h1_0),
        # Outputs
        outs,
        h1,
        # Temporary storage to make entire frame_func allocation free.
        tvm.ndarray.array(np.random.randn(1, n_classes).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, fc_dims).astype("float32")),
        # Data for constructing the module.
        graph,  # the graph JSON.
        lib_f.name,  # the exported shared object.
        relay.save_param_dict(params)  # the serialized parameters.
    )
    return outs.asnumpy(), h1.asnumpy()

def factored_relay_cpp_frame_fast(a1, a2, m, x_0, h1_0):
    tvm_random_seed(10)
    (x, h1) = (tvm.ndarray.array(x_0), tvm.ndarray.array(h1_0))
    T = a1.shape[1]
    (graph, lib, params) = build_fast_wavernn_module(profile=False, sdense=args.sdense)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, prefix="tvm_model_lib", suffix=".so") as lib_f:
        lib.export_library(lib_f.name)
    I_residual = m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    frame_func = tvm.get_global_func("tvm.contrib.wavernn.frame")

    outs = tvm.ndarray.array(np.random.randn(T).astype("float32"))
    h1 = tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32"))
    frame_func(
        # Inputs
        tvm.ndarray.array(I_residual),
        tvm.ndarray.array(fc1_residual),
        tvm.ndarray.array(x_0),
        tvm.ndarray.array(h1_0),
        # Outputs
        outs,
        h1,
        # Temporary storage to make entire frame_func allocation free.
        tvm.ndarray.array(np.random.randn(1, n_classes).astype("float32")),
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
        outs_new, h1_new = factored_premul_frame(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_frame():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_frame(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_frame_fast():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_frame_fast(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_cpp_frame():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_cpp_frame(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_cpp_frame_fast():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = factored_relay_cpp_frame_fast(a1, a2, m, x_0, h1_0)
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-2, atol=1e-2)
        print(h1_ref, h1_new)
        print(outs_ref, outs_new)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-2, atol=1e-2)

def test(target):
    args0 = {}
    args1 = {}
    if args.wdtype != "compare":
        args0["wdtype"] = args.wdtype
        args1["wdtype"] = args.wdtype
    else:
        args0["wdtype"] = "int8"
        args1["wdtype"] = "uint16"
    if args.witype != "compare":
        args0["witype"] = args.witype
        args1["witype"] = args.witype
    else:
        args0["witype"] = "int32"
        args1["witype"] = "uint16"
    if args.sdense != "compare":
        args0["sdense"] = args.sdense
        args1["sdense"] = args.sdense
    else:
        args0["sdense"] = "False"
        args1["sdense"] = "True"
    print(args0)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args0)
    print(args1)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args1)
    print(args0)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args0)
    print(args1)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args1)
    print(args0)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args0)
    print(args1)
    (graph, lib, params) = build_fast_wavernn_module(target, profile=True, **args1)

def skylake():
    (graph, lib, params) = build_fast_wavernn_module("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu", wdtype=args.wdtype, witype="uint16", sdense=args.sdense, profile=True)
    with open(
            "skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_graph.json".format(**globals()),
            "w") as f:
        f.write(graph)

    with open(
            "skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_params.bin".format(**globals()),
            "wb") as f:
        f.write(relay.save_param_dict(params))

    lib.save("skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.o".format(**globals()))
    with open(
            "skl_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.so".format(**globals()),
            "wb") as f:
        lib.export_library(f.name)

def haswell():
    # import pdb; pdb.set_trace()
    (graph, lib, params) = build_fast_wavernn_module("llvm -mcpu=core-avx2", wdtype="uint16", witype="uint16", sdense="True", profile=True)
    with open(
            "hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_graph.json".format(**globals()),
            "w") as f:
        f.write(graph)
    with open(
            "hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.so".format(**globals()),
            "wb") as f:
        lib.export_library(f.name)
    with open(
            "hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_params.bin".format(**globals()),
            "wb") as f:
        f.write(relay.save_param_dict(params))

    (graph, lib, params) = build_fast_wavernn_module("llvm --system-lib -mcpu=core-avx2", wdtype="uint16", witype="uint16", sdense="True", profile=True)
    lib.save("hsw_fast_wavernn_rnn_dims_{rnn_dims}_fc_dims_{fc_dims}_feat_dims_{feat_dims}_aux_dims_{aux_dims}_lib.o".format(**globals()))


def load_tvm(graph_name, lib_name, params_name):
    with open(graph_name, "r") as f:
        graph = f.read()
    lib = tvm.module.load(lib_name)
    with open(params_name, "rb") as f:
        params = relay.load_param_dict(f.read())
    return (graph, lib, params)

def exec_loaded_tvm():
    graph_name = args.graph
    lib_name = args.lib
    params_name = args.params
    with open(graph_name, "r") as f:
        graph = f.read()
    with open(params_name, "rb") as f:
        serialized_params = bytearray(f.read())

    tvm_random_seed(10)
    T = a1.shape[1]
    # (graph, lib, params) = load_tvm(graph_name, lib_name, params_name)
    # import tempfile
    # with tempfile.NamedTemporaryFile(delete=False, prefix="tvm_model_lib", suffix=".so") as lib_f:
    #     lib.export_library(lib_f.name)
    concat = np.concatenate((m[0], a1[0]), axis=1)
    I_residual = concat @ I.weight[:, 1:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)
    frame_func = tvm.get_global_func("tvm.contrib.wavernn.frame")

    outs = tvm.ndarray.array(np.random.randn(T).astype("float32"))
    h1 = tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32"))
    frame_func(
        # Inputs
        tvm.ndarray.array(I_residual.detach().numpy()),
        tvm.ndarray.array(fc1_residual.detach().numpy()),
        tvm.ndarray.array(x_0.detach().numpy()),
        tvm.ndarray.array(h1_0.detach().numpy()),
        # Outputs
        outs,
        h1,
        # Temporary storage to make entire frame_func allocation free.
        tvm.ndarray.array(np.random.randn(1, n_classes).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, rnn_dims).astype("float32")),
        tvm.ndarray.array(np.random.randn(1, fc_dims).astype("float32")),
        # Data for constructing the module.
        graph,  # the graph JSON.
        lib_name,  # the exported shared object.
        serialized_params  # the serialized parameters.
    )
    return outs.asnumpy(), h1.asnumpy()

def test_load():
    with torch.no_grad():
        outs_ref, h1_ref = reference()
        outs_new, h1_new = exec_loaded_tvm()
        np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

# test_relay_frame_fast()
# test_relay_cpp_frame()
if args.verify:
    test_relay_cpp_frame_fast()
else:
# test("llvm -mcpu=core-avx2 -target=x86_64-linux-gnu")
    test("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu")
# skylake()
# haswell()
# test_load()
