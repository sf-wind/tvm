import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)


rnn_dims = 512
fc_dims = 512

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

x = torch.randn(1, 1)
h1 = torch.randn(1, rnn_dims)
h2 = torch.randn(1, rnn_dims)
m_t = torch.randn(1, feat_dims)
a1_t = torch.randn(1, aux_dims)
a2_t = torch.randn(1, aux_dims)

I = nn.Linear(feat_dims + aux_dims + 1, rnn_dims)
rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
fc1 = nn.Linear(rnn_dims + aux_dims, fc_dims)
fc2 = nn.Linear(fc_dims, n_classes)

def reference_sample_one(a1_t, a2_t, m_t, x, h1):
    with torch.no_grad():
        xconcat = torch.cat([x, m_t, a1_t], dim=1)
        xconcat_trns = I(xconcat)
        h1 = rnn1(xconcat_trns, h1)
        xres = xconcat_trns + h1
        xres_concat = torch.cat([xres, a2_t], dim=1)
        x_fc = F.relu(fc1(xres_concat))
        x_prob = fc2(x_fc)
        return (xconcat_trns, x_prob, h1)



def reference():
    return reference_sample_one(a1_t, a2_t, m_t, x, h1)

def test_pytorch_reference():
    (xconcat_trns_ref, x_new_ref, h_new_ref) = reference()
    (xconcat_trns, x_new, h_new) = reference_sample_one(a1_t, a2_t, m_t, x, h1)
    np.testing.assert_allclose(xconcat_trns_ref, xconcat_trns)
    np.testing.assert_allclose(x_new_ref, x_new)
    np.testing.assert_allclose(h_new_ref, h_new)

def factored_sample_one(a1_t, a2_t, m_t, x, h1):
    with torch.no_grad():
        I_weight = I.weight[:]
        I_bias = I.bias[:]

        Ifactored = nn.Linear(1, rnn_dims)
        Ifactored.weight[:, :] = I.weight[:, :1]
        Ifactored.bias[:] = I.bias[:]


        xconcat_trns = Ifactored(x) + m_t @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1_t @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
        h1 = rnn1(xconcat_trns, h1)
        xres = xconcat_trns + h1
        xres_concat = torch.cat([xres, a2_t], dim=1)
        x_fc = F.relu(fc1(xres_concat))
        x_prob = fc2(x_fc)
        return (xconcat_trns, x_prob, h1)

def test_factored_reference():
    (xconcat_trns_ref, x_new_ref, h_new_ref) = reference()
    (xconcat_trns_new, x_new, h_new) = factored_sample_one(a1_t, a2_t, m_t, x, h1)
    np.testing.assert_allclose(xconcat_trns_ref, xconcat_trns_new, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(x_new_ref, x_new, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(h_new_ref, h_new, rtol=1e-4, atol=1e-4)
