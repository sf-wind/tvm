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
import time
import torch
from torchvision.models.resnet import resnet152, resnet101, resnet50, resnet34, resnet18

sys.settrace

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--tune", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_threads", type=int, default=0)
parser.add_argument("--tuner", type=str, default="xgboost",
                    choices=["ga", "xgboost"])
parser.add_argument("--target", type=str, default="core-avx2",
                    choices=["core-avx2", "skylake-avx512"])
parser.add_argument("--default_schedule", action="store_true")
parser.add_argument("--layer", type=int, default=-1)
parser.add_argument("--sublayer", type=int, default=1)
parser.add_argument("--model", type=str, default="resnet152",
                    choices=["resnet152", "resnet18", "resnet50", "resnet34", "resnet101"])
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

context = "llvm -mcpu=" + args.target
skl_target = tvm.target.create(context)
ctx = tvm.context(context, 0)

BATCH = 1
HEIGHT = 56
WIDTH = 56

dtype = "float32"

if args.model == "resnet152":
    pytorch_resnet = resnet152(pretrained=True)
elif args.model == "resnet101":
    pytorch_resnet = resnet101(pretrained=True)
elif args.model == "resnet50":
    pytorch_resnet = resnet50(pretrained=True)
elif args.model == "resnet34":
    pytorch_resnet = resnet34(pretrained=True)
elif args.model == "resnet18":
    pytorch_resnet = resnet18(pretrained=True)
else:
    assert False

pytorch_resnet.eval()

# import pdb; pdb.set_trace()


def conv(x, pbn_conv, name, params):
    weight = relay.var(name + ".weight", shape=pbn_conv.weight.shape)
    c = relay.nn.conv2d(x, weight,
                        data_layout="NCHW", kernel_layout="OIHW",
                        channels=pbn_conv.weight.shape[0],
                        padding=pbn_conv.padding,
                        strides=pbn_conv.stride,
                        dilation=pbn_conv.dilation,
                        groups=pbn_conv.groups,
                        kernel_size=pbn_conv.kernel_size)
    params[name + ".weight"] = pbn_conv.weight
    if pbn_conv.bias is not None:
        bias = relay.var(name + ".bias", shape=pbn_conv.bias.shape)
        c = relay.nn.bias_add(c, bias)
        params[name + ".bias"] = pbn_conv.bias
    return c


def bn(x, pbn_bn, name, params):
    weight = relay.var(name + ".weight", shape=pbn_bn.weight.shape)
    bias = relay.var(name + ".bias", shape=pbn_bn.bias.shape)
    running_var = relay.var(name + ".running_var",
                            shape=pbn_bn.running_var.shape)
    running_mean = relay.var(name + ".running_mean",
                             shape=pbn_bn.running_mean.shape)
    c = relay.nn.batch_norm(x, weight, bias, running_mean, running_var,
                            epsilon=pbn_bn.eps)
    params[name + ".weight"] = pbn_bn.weight
    params[name + ".bias"] = pbn_bn.bias
    params[name + ".running_var"] = pbn_bn.running_var
    params[name + ".running_mean"] = pbn_bn.running_mean
    return c[0]


def downsample(x, ds, prefix, params):
    x = conv(x, ds[0], prefix + "." + "downsample.conv", params)
    x = bn(x, ds[1], prefix + "." + "downsample.bn", params)
    return x


def maxpool(x, pool):
    c = relay.nn.max_pool2d(x, pool_size=(pool.kernel_size, pool.kernel_size),
                            strides=(pool.stride, pool.stride),
                            padding=(pool.padding, pool.padding),
                            ceil_mode=pool.ceil_mode)
    return c


def fc(x, linear, name, params):
    weight = relay.var(name + ".weight", shape=linear.weight.shape)
    c = relay.nn.dense(x, weight)
    params[name + ".weight"] = linear.weight
    if linear.bias is not None:
        bias = relay.var(name + ".bias", shape=linear.bias.shape)
        c = relay.nn.bias_add(c, bias)
        params[name + ".bias"] = linear.bias
    return c


def get_tvm_params(params):
    tvm_params = { k : tvm.nd.array(v.detach().numpy(), ctx)
                  for k,v in params.items() if v is not None}
    return tvm_params


class Bottleneck():
    def __init__(self, pbn, prefix):
        self.pbn = pbn
        self.prefix = prefix
        self.inplanes = pbn.conv1.in_channels
        self.width = pbn.conv1.out_channels
        self.stride = pbn.conv2.stride
        self.groups = pbn.conv2.groups
        self.padding = pbn.conv2.padding
        self.dilation = pbn.conv2.dilation
        self.bottleneck = args.model != "resnet18" and args.model != "resnet34"

    def get_tvm(self, input, params):
        # identity = relay.copy(x)
        x = input
        x = conv(x, self.pbn.conv1, self.prefix + "." + "conv1", params)
        x = bn(x, self.pbn.bn1, self.prefix + "." + "bn1", params)
        x = relay.nn.relu(x)
        x = conv(x, self.pbn.conv2, self.prefix + "." + "conv2", params)
        x = bn(x, self.pbn.bn2, self.prefix + "." + "bn2", params)
        if self.bottleneck:
            x = relay.nn.relu(x)
            x = conv(x, self.pbn.conv3, self.prefix + "." + "conv3", params)
            x = bn(x, self.pbn.bn3, self.prefix + "." + "bn3", params)
        if self.pbn.downsample:
            input = downsample(input, self.pbn.downsample, self.prefix, params)
        x = relay.add(x, input)
        x = relay.nn.relu(x)
        return x


class Resnet():
    def __init__(self, pytorch_resnet):
        self.pytorch_resnet = pytorch_resnet


    def make_layer(self, x, pytorch_layer, prefix, params):
        num = len(pytorch_layer)
        for i in range(num):
            pytorch_block = pytorch_layer[i]
            name = prefix + ".block" + str(i)
            block = Bottleneck(pytorch_block, name)
            x = block.get_tvm(x, params)
        return x

    def get_tvm(self, x, params):
        x = conv(x, self.pytorch_resnet.conv1, "layer0.conv1", params)
        x = bn(x, self.pytorch_resnet.bn1, "layer0.bn1", params)
        x = relay.nn.relu(x)
        x = maxpool(x, self.pytorch_resnet.maxpool)
        x = self.make_layer(x, self.pytorch_resnet.layer1, "layer1", params)
        x = self.make_layer(x, self.pytorch_resnet.layer2, "layer2", params)
        x = self.make_layer(x, self.pytorch_resnet.layer3, "layer3", params)
        x = self.make_layer(x, self.pytorch_resnet.layer4, "layer4", params)
        x = relay.nn.global_avg_pool2d(x)
        x = relay.reshape(x, [-1, self.pytorch_resnet.fc.weight.shape[1]])
        x = fc(x, self.pytorch_resnet.fc, "fc0", params)
        return x
# import pdb; pdb.set_trace()


def tune():
    global func
    with relay.build_config(opt_level=2):
        func = relay.optimize(func, target=skl_target, params=params)
        print(func.astext(show_meta_data=False))
        # import pdb; pdb.set_trace()
        tasks = autotvm.task.extract_from_program(
            func, target=skl_target, params=params, ops=(relay.op.nn.dense,
                                                         relay.op.nn.conv2d))
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


def build_graph():
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

# import pdb; pdb.set_trace()

if args.layer < 0:
    ishape = [BATCH, 3, 224, 224]

    input = torch.randn(ishape)

    with torch.no_grad():
        res = pytorch_resnet(input)
    model = Resnet(pytorch_resnet)

else:
    if args.layer == 1:
        layer = pytorch_resnet.layer1
    elif args.layer == 2:
        layer = pytorch_resnet.layer2
    elif args.layer == 3:
        layer = pytorch_resnet.layer3
    elif args.layer == 4:
        layer = pytorch_resnet.layer4
    else:
        assert False, "Layer out of bound"

    sublayer = layer[args.sublayer]

    ishape = [BATCH, sublayer.conv1.in_channels, HEIGHT, WIDTH]

    input = torch.randn(ishape)

    with torch.no_grad():
        res = sublayer(input)
    model = Bottleneck(sublayer, "layer" + str(args.layer))

# import pdb; pdb.set_trace()
oshape = res.shape
data = relay.var("data", shape=ishape, dtype=dtype)
pytorch_params = {}

outputs = model.get_tvm(data, pytorch_params)
params = get_tvm_params(pytorch_params)

inputs = {
    "data": tvm.nd.array(input.detach().numpy().astype(dtype), ctx)
}


func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))

if args.tune:
    tune()
    import sys
    sys.exit()

if args.default_schedule:
    (graph, lib, new_params) = build_graph()
else:
    with autotvm.apply_history_best("synthesis_autotvm_skl.log"):
        (graph, lib, new_params) = build_graph()

r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty(oshape)
module.get_output(0, ro)
# import pdb; pdb.set_trace()
tvm.testing.assert_allclose(ro.asnumpy(), res, rtol=1e-5, atol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 100)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
