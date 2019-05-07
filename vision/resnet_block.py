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
from torchvision.models.resnet import resnet152

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
parser.add_argument("--layer", type=int, default=1)
parser.add_argument("--sublayer", type=int, default=1)
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

pytorch_resnet = resnet152(pretrained=True)
pytorch_resnet.eval()

'''
ishape = [BATCH, 3, 224, 224]

input = torch.randn(ishape)

with torch.no_grad():
    output = pytorch_resnet(input)
'''

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
# ishape = [1, 64, 1, 1]
input = torch.randn(ishape)

with torch.no_grad():
    res = sublayer(input)

oshape = res.shape
# import pdb; pdb.set_trace()

class Bottleneck():
    def __init__(self, pbn):
        self.pbn = pbn
        self.inplanes = pbn.conv1.in_channels
        self.width = pbn.conv1.out_channels
        self.stride = pbn.conv2.stride
        self.groups = pbn.conv2.groups
        self.padding = pbn.conv2.padding
        self.dilation = pbn.conv2.dilation
        self.outplanes = pbn.conv3.out_channels
        self.params = {
            "conv1.weight": pbn.conv1.weight,
            "conv1.bias": pbn.conv1.bias,
            "bn1.weight": pbn.bn1.weight,
            "bn1.bias": pbn.bn1.bias,
            "bn1.running_var": pbn.bn1.running_var,
            "bn1.running_mean": pbn.bn1.running_mean,
            "conv2.weight": pbn.conv2.weight,
            "conv2.bias": pbn.conv2.bias,
            "bn2.weight": pbn.bn2.weight,
            "bn2.bias": pbn.bn2.bias,
            "bn2.running_var": pbn.bn2.running_var,
            "bn2.running_mean": pbn.bn2.running_mean,
            "conv3.weight": pbn.conv3.weight,
            "conv3.bias": pbn.conv3.bias,
            "bn3.weight": pbn.bn3.weight,
            "bn3.bias": pbn.bn3.bias,
            "bn3.running_var": pbn.bn3.running_var,
            "bn3.running_mean": pbn.bn3.running_mean,
        }
        if pbn.downsample:
            self.params["downsample.conv.weight"] = pbn.downsample[0].weight
            self.params["downsample.conv.bias"] = pbn.downsample[0].bias
            self.params["downsample.bn.weight"] = pbn.downsample[1].weight
            self.params["downsample.bn.bias"] = pbn.downsample[1].bias
            self.params["downsample.bn.running_var"] = pbn.downsample[1].running_var
            self.params["downsample.bn.running_mean"] = pbn.downsample[1].running_mean

    def get_tvm_params(self):
        tvm_params = { k : tvm.nd.array(v.detach().numpy(), ctx)
                      for k,v in self.params.items() if v is not None}
        return tvm_params

    def get_tvm_block(self, x):
        def conv(x, pbn_conv, name):
            weight = relay.var(name + ".weight", shape=pbn_conv.weight.shape)
            c = relay.nn.conv2d(x, weight,
                                data_layout="NCHW", kernel_layout="OIHW",
                                channels=pbn_conv.weight.shape[0],
                                padding=pbn_conv.padding,
                                strides=pbn_conv.stride,
                                dilation=pbn_conv.dilation,
                                groups=pbn_conv.groups,
                                kernel_size=pbn_conv.kernel_size)
            if pbn_conv.bias is not None:
                bias = relay.var(name + ".bias", shape=pbn_conv.bias.shape)
                c = relay.nn.bias_add(c, bias)
            return c

        def bn(x, pbn_bn, name):
            weight = relay.var(name + ".weight", shape=pbn_bn.weight.shape)
            bias = relay.var(name + ".bias", shape=pbn_bn.bias.shape)
            running_var = relay.var(name + ".running_var",
                                    shape=pbn_bn.running_var.shape)
            running_mean = relay.var(name + ".running_mean",
                                     shape=pbn_bn.running_mean.shape)
            c = relay.nn.batch_norm(x, weight, bias, running_mean, running_var,
                                    epsilon=pbn_bn.eps)
            return c[0]

        def downsample(x, ds):
            x = conv(x, ds[0], "downsample.conv")
            x = bn(x, ds[1], "downsample.bn")
            return x

        identity = relay.copy(x)
        x = conv(x, self.pbn.conv1, "conv1")
        x = bn(x, self.pbn.bn1, "bn1")
        x = relay.nn.relu(x)
        x = conv(x, self.pbn.conv2, "conv2")
        x = bn(x, self.pbn.bn2, "bn2")
        x = relay.nn.relu(x)
        x = conv(x, self.pbn.conv3, "conv3")
        x = bn(x, self.pbn.bn3, "bn3")
        if self.pbn.downsample:
            x = downsample(x, self.pbn.downsample)
        x = relay.add(x, identity)
        x = relay.nn.relu(x)
        return x



# import pdb; pdb.set_trace()

tvm_bottleneck = Bottleneck(sublayer)

params = tvm_bottleneck.get_tvm_params()

inputs = {
    "data": tvm.nd.array(input.detach().numpy().astype(dtype), ctx)
}


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

data = relay.var("data", shape=ishape, dtype=dtype)
# import pdb; pdb.set_trace()
# zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = tvm_bottleneck.get_tvm_block(data)

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))



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

ftimer = module.module.time_evaluator("run", ctx, 1000)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
