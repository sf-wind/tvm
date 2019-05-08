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
parser.add_argument("--layout", type=str, default="NCHW",
                    choices=["NCHW", "NHWC"])
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

log_filename = "synthesis_autotvm_{}.log".format(args.target)
context = "llvm -mcpu=" + args.target
skl_target = tvm.target.create(context)
ctx = tvm.context(context, 0)

dtype = "float32"

BATCH = 1
IN_CHANNEL = 64
HEIGHT = 56
WIDTH = 56

OUT_CHANNEL = 128
K_HEIGHT = 3
K_WIDTH = 3

OUT_HEIGHT = 56
OUT_WIDTH = 56

H_PADDING = (HEIGHT - OUT_HEIGHT + K_HEIGHT - 1) // 2
W_PADDING = (WIDTH - OUT_WIDTH + K_WIDTH - 1) // 2

ishape = [BATCH, IN_CHANNEL, HEIGHT, WIDTH]
oshape = [BATCH, OUT_CHANNEL, OUT_HEIGHT, OUT_WIDTH]
wshape = [OUT_CHANNEL, IN_CHANNEL, K_HEIGHT, K_WIDTH]

a = torch.rand(ishape)
b = torch.rand(wshape)
res = torch.nn.functional.conv2d(a, b, padding=(H_PADDING, W_PADDING))
# import pdb; pdb.set_trace()


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
print("conv2d: [{}, {}, {}, {}] x [{}, {}, {}, {}]".format(BATCH, IN_CHANNEL,
      HEIGHT, WIDTH, OUT_CHANNEL, IN_CHANNEL, K_HEIGHT, K_WIDTH))

iishape = ishape
wwshape = wshape
ooshape = oshape
aa = a.detach().numpy().astype(dtype)
bb = b.detach().numpy().astype(dtype)
data_layout = args.layout
kernel_layout = "OIHW"
if args.layout == "NHWC":
    iishape = [BATCH, HEIGHT, WIDTH, IN_CHANNEL]
    wwshape = [K_HEIGHT, K_WIDTH, IN_CHANNEL, OUT_CHANNEL]
    ooshape = [BATCH, OUT_HEIGHT, OUT_WIDTH, OUT_CHANNEL]
    aa = np.transpose(aa, (0, 2, 3, 1))
    bb = np.transpose(bb, (2, 3, 1, 0))
    kernel_layout = "HWIO"

data = relay.var("data", shape=iishape, dtype=dtype)
weight = relay.var("weight", shape=wwshape, dtype=dtype)
# import pdb; pdb.set_trace()
# zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = relay.nn.conv2d(data, weight, data_layout=data_layout, kernel_layout=kernel_layout,
                          channels=OUT_CHANNEL, padding=(H_PADDING, W_PADDING), strides=(1,1),
                          dilation=(1,1), groups=1, kernel_size=(K_HEIGHT,K_WIDTH))

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
# print(func.astext(show_meta_data=False))
relay.ir_pass.infer_type(func)
# print(func.astext(show_meta_data=False))

params = {
    "weight": tvm.nd.array(bb, ctx)
}

inputs = {
    "data": tvm.nd.array(aa, ctx)
}

if args.tune:
    tune()
    import sys
    sys.exit()

if args.default_schedule:
    (graph, lib, new_params) = build_graph()
else:
    with autotvm.apply_history_best(log_filename):
        (graph, lib, new_params) = build_graph()

r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
ro = tvm.nd.empty(ooshape)
module.get_output(0, ro)
roo = ro.asnumpy()
# import pdb; pdb.set_trace()
if args.layout == "NHWC":
    roo = np.transpose(roo, (0, 3, 1, 2))
tvm.testing.assert_allclose(roo, res, rtol=1e-5, atol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 1000)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
