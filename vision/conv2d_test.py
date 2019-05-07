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

dtype = "float32"

BATCH = 1
IN_CHANNEL = 16
HEIGHT = 224
WIDTH = 224

OUT_CHANNEL = 16
K_HEIGHT = 3
K_WIDTH = 3

OUT_HEIGHT = 224
OUT_WIDTH = 224

PADDING = 1

ishape = [BATCH, IN_CHANNEL, HEIGHT, WIDTH]
oshape = [BATCH, OUT_CHANNEL, OUT_HEIGHT, OUT_WIDTH]
wshape = [OUT_CHANNEL, IN_CHANNEL, K_HEIGHT, K_WIDTH]

a = torch.rand(ishape)
b = torch.rand(wshape)
res = torch.nn.functional.conv2d(a, b, padding=1)
# import pdb; pdb.set_trace()

params = {
    "weight": tvm.nd.array(b.detach().numpy().astype(dtype), ctx)
}

inputs = {
    "data": tvm.nd.array(a.detach().numpy().astype(dtype), ctx)
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
print("conv2d: [{}, {}, {}, {}] x [{}, {}, {}, {}]".format(BATCH, IN_CHANNEL,
      HEIGHT, WIDTH, OUT_CHANNEL, IN_CHANNEL, K_HEIGHT, K_WIDTH))
data = relay.var("data", shape=ishape, dtype=dtype)
weight = relay.var("weight", shape=wshape, dtype=dtype)
# import pdb; pdb.set_trace()
# zero = relay.var("zero", shape=(M, N), dtype=dtype)
outputs = relay.nn.conv2d(data, weight, data_layout="NCHW", kernel_layout="OIHW",
                          channels=OUT_CHANNEL, padding=(PADDING, PADDING), strides=(1,1),
                          dilation=(1,1), groups=1, kernel_size=(K_HEIGHT,K_WIDTH))

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
tvm.testing.assert_allclose(ro.asnumpy(), res, rtol=1e-5)

ftimer = module.module.time_evaluator("run", ctx, 1000)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)
