#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

import topi
from topi.util import get_const_int, get_const_tuple
from .. import op as reg
from ..op import OpPattern, schedule_injective

# relu
reg.register_schedule("nn.relu", schedule_injective)
reg.register_pattern("nn.relu", OpPattern.ELEMWISE)

# softmax
@reg.register_schedule("nn.softmax")
def schedule_softmax(_, outputs, target):
    """Schedule definition of softmax"""
    with target:
        return topi.generic.schedule_softmax(outputs)

reg.register_pattern("nn.softmax", OpPattern.OPAQUE)

schedule_broadcast = schedule_injective

@reg.register_schedule("nn.log_softmax")
def schedule_log_softmax(_, outputs, target):
    """Schedule definition of log_softmax"""
    with target:
        return topi.generic.schedule_softmax(outputs)

reg.register_pattern("nn.log_softmax", OpPattern.OPAQUE)


# dense
@reg.register_compute("nn.dense")
def compute_dense(attrs, inputs, out_type, target):
    """Compute definition of dense"""
    data_layout = attrs.get_str("data_layout")
    kernel_layout = attrs.get_str("kernel_layout")
    out_layout = attrs.get_str("out_layout")
    return [topi.nn.dense(inputs[0], inputs[1], None,
                          data_layout,
                          kernel_layout,
                          out_layout)]

@reg.register_schedule("nn.dense")
def schedule_dense(attrs, outputs, target):
    """Schedule definition of dense"""
    with target:
        return topi.generic.schedule_dense(outputs)

@reg.register_alter_op_layout("nn.dense")
def alter_op_layout_conv2d(attrs, inputs, tinfos):
    """Alternate the layout of dense"""
    from ... import op
    return topi.nn.dense_alter_layout(attrs, inputs, tinfos, op)

reg.register_pattern("nn.dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# batch_matmul
@reg.register_compute("nn.batch_matmul")
def compute_batch_matmul(attrs, inputs, out_type, target):
    """Compute definition of batch_matmul"""
    return [topi.nn.batch_matmul(inputs[0], inputs[1])]

@reg.register_schedule("nn.batch_matmul")
def schedule_batch_matmul(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_batch_matmul(outputs)

reg.register_pattern("nn.batch_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sparse_dense
@reg.register_compute("nn.sparse_dense")
def compute_sparse_dense(attrs, inputs, out_type, target):
    """Compute definition of sparse_dense"""
    return [topi.nn.sparse_dense(inputs[0], inputs[1], inputs[2], inputs[3])]

@reg.register_schedule("nn.sparse_dense")
def schedule_sparse_dense(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sparse_dense(outputs)

reg.register_pattern("nn.sparse_dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sparse_dense2
@reg.register_compute("nn.sparse_dense2")
def compute_sparse_dense2(attrs, inputs, out_type, target):
    # import pdb; pdb.set_trace()
    """Compute definition of sparse_dense2"""
    return [topi.nn.sparse_dense2(inputs[0], inputs[1], inputs[2], inputs[3])]

@reg.register_schedule("nn.sparse_dense2")
def schedule_sparse_dense2(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sparse_dense2(outputs)

reg.register_pattern("nn.sparse_dense2", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sparse_dense_kmnk
@reg.register_compute("nn.sparse_dense_kmnk")
def compute_sparse_dense_kmnk(attrs, inputs, out_type, target):
    """Compute definition of sparse_dense_kmnk"""
    return [topi.nn.sparse_dense_kmnk(inputs[0], inputs[1], inputs[2],
                                           inputs[3])]

@reg.register_schedule("nn.sparse_dense_kmnk")
def schedule_sparse_dense_kmnk(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sparse_dense_kmnk(outputs)

reg.register_pattern("nn.sparse_dense_kmnk", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sparse_dense_mknk
@reg.register_compute("nn.sparse_dense_mknk")
def compute_sparse_dense_mknk(attrs, inputs, out_type, target):
    """Compute definition of sparse_dense_mknk"""
    return [topi.nn.sparse_dense_mknk(inputs[0], inputs[1], inputs[2],
                                           inputs[3])]

@reg.register_schedule("nn.sparse_dense_mknk")
def schedule_sparse_dense_mknk(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sparse_dense_mknk(outputs)

reg.register_pattern("nn.sparse_dense_mknk", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sdense
@reg.register_compute("nn.sdense")
def compute_sdense(attrs, inputs, out_type, target):
    """Compute definition of sdense"""
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    return [topi.nn.sdense(inputs[0], inputs[1], inputs[2], inputs[3],
                           data_layout=layout, kernel_layout=kernel_layout)]

@reg.register_schedule("nn.sdense")
def schedule_sdense(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sdense(outputs)

reg.register_pattern("nn.sdense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# grucell
@reg.register_compute("nn.grucell")
def compute_grucell(attrs, inputs, out_type, target):
    """Compute definition of grucell"""
    return [topi.nn.grucell(inputs[0], inputs[1], inputs[2], inputs[3],
                            inputs[4], inputs[5], inputs[6], inputs[7], inputs[8])]

@reg.register_schedule("nn.grucell")
def schedule_grucell(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_grucell(outputs)

reg.register_pattern("nn.grucell", reg.OpPattern.OUT_ELEMWISE_FUSABLE)

# sgrucell
@reg.register_compute("nn.sgrucell")
def compute_sgrucell(attrs, inputs, out_type, target):
    """Compute definition of sgrucell"""
    return [topi.nn.sgrucell(inputs[0], inputs[1], inputs[2], inputs[3],
                             inputs[4], inputs[5], inputs[6], inputs[7],
                             inputs[8], inputs[9], inputs[10], inputs[11],
                             inputs[12], inputs[13], inputs[14], inputs[15],
                             inputs[16])]

@reg.register_schedule("nn.sgrucell")
def schedule_sgrucell(attrs, outputs, target):
    """Schedule definition of batch_matmul"""
    with target:
        return topi.generic.schedule_sgrucell(outputs)

reg.register_pattern("nn.sgrucell", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_dense
@reg.register_compute("nn.gru_gates")
def compute_gru_gates(attrs, inputs, out_type, target):
    """Compute definition of gru_gates"""
    return [topi.nn.gru_gates(inputs[0], inputs[1])]

@reg.register_schedule("nn.gru_gates")
def schedule_gru_gates(attrs, outputs, target):
    with target:
        return topi.generic.schedule_gru_gates(outputs)

reg.register_pattern("nn.gru_gates", reg.OpPattern.ELEMWISE)

# conv2d
@reg.register_compute("nn.conv2d")
def compute_conv2d(attrs, inputs, out_type, target):
    """Compute definition of conv2d"""
    padding = get_const_tuple(attrs.padding)
    strides = get_const_tuple(attrs.strides)
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    out_dtype = attrs.out_dtype
    out_dtype = (inputs[0].dtype if out_dtype in ("same", "")
                 else out_dtype)

    assert layout in ["NCHW", "NHWC", "NCHW4c"]
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        out = topi.nn.conv2d(
            inputs[0], inputs[1], strides, padding,
            dilation, layout, out_dtype=out_dtype)
    elif layout == "NCHW" and \
         get_const_int(inputs[1].shape[0]) == groups and \
         get_const_int(inputs[1].shape[1]) == 1:
        out = topi.nn.depthwise_conv2d_nchw(
            inputs[0], inputs[1], strides, padding, dilation, out_dtype=out_dtype)
    elif layout == "NHWC" and \
         kernel_layout == "HWOI" and\
         get_const_int(inputs[1].shape[2]) == groups and \
         get_const_int(inputs[1].shape[3]) == 1:
        out = topi.nn.depthwise_conv2d_nhwc(
            inputs[0], inputs[1], strides, padding, dilation, out_dtype=out_dtype)
    elif layout in ['NCHW', 'NCHW4c']:
        out = topi.nn.group_conv2d_nchw(inputs[0], inputs[1], strides, padding, dilation, groups,
                                        out_dtype=out_dtype)
    else:
        raise ValueError("not support arbitrary group number for now")
    return [out]


@reg.register_schedule("nn.conv2d")
def schedule_conv2d(attrs, outs, target):
    """Schedule definition of conv2d"""
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    with target:
        if groups == 1 and layout == "NCHW":
            return topi.generic.schedule_conv2d_nchw(outs)
        if groups == 1 and layout == "NCHW4c":
            return topi.generic.schedule_conv2d_nchw(outs)
        if groups == 1 and layout == "NHWC":
            return topi.generic.schedule_conv2d_nhwc(outs)
        if groups != 1:
            if layout == "NCHW":
                # TODO(leyuan, merrymercy, Huyuwei): fold depthwise topi into conv2d.
                return topi.generic.schedule_depthwise_conv2d_nchw(outs)
            if layout == "NHWC" and kernel_layout == "HWOI":
                return topi.generic.schedule_depthwise_conv2d_nhwc(outs)
            if layout == "NCHW4c":
                return topi.generic.schedule_group_conv2d_nchw(outs)
    raise ValueError("No compatible schedule")


@reg.register_alter_op_layout("nn.conv2d")
def alter_op_layout_conv2d(attrs, inputs, tinfos):
    """Alternate the layout of conv2d"""
    from ... import op
    return topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, op)

reg.register_pattern("nn.conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# conv2d_transpose
@reg.register_compute("nn.conv2d_transpose")
def compute_conv2d_transpose(attrs, inputs, out_dtype, target):
    """Compute definition of conv2d_transpose"""
    padding = get_const_tuple(attrs.padding)
    strides = get_const_tuple(attrs.strides)
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    out_dtype = attrs.out_dtype
    out_dtype = (inputs[0].dtype if out_dtype in ("same", "")
                 else out_dtype)
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    out = topi.nn.conv2d_transpose_nchw(inputs[0], inputs[1], strides, padding, out_dtype)
    output_padding = get_const_tuple(attrs.output_padding)
    out = topi.nn.pad(out,
                      [0, 0, 0, 0], [0, 0, output_padding[0], output_padding[1]])
    return [out]

@reg.register_schedule("nn.conv2d_transpose")
def schedule_conv2d_transpose(attrs, outs, target):
    """Schedule definition of conv2d_transpose"""
    with target:
        return topi.generic.schedule_conv2d_transpose_nchw(outs)

reg.register_pattern("nn.conv2d_transpose", OpPattern.OUT_ELEMWISE_FUSABLE)

# bias_add
@reg.register_compute("nn.bias_add")
def compute_bias_add(attrs, inputs, out_dtype, target):
    """Compute definition of conv2d_transpose"""
    axis = attrs.axis
    bias = inputs[1]
    data_ndim = len(inputs[0].shape)
    if axis < 0:
        axis = axis + data_ndim
    num_newaxis = data_ndim - axis - 1

    if num_newaxis:
        bias = topi.expand_dims(bias, axis=1, num_newaxis=num_newaxis)
    return [topi.add(inputs[0], bias)]

reg.register_schedule("nn.bias_add", schedule_injective)
reg.register_pattern("nn.bias_add", OpPattern.BROADCAST)


# max_pool2d
@reg.register_schedule("nn.max_pool2d")
def schedule_max_pool2d(attrs, outs, target):
    """Schedule definition of max_pool2d"""
    layout = attrs.layout
    with target:
        return topi.generic.schedule_pool(outs, layout)

reg.register_pattern("nn.max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d
@reg.register_schedule("nn.avg_pool2d")
def schedule_avg_pool2d(attrs, outs, target):
    """Schedule definition of avg_pool2d"""
    layout = attrs.layout
    with target:
        return topi.generic.schedule_pool(outs, layout)

reg.register_pattern("nn.avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_max_pool2d
@reg.register_schedule("nn.global_max_pool2d")
def schedule_global_max_pool2d(_, outs, target):
    """Schedule definition of global_max_pool2d"""
    with target:
        return topi.generic.schedule_global_pool(outs)

reg.register_pattern("nn.global_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_avg_pool2d
@reg.register_schedule("nn.global_avg_pool2d")
def schedule_global_avg_pool2d(_, outs, target):
    """Schedule definition of global_avg_pool2d"""
    with target:
        return topi.generic.schedule_global_pool(outs)

reg.register_pattern("nn.global_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)

# leaky_relu
reg.register_schedule("nn.leaky_relu", schedule_broadcast)
reg.register_pattern("nn.leaky_relu", OpPattern.ELEMWISE)

# prelu
reg.register_schedule("nn.prelu", schedule_broadcast)
reg.register_pattern("nn.prelu", OpPattern.BROADCAST)

# flatten
reg.register_schedule("nn.batch_flatten", schedule_broadcast)
reg.register_pattern("nn.batch_flatten", OpPattern.INJECTIVE)


# lrn
@reg.register_compute("nn.lrn")
def compute_lrn(attrs, inputs, out_dtype, target):
    """Compute definition of lrn"""
    assert len(inputs) == 1
    return [topi.nn.lrn(inputs[0], attrs.size, attrs.axis,
                        attrs.alpha, attrs.beta, attrs.bias)]

@reg.register_schedule("nn.lrn")
def schedule_lrn(attrs, outs, target):
    """Schedule definition of lrn"""
    with target:
        return topi.generic.schedule_lrn(outs)

reg.register_pattern("nn.lrn", OpPattern.OPAQUE)


# l2_normalize
@reg.register_compute("nn.l2_normalize")
def compute_l2_normalize(attrs, inputs, out_dtype, target):
    """Compute definition of l2 normalize"""
    return [topi.nn.l2_normalize(inputs[0], attrs.eps, attrs.axis)]

@reg.register_schedule("nn.l2_normalize")
def schedule_l2_normalize(attrs, outs, target):
    """Schedule definition of l2 normalize"""
    with target:
        return topi.generic.schedule_l2_normalize(outs)

reg.register_pattern("nn.l2_normalize", OpPattern.OUT_ELEMWISE_FUSABLE)

# upsampling
reg.register_schedule("nn.upsampling", reg.schedule_injective)
def schedule_upsampling(_, outs, target):
    """Schedule definition of upsampling"""
    with target:
        return topi.generic.schedule_injective(outs)
# pad
reg.register_schedule("nn.pad", schedule_broadcast)

# winograd related operators
@reg.register_compute("nn.contrib_conv2d_winograd_without_weight_transform")
def compute_contrib_conv2d_winograd_without_weight_transform(attrs, inputs, out_dtype, target):
    """Compute definition of conv2d_winograd_without_weight_transform"""
    # pylint: disable=assignment-from-no-return
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    data_layout = attrs.get_str("data_layout")
    out_dtype = attrs.get_str("out_dtype")
    tile_size = attrs.get_int("tile_size")
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not supoort arbitrary group number"

    out = topi.nn.conv2d_winograd_without_weight_transform(
        inputs[0], inputs[1], strides, padding, dilation, data_layout,
        out_dtype, tile_size)

    return [out]

@reg.register_schedule("nn.contrib_conv2d_winograd_without_weight_transform")
def schedule_contrib_conv2d_winograd_without_weight_transform(attrs, outs, target):
    """Schedule definition of conv2d_winograd_without_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_without_weight_transform(outs)

reg.register_pattern("nn.contrib_conv2d_winograd_without_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("nn.contrib_conv2d_winograd_weight_transform")
def compute_contrib_conv2d_winograd_weight_transform(attrs, inputs, out_dtype, target):
    """Compute definition of contrib_conv2d_winograd_weight_transform"""
    out = topi.nn.conv2d_winograd_weight_transform(inputs[0], attrs.get_int('tile_size'))
    return [out]

@reg.register_schedule("nn.contrib_conv2d_winograd_weight_transform")
def schedule_contrib_conv2d_winograd_weight_transform(attrs, outs, target):
    """Schedule definition of contrib_conv2d_winograd_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_weight_transform(outs)

reg.register_pattern("nn.contrib_conv2d_winograd_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)


# winograd nnpack related operators
@reg.register_compute("nn.contrib_conv2d_winograd_nnpack_without_weight_transform")
def compute_contrib_conv2d_winograd_nnpack_without_weight_transform(
        attrs, inputs, out_dtype, target):
    """Compute definition of conv2d_winograd_nnpack_without_weight_transform"""
    # pylint: disable=assignment-from-no-return
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    data_layout = attrs.get_str("data_layout")
    out_dtype = attrs.get_str("out_dtype")
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not supoort arbitrary group number"

    # No bias
    out = topi.nn.conv2d_winograd_nnpack_without_weight_transform(
        inputs[0], inputs[1], None, strides, padding, dilation, data_layout,
        out_dtype)

    return [out]

@reg.register_schedule("nn.contrib_conv2d_winograd_nnpack_without_weight_transform")
def schedule_contrib_conv2d_winograd_nnpack_without_weight_transform(attrs, outs, target):
    """Schedule definition of conv2d_winograd_nnpack_without_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_nnpack_without_weight_transform(outs)

reg.register_pattern("nn.contrib_conv2d_winograd_nnpack_without_weight_transform",
                     OpPattern.OPAQUE)


@reg.register_compute("nn.contrib_conv2d_winograd_nnpack_weight_transform")
def compute_contrib_conv2d_winograd_nnpack_weight_transform(attrs, inputs, out_dtype, target):
    """Compute definition of contrib_conv2d_winograd_nnpack_weight_transform"""
    convolution_algorithm = attrs.get_int('convolution_algorithm')
    out = topi.nn.conv2d_winograd_nnpack_weight_transform(
        inputs[0], convolution_algorithm, out_dtype)
    return [out]

@reg.register_schedule("nn.contrib_conv2d_winograd_nnpack_weight_transform")
def schedule_contrib_conv2d_winograd_nnpack_weight_transform(attrs, outs, target):
    """Schedule definition of contrib_conv2d_winograd_nnpack_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_nnpack_weight_transform(outs)

reg.register_pattern("nn.contrib_conv2d_winograd_nnpack_weight_transform",
                     OpPattern.OPAQUE)


@reg.register_compute("nn.contrib_conv2d_NCHWc")
def compute_contrib_conv2d_NCHWc(attrs, inputs, out_dtype, target):
    """Compute definition of conv2d NCHWc"""
    # pylint: disable=assignment-from-no-return
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs.get_str("data_layout")
    out_layout = attrs.get_str("out_layout")
    out_dtype = attrs.get_str("out_dtype")
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype

    out = topi.nn.conv2d_NCHWc(inputs[0], inputs[1], strides, padding, dilation,
                               data_layout, out_layout, out_dtype)
    return [out]

@reg.register_schedule("nn.contrib_conv2d_NCHWc")
def schedule_contrib_conv2d_NCHWc(attrs, outs, target):
    """Schedule definition of contrib_conv2d_NCHWc"""
    with target:
        return topi.generic.schedule_conv2d_NCHWc(outs)

reg.register_pattern("nn.contrib_conv2d_NCHWc",
                     OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("nn.contrib_depthwise_conv2d_NCHWc")
def compute_contrib_depthwise_conv2d_NCHWc(attrs, inputs, out_dtype, target):
    """Compute definition of depthwise conv2d NCHWc"""
    # pylint: disable=assignment-from-no-return
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs.get_str("data_layout")
    out_layout = attrs.get_str("out_layout")
    out_dtype = attrs.get_str("out_dtype")
    out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype

    out = topi.nn.depthwise_conv2d_NCHWc(inputs[0], inputs[1], strides, padding, dilation,
                                         data_layout, out_layout, out_dtype)
    return [out]

@reg.register_schedule("nn.contrib_depthwise_conv2d_NCHWc")
def schedule_contrib_depthwise_conv2d_NCHWc(attrs, outs, target):
    """Schedule definition of contrib_conv2d_NCHWc"""
    with target:
        return topi.generic.schedule_depthwise_conv2d_NCHWc(outs)

reg.register_pattern("nn.contrib_depthwise_conv2d_NCHWc",
                     OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("nn.deformable_conv2d")
def compute_deformable_conv2d(attrs, inputs, out_dtype, target):
    """Compute definition of deformable_conv2d"""
    padding = get_const_tuple(attrs.padding)
    strides = get_const_tuple(attrs.strides)
    dilation = get_const_tuple(attrs.dilation)
    deformable_groups = attrs.deformable_groups
    groups = attrs.groups
    out_dtype = attrs.out_dtype
    out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
    with target:
        out = topi.nn.deformable_conv2d_nchw(inputs[0], inputs[1], inputs[2], strides, padding,
                                             dilation, deformable_groups, groups, out_dtype)
    return [out]

@reg.register_schedule("nn.deformable_conv2d")
def schedule_deformable_conv2d(attrs, outs, target):
    """Schedule definition of deformable_conv2d"""
    with target:
        return topi.generic.schedule_deformable_conv2d_nchw(outs)

reg.register_pattern("nn.deformable_conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)
