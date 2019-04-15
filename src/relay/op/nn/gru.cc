/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.nn.dense
TVM_REGISTER_NODE_TYPE(GRUGatesAttrs);

bool GRUGatesRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* input_transform = types[0].as<TensorTypeNode>();
  Array<IndexExpr> oshape({input_transform->shape[0], input_transform->shape[1] / 3});
  reporter->Assign(types[2], TensorTypeNode::make(oshape, input_transform->dtype));
  return false;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeGRUGates(Expr input_transform, Expr hidden_transform) {
  auto attrs = make_node<GRUGatesAttrs>();
  static const Op& op = Op::Get("nn.gru_gates");
  return CallNode::make(op, {input_transform, hidden_transform}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.gru_gates")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 2>(MakeGRUGates, args, rv);
    });

RELAY_REGISTER_OP("nn.gru_gates")
    .describe(R"code()code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.GRUGatesAttrs")
    .set_num_inputs(2)
    .add_argument("input_transform", "2D Tensor", "")
    .add_argument("hidden_transform", "2D Tensor", "")
    .set_support_level(1)
    .add_type_rel("GRUGates", GRUGatesRel);



TVM_REGISTER_NODE_TYPE(GRUCellAttrs);

bool GRUCellRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 10);
  const auto* input = types[0].as<TensorTypeNode>();
  //const auto* h = types[1].as<TensorTypeNode>();
  //const auto* w_xz = types[2].as<TensorTypeNode>();
  const auto* tb_hn = types[8].as<TensorTypeNode>();
  if (input == nullptr) return false;

  Array<IndexExpr> oshape({input->shape[0], tb_hn->shape[0]});
  reporter->Assign(types[9], TensorTypeNode::make(oshape, input->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeGRUCell(Expr input, Expr tw_x, Expr tb_x, Expr tw_z, Expr tb_z, Expr tw_in,
                 Expr tb_in, Expr tw_hn, Expr tb_hn) {
  auto attrs = make_node<GRUCellAttrs>();
  static const Op& op = Op::Get("nn.grucell");
  return CallNode::make(op, {input, tw_x, tb_x, tw_z, tb_z, tw_in, tb_in, tw_hn, tb_hn}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.grucell")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 9>(MakeGRUCell, args, rv);
    });

RELAY_REGISTER_OP("nn.grucell")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **x**: `(x1, x2, ..., xn, input_dim)`
- **h**: `(h1, h2, ..., hn, input_dim)`
- **w_xz**: `(units, input_dim)`
- **w_n**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.GRUCellAttrs")
    .set_num_inputs(9)
    .add_argument("input", "nD Tensor", "Input data.")
    .add_argument("tw_x", "nD Tensor", "Hidden data.")
    .add_argument("tb_x", "nD Tensor", "Weight matrix for x and z.")
    .add_argument("tw_z", "nD Tensor", "Hidden data.")
    .add_argument("tb_z", "nD Tensor", "Weight matrix for x and z.")
    .add_argument("tw_in", "nD Tensor", "Weight matrix for n,")
    .add_argument("tb_in", "nD Tensor", "Hidden data.")
    .add_argument("tw_hn", "nD Tensor", "Weight matrix for x and z.")
    .add_argument("tb_hn", "nD Tensor", "Weight matrix for n,")
    .set_support_level(1)
    .add_type_rel("GRUCell", GRUCellRel);


TVM_REGISTER_NODE_TYPE(SGRUCellAttrs);

bool SGRUCellRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 18);
  const auto* x = types[0].as<TensorTypeNode>();
  //const auto* h = types[1].as<TensorTypeNode>();
  //const auto* w_xz_data = types[2].as<TensorTypeNode>();
  //const auto* w_xz_indices = types[3].as<TensorTypeNode>();
  //const auto* w_xz_indptr = types[4].as<TensorTypeNode>();
  const auto* w_hn_data = types[13].as<TensorTypeNode>();
  //const auto* w_n_indices = types[6].as<TensorTypeNode>();
  const auto* w_hn_indptr = types[15].as<TensorTypeNode>();
  if (x == nullptr) return false;

  Array<IndexExpr> oshape({x->shape[0], (w_hn_indptr->shape[0]-1) * w_hn_data->shape[1]});
  reporter->Assign(types[17], TensorTypeNode::make(oshape, x->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSGRUCell(Expr input, Expr w_x_data, Expr w_x_indices, Expr w_x_indptr, Expr b_x,
                  Expr w_z_data, Expr w_z_indices, Expr w_z_indptr, Expr b_z,
                  Expr w_in_data, Expr w_in_indices, Expr w_in_indptr, Expr b_in,
                  Expr w_hn_data, Expr w_hn_indices, Expr w_hn_indptr, Expr b_hn) {
  auto attrs = make_node<SGRUCellAttrs>();
  static const Op& op = Op::Get("nn.sgrucell");
  return CallNode::make(op, {input, w_x_data, w_x_indices, w_x_indptr, b_x, \
        w_z_data, w_z_indices, w_z_indptr, b_z, \
        w_in_data, w_in_indices, w_in_indptr, b_in, \
        w_hn_data, w_hn_indices, w_hn_indptr, b_hn}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.sgrucell")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 17>(MakeSGRUCell, args, rv);
    });

RELAY_REGISTER_OP("nn.sgrucell")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **x**: `(x1, x2, ..., xn, input_dim)`
- **h**: `(h1, h2, ..., hn, input_dim)`
- **w_xz**: `(units, input_dim)`
- **w_n**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.SGRUCellAttrs")
    .set_num_inputs(17)
    .add_argument("x", "nD Tensor", "Input data.")
    .add_argument("h", "nD Tensor", "Hidden data.")
    .add_argument("w_x_data", "1D Tensor", "Weight data matrix.")
    .add_argument("w_x_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("w_x_indptr", "1D Tensor", "Weight indptr matrix.")
    .add_argument("b_n", "1D Tensor", "Weight data matrix.")
    .add_argument("w_z_data", "1D Tensor", "Weight data matrix.")
    .add_argument("w_z_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("w_z_indptr", "1D Tensor", "Weight indptr matrix.")
    .add_argument("b_z", "1D Tensor", "Weight data matrix.")
    .add_argument("w_in_data", "1D Tensor", "Weight data matrix.")
    .add_argument("w_in_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("w_in_indptr", "1D Tensor", "Weight indptr matrix.")
    .add_argument("b_in", "1D Tensor", "Weight data matrix.")
    .add_argument("w_hn_data", "1D Tensor", "Weight data matrix.")
    .add_argument("w_hn_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("w_hn_indptr", "1D Tensor", "Weight indptr matrix.")
    .add_argument("b_hn", "1D Tensor", "Weight data matrix.")
    .set_support_level(1)
    .add_type_rel("SGRUCell", SGRUCellRel);

}  // namespace relay
}  // namespace tvm
