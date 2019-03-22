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
TVM_REGISTER_NODE_TYPE(SparseDenseAttrs);

bool SparseDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  // const auto* weight_data = types[1].as<TensorTypeNode>();
  // const auto* weight_indices = types[2].as<TensorTypeNode>();
  const auto* weight_indptr = types[3].as<TensorTypeNode>();
  if (data == nullptr) return false;

  Array<IndexExpr> oshape({data->shape[0], weight_indptr->shape[0] - 1});
  reporter->Assign(types[4], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSparseDense(Expr data, Expr weight_data, Expr weight_indices, Expr weight_indptr) {
  auto attrs = make_node<SparseDenseAttrs>();
  static const Op& op = Op::Get("nn.sparse_dense");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.sparse_dense")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 4>(MakeSparseDense, args, rv);
    });

RELAY_REGISTER_OP("nn.sparse_dense")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.SparseDenseAttrs")
    .set_num_inputs(4)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight_data", "1D Tensor", "Weight data matrix.")
    .add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
    .set_support_level(1)
    .add_type_rel("SparseDense", SparseDenseRel);



// relay.nn.dense
TVM_REGISTER_NODE_TYPE(SparseDense2Attrs);

bool SparseDense2Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  // const auto* weight_data = types[1].as<TensorTypeNode>();
  // const auto* weight_indices = types[2].as<TensorTypeNode>();
  const auto* weight_indptr = types[3].as<TensorTypeNode>();
  if (data == nullptr) return false;

  Array<IndexExpr> oshape({weight_indptr->shape[0] - 1, data->shape[1]});
  reporter->Assign(types[4], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSparseDense2(Expr data, Expr weight_data, Expr weight_indices, Expr weight_indptr) {
  auto attrs = make_node<SparseDense2Attrs>();
  static const Op& op = Op::Get("nn.sparse_dense2");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.sparse_dense2")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 4>(MakeSparseDense2, args, rv);
    });

RELAY_REGISTER_OP("nn.sparse_dense2")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.SparseDense2Attrs")
    .set_num_inputs(4)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight_data", "1D Tensor", "Weight data matrix.")
    .add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
    .set_support_level(1)
    .add_type_rel("SparseDense2", SparseDense2Rel);


TVM_REGISTER_NODE_TYPE(SparseDenseStructureAttrs);

bool SparseDenseStructureRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  // const auto* weight_data = types[1].as<TensorTypeNode>();
  // const auto* weight_indices = types[2].as<TensorTypeNode>();
  const auto* weight_indptr = types[3].as<TensorTypeNode>();
  if (data == nullptr) return false;

  Array<IndexExpr> oshape({weight_indptr->shape[0] - 1, data->shape[2]});
  reporter->Assign(types[4], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSparseDenseStructure(Expr data, Expr weight_data, Expr weight_indices,
                              Expr weight_indptr) {
  auto attrs = make_node<SparseDenseStructureAttrs>();
  static const Op& op = Op::Get("nn.sparse_dense_structure");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr,
                             }, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.sparse_dense_structure")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 4>(MakeSparseDenseStructure, args, rv);
    });

RELAY_REGISTER_OP("nn.sparse_dense_structure")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.SparseDenseStructureAttrs")
    .set_num_inputs(4)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight_data", "1D Tensor", "Weight data matrix.")
    .add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
    .set_support_level(1)
    .add_type_rel("SparseDenseStructure", SparseDenseStructureRel);



TVM_REGISTER_NODE_TYPE(SparseDenseStructure2Attrs);

bool SparseDenseStructure2Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* data = types[0].as<TensorTypeNode>();
  // const auto* weight_data = types[1].as<TensorTypeNode>();
  // const auto* weight_indices = types[2].as<TensorTypeNode>();
  const auto* weight_indptr = types[3].as<TensorTypeNode>();
  if (data == nullptr) return false;

  Array<IndexExpr> oshape({weight_indptr->shape[0] - 1, data->shape[2]});
  reporter->Assign(types[4], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeSparseDenseStructure2(Expr data, Expr weight_data, Expr weight_indices,
                              Expr weight_indptr) {
  auto attrs = make_node<SparseDenseStructure2Attrs>();
  static const Op& op = Op::Get("nn.sparse_dense_structure2");
  return CallNode::make(op, {data, weight_data, weight_indices, weight_indptr,
                             }, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.sparse_dense_structure2")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 4>(MakeSparseDenseStructure2, args, rv);
    });

RELAY_REGISTER_OP("nn.sparse_dense_structure2")
    .describe(R"code(Applies a sparse linear transformation: :math:`Y = XW^T` with X sparse.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.SparseDenseStructure2Attrs")
    .set_num_inputs(4)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight_data", "1D Tensor", "Weight data matrix.")
    .add_argument("weight_indices", "1D Tensor", "Weight indices matrix.")
    .add_argument("weight_indptr", "1D Tensor", "Weight indptr matrix.")
    .set_support_level(1)
    .add_type_rel("SparseDenseStructure2", SparseDenseStructure2Rel);

}  // namespace relay
}  // namespace tvm
