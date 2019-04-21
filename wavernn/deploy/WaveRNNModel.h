#pragma once

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <random>
#include <src/runtime/graph/graph_runtime.h>

template <typename T>
using auto_del_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

class WaveRNNModel {
 public:
  static std::string name() {
    return "WaveRNNModel";
  }

  float evaluate();

  void init();


 private:
  void loadTVMModel();
  std::vector<float> tvmSample(
      DLTensor* I_residual,
      DLTensor* fc1_residual,
      DLTensor* x,
      DLTensor* h1,
      std::mt19937& seed) const;

  // tvm path
  std::string graph_json_;
  std::string lib_dot_o_path_;
  std::string lib_dot_so_path_;
  std::string serialized_params_;

  // tvm model attributes
  int rnn_dim_; // 1024
  int fc2_dim_; // 1024
  int feature_dim_; // 19
  int aux_dim_; // 64
  int bits_;

  // tvm model instance
  std::unique_ptr<tvm::runtime::Module> sample_module_;
  tvm::runtime::GraphRuntime* gr_;

  // Reusable dl tensor
  DLTensor* x_proba_;
  DLTensor* I_residual_t_;
  DLTensor* fc1_residual_t_;

  // Reusable tensor to construct residual
  DLTensor* I_weights_;
  DLTensor* fc1_weights_A2_;
};
