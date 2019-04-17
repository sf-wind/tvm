#include "WaveRNNModel.h"

#include <dlpack/dlpack.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <streambuf>
#include <string>
#include <vector>

typedef std::chrono::high_resolution_clock Clock;

namespace {

std::string readToString(const std::string& file) {
  std::ifstream t(file);
  std::string str(
      (std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  if (!t.is_open()) {
    throw std::runtime_error("Unable to read from " + file + "!");
  }
  return str;
}

void printRTF(
    std::string name,
    std::chrono::duration<long, std::ratio<1, 1000000000>> dur,
    int sample_count) {
  std::cout
      << ">>>> " << name << ":\t"
      << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
      << " ms.\t"
      << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() /
          sample_count
      << " us per sample.\t"
      << std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() /
          sample_count * 24 / (float)1000000
      << " RTF \n";
}

float sampleProba(const float* p, std::mt19937& seed) {
  std::uniform_real_distribution<float> unif(0, 1);
  const float rand = unif(seed);
  size_t rand_sample = 0;
  float prob_sum = p[0];
  while (prob_sum < rand) {
    prob_sum += p[++rand_sample];
  }
  return static_cast<float>(rand_sample);
}

} // namespace

size_t WaveRNNModel::evaluate() {
  init();
  auto t_start = Clock::now();
  t_start = Clock::now();

  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t I_residual_shape[2] = {8, rnn_dim_};
  DLTensor *I_residual;
  TVMArrayAlloc(I_residual_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &I_residual);

  int64_t fc1_residual_shape[2] = {8, rnn_dim_};
  DLTensor *fc1_residual;
  TVMArrayAlloc(fc1_residual_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &fc1_residual);

  auto dur_smp_net_input = Clock::now() - t_start;
  t_start = Clock::now();

  DLTensor *x;
  int64_t x_shape[2] = {1, 1};
  TVMArrayAlloc(x_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);
  DLTensor *h1;
  int64_t h1_shape[2] = {1, rnn_dim_};
  TVMArrayAlloc(h1_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &h1);

  auto seed = std::mt19937(0);
  std::vector<float> outs = tvmSample(
      I_residual,
      fc1_residual,
      x,
      h1,
      seed);

  auto dur_smp_net_time = Clock::now() - t_start;

  std::cout << "dur_smp_net_time: " << std::chrono::duration_cast<std::chrono::milliseconds>(dur_smp_net_time).count() << std::endl;

  return 0;
}

void WaveRNNModel::init() {

  std::string name = "hsw_fast_wavernn_rnn_dims_1024_fc_dims_1024_feat_dims_19_aux_dims_64";
  graph_json_ = name + "_graph.json";
  lib_dot_o_path_ = name + "_lib.o";
  lib_dot_so_path_ = name + "_lib.so";
  serialized_params_ = name + "_params.bin";

  loadTVMModel();

  rnn_dim_ = 1024;
  fc2_dim_ = rnn_dim_;
  feature_dim_ = 19;
  aux_dim_ = 64;
  bits_ = 256;

  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t x_proba_shape[2] = {1, bits_};

  TVMArrayAlloc(x_proba_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x_proba_);

  int64_t I_residual_t_shape[2] = {1, rnn_dim_};
  TVMArrayAlloc(I_residual_t_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &I_residual_t_);

  int64_t fc1_residual_t_shape[2] = {1, rnn_dim_};
  TVMArrayAlloc(fc1_residual_t_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &fc1_residual_t_);

  int64_t I_weights_shape[3] = {1, 1, 1 + feature_dim_ + aux_dim_};
  TVMArrayAlloc(I_weights_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &I_weights_);

  int64_t fc1_weights_A2_shape[2] = {1, rnn_dim_};
  TVMArrayAlloc(fc1_weights_A2_shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &fc1_weights_A2_);
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

void WaveRNNModel::loadTVMModel() {
  tvm::runtime::Module lib =
      tvm::runtime::Module::LoadFromFile(lib_dot_so_path_, "so");

  constexpr int device_type = kDLCPU;
  constexpr int device_id = 0;

  sample_module_ = make_unique<tvm::runtime::Module>(
      static_cast<const tvm::runtime::Module&>(
          (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(
              readToString(graph_json_), lib, device_type, device_id)));

  gr_ = static_cast<tvm::runtime::GraphRuntime*>(sample_module_->operator->());
  gr_->LoadParams(readToString(serialized_params_));
  gr_->GetInputIndex("x");
  std::cout << "tvmModel Loaded\n";
}

std::vector<float> WaveRNNModel::tvmSample(
    DLTensor* I_residual,
    DLTensor* fc1_residual,
    DLTensor* x,
    DLTensor* h1,
    std::mt19937& seed) const {
  const size_t T = I_residual->shape[0];
  std::vector<float> outs;

  std::chrono::duration<long, std::ratio<1, 1000000000>> dur =
      Clock::now() - Clock::now();
  std::chrono::duration<long, std::ratio<1, 1000000000>> dur_input =
      Clock::now() - Clock::now();

  std::chrono::duration<long, std::ratio<1, 1000000000>> dur_output =
      Clock::now() - Clock::now();

  int x_idx = gr_->GetInputIndex("x");
  int h1_idx = gr_->GetInputIndex("h1");
  int I_residual_idx = gr_->GetInputIndex("I_residual");
  int fc1_residual_idx = gr_->GetInputIndex("fc1_residual");

  for (size_t t = 0; t < T; ++t) {
    auto t_start1 = Clock::now();
    gr_->SetInput(x_idx, x);
    gr_->SetInput(h1_idx, h1);
    I_residual_t_->data =
        static_cast<float*>(I_residual->data) + t * I_residual->shape[1];
    gr_->SetInput(
        I_residual_idx, const_cast<DLTensor*>(I_residual_t_));
    fc1_residual_t_->data =
        static_cast<float*>(fc1_residual->data) + t * fc1_residual->shape[1];
    gr_->SetInput(
        fc1_residual_idx, const_cast<DLTensor*>(fc1_residual_t_));

    auto t_start = Clock::now();
    gr_->Run();
    dur += Clock::now() - t_start;
    // Compute and update new sampled x values
    gr_->CopyOutputTo(0, x_proba_);
    // Update h1
    gr_->CopyOutputTo(1, h1);

    const float x_t =
        sampleProba(static_cast<const float*>(x_proba_->data), seed);
    outs.push_back(x_t);
    static_cast<float*>(x->data)[0] = x_t / 256;
  }

  printRTF("dur_tvm_for_loop", dur, T);

  return outs;
}
