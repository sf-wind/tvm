/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "WaveRNNModel.h"

int main(void) {
  // Normally we can directly
  auto model = WaveRNNModel();
  model.evaluate();
  return 0;
}
