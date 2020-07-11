# Copyright 2020 The TensorFlow Runtime Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
r"""Generate performance test cases for BEFExecutor.

Usage:
  python gen_benchmark_mlir.py <test_case> <optional:num_kernels>

  where each 'test_case' is a key in main()'s generator_map.

  Example command:
  $ python3 gen_benchmark_mlir.py fully_serial --num_kernels 120 > \
      fully_serial.mlir
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlir_tests.bef_perf.gen_benchmark_mlir_lib import gen_benchmark_mlir_main  # from @tf_runtime
from mlir_tests.bef_perf.gen_benchmark_mlir_lib import generate_benchmark_mlir  # from @tf_runtime


def generate_fully_serial_mlir(num_kernels):
  """Generate a fully serial DAG for benchmarking BEFExecutor."""

  body = """
  // The pseudo-code for this mlir function is as follows:
  //
  // a = 1
  // c0 = 1
  // c1 = c0 + a
  // c2 = c1 + a
  // c3 = c2 + a
  // ...
  // Since each c_i depends on c_{i-1}, all c_i's need to be computed in serial.

  %a = tfrt.constant.i32 1
  %c0 = tfrt.constant.i32 1
"""

  def gen_line(c):
    return '  %c{} = "tfrt.add.i32"(%c{}, %a) : (i32, i32) -> i32'.format(
        c + 1, c)

  body += '\n'.join([gen_line(c) for c in range(num_kernels)])
  # Add return statement.
  body += '\n  tfrt.return %c{} : i32'.format(num_kernels)

  return generate_benchmark_mlir('BM_full_serial_{}'.format(num_kernels), body)


def generate_fully_parallel_mlir(num_kernels):
  """Generate a fully parallel DAG for benchmarking BEFExecutor."""

  body = """
  // The pseudo-code for this mlir function is as follows:
  //
  // a = 1
  // c0 = 1
  // c1 = c0 + a
  // c2 = c0 + a
  // c3 = c0 + a
  // ...
  //
  // Since c_i's have no dependency with each other, they can be computed in
  // parallel.

  %a = tfrt.constant.i32 1
  %c0 = tfrt.constant.i32 1
"""

  def gen_line(c):
    return '  %c{} = "tfrt_test.async_add.i32"(%c0, %a) : (i32, i32) -> i32'.format(
        c + 1)

  body += '\n'.join([gen_line(c) for c in range(num_kernels)])
  # Add return statement.
  body += '\n  tfrt.return %c{} : i32'.format(num_kernels)

  return generate_benchmark_mlir('BM_full_parallel_{}'.format(num_kernels),
                                 body)


def generate_star_mlir(num_kernels):
  """Generate a fully parallel DAG for benchmarking BEFExecutor."""

  body = """
  // The pseudo-code for this mlir function is as follows:
  //
  // a = 1
  // c0 = 1
  // c1 = c0 + a
  // c2 = c0 + a
  // c3 = c0 + a
  // ...
  // s = sum(c0, c1, c2 ...)
  //
  // Since c_i's have no dependency with each other, they can be computed in
  // parallel. s depends on all of c_i's, thus can only be computed after
  // the computation for all c_i's is done.

  %a = tfrt.constant.i32 1
  %c0 = tfrt.constant.i32 1
"""

  def gen_line(c):
    return '  %c{} = "tfrt_test.async_add.i32"(%c0, %a) : (i32, i32) -> i32'.format(
        c + 1)

  # Construct sum statement:
  # %s = "tfrt_test.sum100"(%c1, %c2, ...) : (i32, ..., i32) -> i32
  sum_line = '  %s = "tfrt_test.sum"({args}) : ({arg_types}) -> i32'.format(
      args=', '.join(['%c{}'.format(i + 1) for i in range(num_kernels)]),
      arg_types=', '.join(['i32' for i in range(num_kernels)]),
  )

  body += '\n'.join([gen_line(c) for c in range(num_kernels)])
  body += ('\n' + sum_line)
  # Add return statement.
  body += '\n  tfrt.return %s : i32'

  return generate_benchmark_mlir('BM_star_{}'.format(num_kernels), body)


def generate_dense_host_tensor(num_kernels):
  """Benchmark DHTIndexableView overhead.

  Generate a no-op dense host tensor program for benchmarking the overhead of
  DHTIndexableView.
  """

  body = """
  // The pseudo-code for this mlir function is as follows:
  //
  // t = dense_host_tensor
  // c0 = tfrt.chain
  // c1 = dht.no_op_dht.i32.2(t, c0)
  // c2 = dht.no_op_dht.i32.2(t, c1)
  // c3 = dht.no_op_dht.i32.2(t, c2)
  // ...
  // return cn

  %t = dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
  %c0 = tfrt.new.chain
"""

  def gen_line(c):
    return (
        '  %c{} = "dht.no_op_dht.i32.2"(%t, %c{}) : '
        '(!dht.dense_host_tensor.i32.2, !tfrt.chain) -> !tfrt.chain').format(
            c + 1, c)

  body += '\n'.join([gen_line(c) for c in range(num_kernels)])
  # Add return statement.
  body += '\n  tfrt.return %c{} : !tfrt.chain'.format(num_kernels)

  return generate_benchmark_mlir(
      'BM_DenseHostTensor_{}'.format(num_kernels),
      body) + '\n' + generate_host_tensor(num_kernels)


def generate_host_tensor(num_kernels):
  """Benchmark DHTIndexableView overhead.

  Generate a no-op host tensor program for benchmarking the overhead of
  DHTIndexableView.
  """

  body = """
  // The pseudo-code for this mlir function is as follows:
  //
  // t = dense_host_tensor
  // c0 = tfrt.chain
  // c1 = dht.no_op_ht(t, c0)
  // c2 = dht.no_op_ht(t, c1)
  // c3 = dht.no_op_ht(t, c2)
  // ...
  // return cn

  %t = dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
  %c0 = tfrt.new.chain
"""

  def gen_line(c):
    return (
        '  %c{} = "dht.no_op_ht"(%t, %c{}) : '
        '(!dht.dense_host_tensor.i32.2, !tfrt.chain) -> !tfrt.chain').format(
            c + 1, c)

  body += '\n'.join([gen_line(c) for c in range(num_kernels)])
  # Add return statement.
  body += '\n  tfrt.return %c{num_kernels} : !tfrt.chain'

  return generate_benchmark_mlir('BM_HostTensor_{}'.format(num_kernels), body)


def main():
  generator_map = {
      'fully_serial': generate_fully_serial_mlir,
      'fully_parallel': generate_fully_parallel_mlir,
      'star': generate_star_mlir,
      'dense_host_tensor': generate_dense_host_tensor,
  }
  gen_benchmark_mlir_main(generator_map)


if __name__ == '__main__':
  main()
