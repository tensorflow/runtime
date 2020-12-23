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

# Lint as: python3
r"""BEF Executor Performance benchmark driver.

This program runs the functions in the input MLIR code and prints the benchmark
results (lines prefixed with BM:) in a tabular format. The benchmark is assumed
to be performed with the tfrt_test.benchmark kernel (see
lib/test_kernels/benchmark_kernels.cc).

The following metrics are collected:

  Duration(us): The total benchmark duration for this function in microseconds
  Count: The total number of runs for this function
  Time Min(us): The minimum wall time for this function in microseconds
  Time 50%(us): The median (50%) wall time for this function in microseconds
  Time 95%(us): The 95 percentile wall time for this function in microseconds
  Time 99%(us): The 99 percentile wall time for this function in microseconds
  CPU Min(us): The minimum CPU time for this function in microseconds
  CPU 50%(us): The median (50%) CPU time for this function in microseconds
  CPU 95%(us): The 95 percentile CPU time for this function in microseconds
  CPU 99%(us): The 99 percentile CPU time for this function in microseconds

Usage:

  See the example commands below to build the bef_perf binary and pass it the
  names of generated .mlir files.

  # Example commands:
  $ bef_perf=mlir_tests/bef_perf
  $ bazel build $bef_perf/...
  $ bazel-bin/$bef_perf/bef_perf bazel-bin/$bef_perf/*.mlir
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from mlir_tests.bef_perf.benchmark_utils import Env  # from @tf_runtime


assert sys.version_info >= (3, 5), \
    'Detected Python version %s. Please use Python >= 3.5.' % sys.version

BUILD_DIR = 'bazel-bin'


def print_perf_results(results, cpu_info):
  """Format and print the benchmark results in tabular format.

   An example output is as follows (with some metrics omitted):

  CPU Info: Intel(R) Xeon(R) CPU @ 2.20GHz
  Num cores: 8
  Frequency: 2200.0 MHz
                   Duration(us)  Count    Time Min(us)   Time 50%(us)
  full_parallel_100     1000002    61319      14        15
  full_serial_100       1000004    96780      10        10
  star_100              1000017    46466      20        21

  Args:
    results: A dict from function names to the performance result dicts
    cpu_info: A dict of cpu info, including cpu_info (brand etc), num_cores,
      mhz_per_cpu, cache_size (dict of cache sizes L1, L2, ...)
  """
  if not results:
    return

  print('CPU Info: {}'.format(cpu_info['cpu_info']))
  print('Num cores: {}'.format(cpu_info['num_cpus']))
  print('Frequency: {} MHz'.format(cpu_info['mhz_per_cpu']))

  metrics = None
  row_format = None

  for name, res_dict in results.items():
    if not metrics:
      metrics = res_dict.keys()
      row_format = '{:<25}' + '{:^15}' * len(metrics)

      # print the header.
      print(row_format.format('', *metrics))

    results = [res_dict[m] for m in metrics]
    print(row_format.format(name, *results))


def run_benchmark(env: Env, in_file):
  """Run the benchmark functions contained in an MLIR file.

  Args:
    env: Runtime environment
    in_file: MLIR file containing functions for benchmarking

  Returns:
    A dict from function names to the performance result dicts
  """

  print('Running benchmarks in', in_file.name)
  # Run file_path through mlir_to_bef and bef_executor and extract the
  # benchmark result.
  return env.run_mlir(in_file.read())


def main():
  parser = argparse.ArgumentParser(
      description='Run benchmark functions in the input mlir files.')
  parser.add_argument(
      'mlirs',
      metavar='MLIRS',
      nargs='*',
      type=argparse.FileType('r'),
      default=[sys.stdin],
      help='MLIR code containing the function to be '
      'benchmarked. Default: STDIN.')
  parser.add_argument(
      '--host_allocator_type',
      default='malloc',
      help='Type of host allocator (malloc, ...)')
  parser.add_argument(
      '--work_queue_type',
      default='s',
      help='Type of work queue (s(default), mstd, ...)')
  parser.add_argument(
      '--tfrt_translate',
      default=os.path.join(BUILD_DIR, 'tools/tfrt_translate'),
      help='Path to tfrt_translate')
  parser.add_argument(
      '--bef_executor',
      default=os.path.join(BUILD_DIR, 'tools/bef_executor'),
      help='Path to bef_executor')

  args = parser.parse_args()

  env = Env(args.tfrt_translate, args.bef_executor, args.host_allocator_type,
            args.work_queue_type)

  # Run through each of the input mlir files.
  print('-' * 40)
  result_list = [run_benchmark(env, in_file) for in_file in args.mlirs]
  print('-' * 40)

  # Merge, format, and print the benchmark results.
  merged_results = dict()
  for r in result_list:
    merged_results.update(r)

  if not merged_results:
    print(
        'Empty result. Please check if MLIR file is correct.', file=sys.stderr)

  cpu_info = env.get_cpu_info()

  print_perf_results(merged_results, cpu_info)


if __name__ == '__main__':
  main()
