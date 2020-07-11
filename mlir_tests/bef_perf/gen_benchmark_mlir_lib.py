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
"""Infrastructure for generating BEFExecutor performance test cases.

See gen_benchmark_mlir.py for a usage example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def _indent_lines(s, indentation):
  """Indent lines in s with the given indentation."""
  lines = []
  for line in s.strip().split('\n'):
    stripped = line.strip()
    if stripped:
      lines.append(' ' * indentation + stripped)
    else:
      lines.append(stripped)

  return '\n'.join(lines)


def generate_benchmark_mlir(func_name, body):
  """Return the MLIR code for benchmarking func_name."""

  body = _indent_lines(body, indentation=4)

  return """
func @{func_name}() {{
  tfrt_test.benchmark "{func_name}"()
    duration_secs = 10,
    max_count = 1000000,
    num_warmup_runs = 10 {{
{body}
  }}
  tfrt.return
}}
""".format(
    func_name=func_name, body=body)


def gen_benchmark_mlir_main(generator_map):
  """Benchmark code generator's main() function.

  Args:
    generator_map: A dict mapping from test case names to functions that
      generate code for that test.

  Raises:
    RuntimeError: When the test case name is not known.
  """
  # pyformat: disable
  parser = argparse.ArgumentParser()
  parser.add_argument('tests', metavar='TEST', nargs='+',
                      help='The test case to generate.')
  parser.add_argument('--num_kernels', metavar='NUM_KERNELS', nargs='?',
                      type=int, default=100,
                      help='Number of kernels in a test. Default is 100.')
  args = parser.parse_args()

  header = """// This code is auto-generated from gen_benchmark_mlir.py. Do not edit manually.
// To generate this file run:
//   $ python3 ./gen_benchmark_mlir.py {test_names} --num_kernels {num_kernels}
""".format(test_names=' '.join(args.tests),
           num_kernels=args.num_kernels)
  # pyformat: enable

  print(header)

  for test in args.tests:
    if test not in generator_map:
      raise RuntimeError('Unknown test case {}'.format(test))

    generator = generator_map[test]
    print(generator(args.num_kernels))
