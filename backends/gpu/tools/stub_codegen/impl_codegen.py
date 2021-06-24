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
"""Generates dynamic loading stubs for functions in CUDA and HIP APIs."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import sys
import clang.cindex


def main():
  parser = argparse.ArgumentParser(
      description='Generate dynamic loading stubs for CUDA and HIP APIs.')
  parser.add_argument(
      'input', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument(
      'output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
  args = parser.parse_args()

  config = json.load(args.input)

  # See e.g. backends/gpu/lib/stream/cuda_stub.cc
  function_impl = """
    return DynamicCall<decltype({0}), &{0}>({1});
  """

  index = clang.cindex.Index.create()
  translation_unit = index.parse(config['header'], args=config['extra_args'])

  for cursor in translation_unit.cursor.get_children():
    if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
      continue

    if cursor.spelling not in config['functions']:
      continue

    with open(cursor.location.file.name, 'r', encoding='latin-1') as file:
      start = cursor.extent.start.offset
      end = cursor.extent.end.offset
      declaration = file.read()[start:end]

    arg_names = [arg.spelling for arg in cursor.get_arguments()]
    implementation = function_impl.format(
        cursor.spelling, ', '.join(['"%s"' % cursor.spelling] + arg_names))

    args.output.write('%s {%s\n}\n\n' % (declaration, implementation))


if __name__ == '__main__':
  main()
