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


def getSource(cursor):
  with open(cursor.location.file.name, 'r', encoding='latin-1') as file:
    start = cursor.extent.start.offset
    end = cursor.extent.end.offset
    return file.read()[start:end]


def main():
  parser = argparse.ArgumentParser(
      description='Generate dynamic loading stubs for CUDA and HIP APIs.')
  parser.add_argument(
      'input', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument(
      'output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
  args = parser.parse_args()

  config = json.load(args.input)

  index = clang.cindex.Index.create()
  translation_unit = index.parse(
      config['header'],
      args=config['extra_args'],
      options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

  def HandleFunction(cursor):
    if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
      return

    if cursor.spelling not in config['functions']:
      return

    args.output.write('%s;\n\n' % getSource(cursor))

  def HandleEnum(cursor, typedef=''):
    if cursor.kind != clang.cindex.CursorKind.ENUM_DECL:
      return

    enum = cursor.spelling
    if enum == typedef:
      return  # Pure enum has already been visited.

    if not any(x in config.get('enums', []) for x in [enum, typedef]):
      return

    if typedef:
      args.output.write('typedef ')
    args.output.write('enum %s {\n' % enum)

    for enumerator in cursor.get_children():
      args.output.write('  %s,\n' % getSource(enumerator))

    args.output.write('} %s;\n\n' % typedef)

  def HandleTypedef(cursor):
    if cursor.kind != clang.cindex.CursorKind.TYPEDEF_DECL:
      return

    HandleEnum(cursor.underlying_typedef_type.get_canonical().get_declaration(),
               cursor.spelling)

  for cursor in translation_unit.cursor.get_children():
    HandleFunction(cursor)
    HandleEnum(cursor)
    HandleTypedef(cursor)


if __name__ == '__main__':
  main()
