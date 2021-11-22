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
"""Lit configuration."""

from __future__ import absolute_import

import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

cwd = os.getcwd()

# pylint: disable=undefined-variable

# name: The name of this test suite.
config.name = 'TFRT'

# test_format: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: Base path of the test files. The remainder (none in this
# case) is the lit argument stripped by the path to the suite directory.
config.test_source_root = os.path.dirname(os.environ['TEST_BINARY'])

# test_exec_root: Base path to the execution directory. The remainder is the lit
# argument directory stripped by the path to the suite directory.
config.test_exec_root = '.'

config.llvm_tools_dir = os.path.join(cwd, '..', 'llvm-project', 'llvm')

# pylint: enable=undefined-variable

llvm_config.use_default_substitutions()


def _AddToolSubstitutions(targets):
  paths = [
      t.lstrip('/').replace('@', '../').replace('//', '/').replace(':', '/')
      for t in targets
  ]
  llvm_config.add_tool_substitutions([
      ToolSubst(os.path.basename(p), os.path.join(cwd, p), unresolved='ignore')
      for p in paths
  ], [])


_AddToolSubstitutions([
    '//backends/gpu:tfrt_gpu_executor',
    '//backends/gpu:tfrt_gpu_translate',
    '//backends/gpu:tfrt_gpu_opt',
    '//tools:bef_executor',
    '//tools:bef_executor_debug_tracing',
    '//tools:bef_executor_lite',
    '//tools:code_size_test_driver',
    '//tools:tfrt_opt',
    '//tools:tfrt_translate',
])
