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
from __future__ import division
from __future__ import print_function

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# pylint: disable=undefined-variable

# name: The name of this test suite.
config.name = 'TFRT'

# test_format: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = config.tfrt_test_dir

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.runfile_srcdir

llvm_config.use_default_substitutions()

llvm_config.config.substitutions.append(
    ('%tfrt_bindir', 'tensorflow/compiler/aot'))

tool_dirs = config.tfrt_tools_dirs + [config.llvm_tools_dir]

tool_names = [
    'bef_executor', 'bef_executor_lite', 'bef_name', 'tfrt_translate',
    'tfrt_opt', 'tfrt_gpu_translate', 'tfrt_gpu_opt', 'code_size_test_driver',
    'bef_executor_debug_tracing'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)
# pylint: enable=undefined-variable
