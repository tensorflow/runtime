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
"""Lit site configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys
import lit.llvm

runfile_srcdir = os.environ['TEST_SRCDIR']
tfrt_workspace = os.environ['TEST_WORKSPACE']

if platform.system() != 'Linux':
  sys.exit('Currently TFRT only supports lit tests on Linux.')

# pylint: disable=undefined-variable
config.runfile_srcdir = os.path.join(runfile_srcdir, tfrt_workspace)

config.llvm_tools_dir = os.path.join(runfile_srcdir, 'llvm-project', 'llvm')

config.tfrt_tools_dirs = [
    os.path.join(runfile_srcdir, tfrt_workspace, 'tools'),
    os.path.join(runfile_srcdir, tfrt_workspace, 'backends', 'gpu'),
]

test_target_dir = os.environ['TEST_TARGET'].strip('/').rsplit(':')[0]
config.tfrt_test_dir = os.path.join(runfile_srcdir, tfrt_workspace,
                                    test_target_dir)

lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(
    config, os.path.join(runfile_srcdir, tfrt_workspace,
                         'mlir_tests/lit.cfg.py'))
# pylint: enable=undefined-variable
