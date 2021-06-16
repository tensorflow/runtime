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
"""Tests the btf_info tool"""

import tempfile
import os.path
import subprocess
import unittest
import numpy as np

from utils import btf_writer  # from @tf_runtime

EXPECTED = ('[0] DenseHostTensor dtype = i8,'
            ' shape = [3, 5],'
            ' values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\n'
            '[1] CooHostTensor dtype = i64,'
            ' shape = [2, 4],'
            ' indices = [0, 1, 0, 2, 0, 3, 1, 1, 1, 2, 1, 3],'
            ' values = [1, 2, 3, 5, 6, 7]\n')


class BtfInfoTest(unittest.TestCase):

  def test_prints_btf_info(self):
    try:
      btf_fd, btf_path = tempfile.mkstemp('test-btf')

      with btf_writer.BTFWriter(open(btf_fd, 'wb')) as writer:
        tensor1 = np.arange(15, dtype=np.int8).reshape((3, 5))
        writer.add_tensors(tensor1, tensor_type=btf_writer.TensorLayout.RMD)

        tensor2 = np.arange(8, dtype=np.int64).reshape(2, 4)
        tensor2[1, 0] = 0
        writer.add_tensors(tensor2, tensor_type=btf_writer.TensorLayout.COO)

      result = subprocess.run(
          ['third_party/tf_runtime/tools/btf_info', btf_path],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE)

      self.assertEqual(result.stdout.decode('utf-8'), EXPECTED)
      self.assertEqual(result.stderr.decode('utf-8'), '')

    finally:
      os.remove(btf_path)


if __name__ == '__main__':
  unittest.main()
