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
"""An util library that runs the input->mlir_to_bef->bef_executor pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from cpuinfo import cpuinfo
from datetime import datetime
import re
import subprocess

STR_ENCODING = 'utf-8'


class Env:
  """Runtime environment."""

  def __init__(self, tfrt_translate, bef_executor, host_allocator_type,
               work_queue_type):
    """Initialize various runtime env parameters."""
    self.tfrt_translate = tfrt_translate
    self.bef_executor = bef_executor
    self.host_allocator_type = host_allocator_type
    self.work_queue_type = work_queue_type

  def run_mlir(self, in_str, additional_executor_flags=''):
    """Execute the given mlir.

    Run the given mlir string through mlir_to_bef and then through bef_executor.

    Args:
      in_str: an input string, which contains MLIR to be executed.
      additional_executor_flags: a string, which is the flag being passed to the
        bef_executor binary. Example '-shared_libs /path/to/shared/lib'

    Returns:
      A dict {function name : {metrics: value}} of parsed output
      from bef_executor. example:
      {"fully_serial_100" : {"Min(us)" : "252"}, {"50%(us)" : "259"}}
    """
    cmd = ('{} -mlir-to-bef | {} {} --host_allocator_type={} '
           '--work_queue_type={}').format(self.tfrt_translate,
                                          self.bef_executor,
                                          additional_executor_flags,
                                          self.host_allocator_type,
                                          self.work_queue_type)
    proc = subprocess.run(
        cmd,
        input=in_str.encode(STR_ENCODING),
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    return self._parse_bm_output(proc.stdout.decode(STR_ENCODING))

  @staticmethod
  def get_cpu_info():
    """Get CPU information, including num_cpus, mhz_per_cpu, etc.

    Returns:
      A dict of cpu info, including cpu_info (brand etc), num_cores,
      mhz_per_cpu, cache_size (dict of cache sizes L1, L2, ...)
    """
    info = cpuinfo.get_cpu_info()
    cpu_info = {}
    cpu_info['cpu_info'] = info['brand']
    cpu_info['num_cpus'] = info['count']
    # Assuming frequencies for all cores are the same.
    cpu_info['mhz_per_cpu'] = info['hz_advertised_raw'][0] / 1.0e6
    l1_cache_size = re.match(r'(\d+)', str(info.get('l1_cache_size', '')))
    l2_cache_size = re.match(r'(\d+)', str(info.get('l2_cache_size', '')))
    cpu_info['cache_size'] = {}
    if l1_cache_size:
      # If a value is returned, it's in KB.
      cpu_info['cache_size']['L1'] = int(l1_cache_size.group(0)) * 1024
    if l2_cache_size:
      # If a value is returned, it's in KB.
      cpu_info['cache_size']['L2'] = int(l2_cache_size.group(0)) * 1024

    return cpu_info

  @staticmethod
  def _parse_bm_output(output: str):
    """Parse the output of benchmark kernel.

    Args:
      output: Raw output from the benchmark kernel

    Returns:
      A dict from function name to performance result dict for the function.
      example: {"fully_serial_100" : {"Min(us)" : "252"}, {"50%(us)" : "259"}}
    """
    # Use OrderedDict to keep the metric ordering as output by the benchmark
    # kernel.
    results = collections.defaultdict(collections.OrderedDict)

    for line in output.split('\n'):
      # Benchmark results are prefixed with BM:. An example output line is:
      #   BM:BM_fully_serial_100:Min(us): 9
      if line.startswith('BM:'):
        parts = line.split(':')
        name = parts[1]
        metric = parts[2]
        value = parts[3]

        results[name][metric] = value

    return results
