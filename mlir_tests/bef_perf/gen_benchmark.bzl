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

"""BUILD rules for generating benchmark .mlir files."""

# Runs gen_benchmark_mlir to create ${benchmark_name}.mlir.
def gen_benchmark(benchmark_name = "", num_kernels = 100):
    native.genrule(
        name = "gen_" + benchmark_name,
        outs = [benchmark_name + ".mlir"],
        cmd = "$(location gen_benchmark_mlir) --num_kernels=" + str(num_kernels) + " " + benchmark_name + " > $@",
        exec_tools = ["gen_benchmark_mlir"],
        output_to_bindir = True,  # Match OSS
    )
