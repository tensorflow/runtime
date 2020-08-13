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

"""BUILD rules for translating .mlir to .bef and running TFRT .mlir tests."""

load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

def mlir_to_bef(name, tfrt_translate):
    """Runs "tfrt_translate -mlir-to-bef $test.mlir" to create $test.bef."""
    native.genrule(
        name = "mlir_to_bef." + name,
        srcs = [name],
        outs = [name[:-5] + ".bef"],
        cmd = "$(location " + tfrt_translate + ") -mlir-to-bef $(location " + name + ") > $@",
        exec_tools = [tfrt_translate],
    )

def glob_tfrt_lit_tests(
        name = "glob_tfrt_lit_tests",
        data = [],
        default_size = "small",
        default_tags = [],
        exclude = [],
        # Do not run "tfrt_translate -mlir-to-bef" on these files.
        no_bef_translation = [],
        size_override = {},
        tags_override = {},
        tfrt_translate = ""):
    """Run mlir_to_bef on all .mlir files and invoke glob_lit_tests."""

    if tfrt_translate == "":
        tfrt_translate = "@tf_runtime//tools:tfrt_translate"

    mlir_files = native.glob(
        include = ["**/*.mlir"],
        exclude = exclude + no_bef_translation,
        exclude_directories = 1,
    )

    # Pass generated .bef files to glob_lit_tests as per_test_extra_data.
    per_test_extra_data = {}
    for mlir_file in mlir_files:
        mlir_to_bef(mlir_file, tfrt_translate)
        per_test_extra_data[mlir_file] = [mlir_file[:-5] + ".bef"]

    glob_lit_tests(
        data = data,
        per_test_extra_data = per_test_extra_data,
        test_file_exts = ["mlir"],
        default_size = default_size,
        default_tags = default_tags,
        exclude = exclude,
        size_override = size_override,
        tags_override = tags_override,
    )
