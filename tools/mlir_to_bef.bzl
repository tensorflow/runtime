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
    """Runs "tfrt_translate -mlir-to-bef $test.mlir" to create $test.bef.

    Args:
      name: the name of mlir test.
      tfrt_translate: translation tool to use.

    Returns:
      the name of generated bef file.
    """
    bef_file = name + ".bef"
    rule_name = "bef_" + name
    native.genrule(
        name = rule_name,
        srcs = [name],
        outs = [bef_file],
        cmd = "$(location " + tfrt_translate + ") -mlir-to-bef $(location " + name + ") > $@",
        exec_tools = [tfrt_translate],
    )
    return bef_file

def glob_tfrt_lit_tests(
        name = "glob_tfrt_lit_tests",
        data = [],
        # copybara:uncomment driver = "@tf_runtime//mlir_tests:run_lit.sh",
        exclude = [],
        # Do not run "tfrt_translate -mlir-to-bef" on these files.
        no_bef_translation = [],
        tfrt_translate = "",
        tags_override = {},
        **kwargs):
    """Run mlir_to_bef on all .mlir files and invoke glob_lit_tests."""

    if tfrt_translate == "":
        tfrt_translate = "@tf_runtime//tools:tfrt_translate"

    mlir_files = native.glob(
        include = ["**/*.mlir"],
        exclude = exclude + no_bef_translation,
        exclude_directories = 1,
    )

    # copybara:uncomment_begin
    # # TODO(csigg): Change to `select()` once lit_test supports it.
    # data = data + ["@llvm-project//mlir:run_lit.sh"]
    # copybara:uncomment_end

    # Pass generated .bef files to glob_lit_tests as per_test_extra_data.
    per_test_extra_data = {}
    for mlir_file in mlir_files:
        bef_file = mlir_to_bef(mlir_file, tfrt_translate)
        per_test_extra_data[mlir_file] = [bef_file]

        # Generate mpm files to allow running tests on production machines (e.g. borg)
        # copybara:uncomment mlir_to_mpm(mlir_file, bef_file, data)

    glob_lit_tests(
        data = data,
        per_test_extra_data = per_test_extra_data,
        # copybara:uncomment driver = driver,
        test_file_exts = ["mlir"],
        exclude = exclude,
        tags_override = tags_override,
        **kwargs
    )
