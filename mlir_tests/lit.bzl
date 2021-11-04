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

# Test definitions for Lit, the LLVM test runner.
#
# TODO(b/136126535): consider upstreaming at least a subset of this file.
"""Lit test macros"""

def glob_lit_tests(
        cfgs,
        test_file_exts,
        exclude = [],
        data = [],
        per_test_extra_data = {},
        default_size = "small",
        size_override = {},
        default_tags = [],
        tags_override = {},
        features = [],
        **kwargs):
    """Creates all plausible Lit tests (and their inputs) in this package.

    Args:
      cfgs: label, lit config files.
      test_file_exts: [str], extensions for files that are tests.
      exclude: [str], paths to exclude (for tests and inputs).
      data: [str], additional input data to the test.
      per_test_extra_data: {str: [str]}, extra data to attach to a given file.
      default_size: str, the test size for targets not in "size_override".
      size_override: {str: str}, sizes to use for specific tests.
      default_tags: [str], additional tags to attach to the test.
      tags_override: {str: str}, tags to add to specific tests.
      features: [str], extra arguments to pass to lit.
      **kwargs: arguments to pass on to lit_test()
    """

    # Add some default data/features.
    data = data + [
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:not",
    ]
    features = features + ["--verbose", "--show-all"]

    include = ["**/*." + ext for ext in test_file_exts]
    for test in native.glob(include, exclude):
        input = test + ".input"

        lit_input(
            name = input,
            srcs = [test],
            cfgs = cfgs,
        )

        native.py_test(
            name = test + ".test",
            srcs = ["@llvm-project//llvm:lit"],
            args = ["@$(rootpath :%s)" % input] + features,
            data = [input] + data + per_test_extra_data.get(test, []),
            tags = default_tags + tags_override.get(test, []),
            size = size_override.get(test, default_size),
            main = "lit.py",
            **kwargs
        )

def _lit_input_impl(ctx):
    dirs = {
        f.short_path[:-len(f.basename)]: True
        for f in ctx.files.cfgs
    }
    if len(dirs) != 1:
        fail("All 'cfgs' files must be in same directory")
    cfgs_dir = dirs.keys()[0]

    inputs = []
    for src in ctx.attr.srcs:
        # Use the cfgs directory so that the suite can be found even if it's not
        # in a parent directory of the test file. Add just basename of the input
        # so that tests are executed in the config.test_exec_root directory.
        # This requires that config.test_source_root is set to the directory of
        # the 'TEST_BINARY' environment variable that bazel provides.
        inputs.extend([cfgs_dir + f.basename for f in src.files.to_list()])

    output = ctx.actions.declare_file(ctx.label.name + ".out")
    ctx.actions.write(output, "\n".join(inputs))

    runfiles = [output] + ctx.files.cfgs + ctx.files.srcs
    return [DefaultInfo(
        files = depset([output]),
        runfiles = ctx.runfiles(runfiles),
    )]

lit_input = rule(
    implementation = _lit_input_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "Files to test.",
            allow_files = True,
            mandatory = True,
        ),
        "cfgs": attr.label(
            doc = "Suite cfg files.",
            allow_files = True,
            mandatory = True,
        ),
    },
    doc = "Generates a file containing the lit tests to run.",
)
