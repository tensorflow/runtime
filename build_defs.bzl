# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BUILD rules for tfrt."""

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

def if_google(google_value, oss_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value  # copybara:comment_replace return google_value

TFRT_COPTS = select({
    "@tf_runtime//:windows": [
        "-Zc:inline",
        "-Zc:strictStrings",
        "-Zc:rvalueCast",
        "-Oi",
        "-wd4141",
        "-wd4146",
        "-wd4180",
        "-wd4244",
        "-wd4258",
        "-wd4267",
        "-wd4291",
        "-wd4345",
        "-wd4351",
        "-wd4355",
        "-wd4456",
        "-wd4457",
        "-wd4458",
        "-wd4459",
        "-wd4503",
        "-wd4624",
        "-wd4722",
        "-wd4800",
        "-wd4100",
        "-wd4127",
        "-wd4512",
        "-wd4505",
        "-wd4610",
        "-wd4510",
        "-wd4702",
        "-wd4245",
        "-wd4706",
        "-wd4310",
        "-wd4701",
        "-wd4703",
        "-wd4389",
        "-wd4611",
        "-wd4805",
        "-wd4204",
        "-wd4577",
        "-wd4091",
        "-wd4592",
        "-wd4319",
        "-wd4324",
        "-w14062",
        "-we4238",
    ],
    "//conditions:default": ["-Wno-unused-local-typedef"],
    "@tf_runtime//:disable_rtti_and_exceptions": [
        "-fno-rtti",  # Disable RTTI.
        "-fno-exceptions",  # Disable exceptions.
    ],
})

TFRT_LINKOPTS = select({
    "@tf_runtime//:windows": [],
    "//conditions:default": [
        "-ldl",
        "-lm",
        "-lpthread",  # copybara:comment
    ],
})

TFRT_FEATURES = select({
    "//conditions:default": [],
    "@tf_runtime//:disable_rtti_and_exceptions": [
        # Precompiled header modules do not work with fno-rtti or
        # fno-exceptions. See b/137799263.
        "-use_header_modules",
    ],
})

# Use relative include path.
TFRT_INCLUDES = ["include"]

def tfrt_cc_library(
        name = "",
        srcs = [],
        deps = [],
        includes = [],
        copts = [],
        linkopts = [],
        features = [],
        alwayslink_static_registration_src = "",
        alwayslink_static_registration_deps = [],
        **kwargs):
    """A cc_library with tfrt-specific options."""
    for tfrt_inc in TFRT_INCLUDES:
        for inc in includes:
            if tfrt_inc == inc:
                fail(
                    "Found include path '" + inc + "' that is already in TFRT_INCLUDES",
                    "includes",
                )

    native.cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        includes = TFRT_INCLUDES + includes,
        copts = TFRT_COPTS + copts,
        linkopts = TFRT_LINKOPTS + linkopts,
        features = TFRT_FEATURES + features,
        **kwargs
    )

    # Generate a second target with "_alwayslink" suffix in its name.
    if alwayslink_static_registration_src != "":
        native.cc_library(
            name = name + "_alwayslink",
            srcs = [alwayslink_static_registration_src],
            # Depend on non-alwayslink target to avoid duplicate symbol linker
            # error.
            deps = deps + alwayslink_static_registration_deps + [":" + name],
            includes = TFRT_INCLUDES + includes,
            copts = TFRT_COPTS + copts,
            linkopts = TFRT_LINKOPTS + linkopts,
            features = TFRT_FEATURES + features,
            alwayslink = 1,
            **kwargs
        )

def tfrt_cc_binary(
        includes = [],
        copts = [],
        linkopts = [],
        features = [],
        **kwargs):
    """A cc_binary with tfrt-specific options."""
    native.cc_binary(
        copts = TFRT_COPTS + copts,
        includes = TFRT_INCLUDES + includes,
        linkopts = TFRT_LINKOPTS + linkopts,
        features = TFRT_FEATURES,
        **kwargs
    )

def tfrt_cc_test(
        includes = [],
        copts = select({
            "@tf_runtime//:windows": [],
            "//conditions:default": [
                "-Wno-private-header",
            ],
        }),
        linkopts = [],
        features = [],
        **kwargs):
    """A cc_test with tfrt-specific options."""
    native.cc_test(
        includes = TFRT_INCLUDES + includes,
        copts = TFRT_COPTS + copts,
        linkopts = TFRT_LINKOPTS + linkopts,
        features = TFRT_FEATURES + features,
        **kwargs
    )

def tfrt_py_binary(
        tags = [],
        **kwargs):
    """A py_binary with tfrt-specific options."""
    native.py_binary(
        tags = tags + ["do_not_disable_rtti"],
        **kwargs
    )

def _make_variable_impl(ctx):
    value = ctx.build_setting_value
    if value not in ctx.attr.values:
        fail("Error setting " + str(ctx.label) + ": invalid value '" +
             value + "'. Allowed values are " + ctx.attr.values)
    return platform_common.TemplateVariableInfo({ctx.label.name: value})

make_variable = rule(
    implementation = _make_variable_impl,
    build_setting = config.string(flag = True),
    attrs = {"values": attr.string_list()},
)
