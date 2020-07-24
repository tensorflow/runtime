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

TFRT_COPTS = select({
    "//conditions:default": [],
    "@tf_runtime//:disable_rtti_and_exceptions": [
        # Disable RTTI.
        "-fno-rtti",
        # Disable exceptions.
        "-fno-exceptions",
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
        features = [],
        alwayslink_static_registration_src = "",
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
            deps = deps + [":" + name],
            includes = TFRT_INCLUDES + includes,
            copts = TFRT_COPTS + copts,
            features = TFRT_FEATURES + features,
            alwayslink = 1,
            **kwargs
        )

def tfrt_cc_binary(
        includes = [],
        copts = [],
        features = [],
        **kwargs):
    """A cc_binary with tfrt-specific options."""
    native.cc_binary(
        copts = TFRT_COPTS + copts,
        includes = TFRT_INCLUDES + includes,
        features = TFRT_FEATURES,
        **kwargs
    )

def tfrt_cc_test(
        includes = [],
        copts = [],
        features = [],
        **kwargs):
    """A cc_test with tfrt-specific options."""
    native.cc_test(
        includes = TFRT_INCLUDES + includes,
        copts = TFRT_COPTS + copts,
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
