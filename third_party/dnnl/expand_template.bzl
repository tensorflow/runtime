"""Rule for simple expansion of template files."""

def _expand_template_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

# Rule for simple expansion of template files.
expand_template = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The template file to expand",
        ),
        "substitutions": attr.string_dict(
            mandatory = True,
            doc = "A dictionary mapping strings to their substitutions",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The destination of the expanded file",
        ),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = _expand_template_impl,
    doc = """Performs a search over the template file for the keys in
substitutions, and replaces them with the corresponding values.

Typical usage:
  expand_template(
      name = "ExpandMyTemplate",
      src = "my.template",
      out = "my.txt",
      substitutions = {
        "$VAR1": "foo",
        "$VAR2": "bar",
      }
  )""",
)
