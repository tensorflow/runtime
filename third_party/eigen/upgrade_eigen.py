"""Update workspace.bzl with latest version of Eigen.

python upgrade_eigen.py [COMMIT] [SHA256]
"""

from __future__ import print_function
import os
import re
import sys

COMMIT_PATTERN = re.compile(r'EIGEN_COMMIT = "[0-9a-f]{40}"')
SHA256_PATTERN = re.compile(r'EIGEN_SHA256 = "[0-9a-f]{64}"')
WORKSPACE_FILE = 'workspace.bzl'


def main(args):
  """Update eigen version in workspace.bzl."""
  if '--help' in args or '-h' in args or 'help' in args:
    print(__doc__.strip())
    sys.exit(0)

  current_path = os.path.dirname(os.path.abspath(__file__))
  workspace_file = os.path.join(current_path, WORKSPACE_FILE)
  if not os.path.exists(workspace_file):
    print(
        'Please run from the same directory as the workspace.bzl file',
        file=sys.stderr)
    sys.exit(1)

  with open(workspace_file) as f:
    workspace_data = f.read()

  commit = args[0][:40]
  workspace_data, count = COMMIT_PATTERN.subn('EIGEN_COMMIT = "%s"' % commit,
                                              workspace_data)
  if not count:
    print('Failed to replace commit.', file=sys.stderr)
    sys.exit(1)

  sha256 = args[1][:64]
  workspace_data, count = SHA256_PATTERN.subn('EIGEN_SHA256 = "%s"' % sha256,
                                              workspace_data)
  if not count:
    print('Failed to replace sha256.', file=sys.stderr)
    sys.exit(1)

  if not os.access(workspace_file, os.W_OK):
    print('Cannot edit workspace file.')
    sys.exit(1)

  with open(workspace_file, 'w') as f:
    f.write(workspace_data)

  print('Updated ' + workspace_file, file=sys.stderr)


if __name__ == '__main__':
  main(args=sys.argv[1:])
