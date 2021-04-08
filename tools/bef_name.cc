// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- BEF File Name utility ----------------------------------------------===//
//
// Given the name of a .mlir file as a command line argument, print the name of
// the corresponding .bef file.
//
// $ bef_name foo.mlir
// foo.bef

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr,
            "%s: Wrong number of arguments, expecting one MLIR file name "
            "argument\n",
            argv[0]);
    return 1;
  }

  const std::string mlir_file_name(argv[1]);
  const std::string mlir_suffix = ".mlir";
  const int size_without_suffix = mlir_file_name.size() - mlir_suffix.size();
  if (mlir_file_name.substr(size_without_suffix) != mlir_suffix) {
    fprintf(stderr, "%s: File name (%s) does not end in .mlir\n", argv[0],
            argv[1]);
    return 1;
  }

  printf("%s.bef\n", mlir_file_name.substr(0, size_without_suffix).c_str());
  return 0;
}
