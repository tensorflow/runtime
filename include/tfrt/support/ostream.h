//===- ostream.h - Lightweight raw output stream ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/
//
// This file defines a lightweight output stream and methods.
// The definition is derived from llvm::raw_fd_ostream. It has mostly
// identical functionalities, with terminal color related methods being removed.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_SUPPORT_OSTREAM_H_
#define TFRT_SUPPORT_OSTREAM_H_

// The lightweight implementation of tfrt::raw_fd_ostream derived from
// llvm::raw_fd_ostream.
#include "llvm_derived/ostream.h"

#endif  // TFRT_SUPPORT_OSTREAM_H_
