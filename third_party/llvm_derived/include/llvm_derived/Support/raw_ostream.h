//===- raw_ostream.h - Lightweight raw output stream ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a lightweight output stream and methods.
// The definition is derived from llvm::raw_fd_ostream. It has mostly
// identical functionalities, with terminal color related methods being removed.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LLVM_DERIVED_OSTREAM_H_
#define TFRT_LLVM_DERIVED_OSTREAM_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <system_error>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {

// TODO(tfrt-devs): Port support for Windows from llvm.
#ifdef _WIN32
using raw_fd_ostream = llvm::raw_fd_ostream;
#else   // !_WIN32
//===----------------------------------------------------------------------===//
// Lightweight File Output Streams
//
// A drop in replacement for llvm::raw_fd_ostream. This is a stream that writes
// to a file descriptor. This raw_fd_ostream removes support for color, WIN32,
// and does not report fatal error, compared to llvm::raw_fd_ostream.
//===----------------------------------------------------------------------===//
class raw_fd_ostream : public llvm::raw_pwrite_stream {
  int FD;
  bool ShouldClose;
  bool SupportsSeeking = false;

  std::error_code EC;

  uint64_t pos = 0;

  // See raw_ostream::write_impl.
  void write_impl(const char *Ptr, size_t Size) override;

  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override;

  // Return the current position within the stream, not counting the bytes
  // currently in the buffer.
  uint64_t current_pos() const override { return pos; }

  // Determine an efficient buffer size.
  size_t preferred_buffer_size() const override;

  // Set the flag indicating that an output error has been encountered.
  void error_detected(std::error_code EC) { this->EC = EC; }

  void anchor() override;

 public:
  // Open the specified file for writing. If an error occurs, information
  // about the error is put into EC, and the stream should be immediately
  // destroyed;
  // \p Flags allows optional flags to control how the file will be opened.
  //
  // As a special case, if Filename is "-", then the stream will use
  // STDOUT_FILENO instead of opening a file. This will not close the stdout
  // descriptor.
  raw_fd_ostream(llvm::StringRef Filename, std::error_code &EC,
                 llvm::sys::fs::OpenFlags Flags);
  raw_fd_ostream(llvm::StringRef Filename, std::error_code &EC,
                 llvm::sys::fs::CreationDisposition Disp,
                 llvm::sys::fs::FileAccess Access,
                 llvm::sys::fs::OpenFlags Flags);
  // FD is the file descriptor that this writes to.  If ShouldClose is true,
  // this closes the file when the stream is destroyed. If FD is for stdout or
  // stderr, it will not be closed.
  raw_fd_ostream(int fd, bool shouldClose, bool unbuffered = false);

  ~raw_fd_ostream() override;

  // Manually flush the stream and close the file. Note that this does not call
  // fsync.
  void close();

  bool supportsSeeking() { return SupportsSeeking; }

  // Flushes the stream and repositions the underlying file descriptor position
  // to the offset specified from the beginning of the file.
  uint64_t seek(uint64_t off);

  std::error_code error() const { return EC; }

  // Return the value of the flag in this raw_fd_ostream indicating whether an
  // output error has been encountered.
  // This doesn't implicitly flush any pending output.  Also, it doesn't
  // guarantee to detect all errors unless the stream has been closed.
  bool has_error() const { return bool(EC); }

  // Set the flag read by has_error() to false. If the error flag is set at the
  // time when this raw_ostream's destructor is called, report_fatal_error is
  // called to report the error. Use clear_error() after handling the error to
  // avoid this behavior.
  //
  //   "Errors should never pass silently.
  //    Unless explicitly silenced."
  //      - from The Zen of Python, by Tim Peters
  //
  void clear_error() { EC = std::error_code(); }
};
#endif  // !_WIN32

// Drop in replacement for llvm::outs() and llvm::errs() since those use
// llvm::raw_fd_ostream, which introduces unecessary dependencies and is hurtful
// to tfrt binary size.

// This returns a reference to a raw_ostream for standard output.
// Use it like: outs() << "foo" << "bar";
llvm::raw_ostream &outs();

// This returns a reference to a raw_ostream for standard error. Use it like:
// errs() << "foo" << "bar";
llvm::raw_ostream &errs();

}  // namespace tfrt

#endif  // TFRT_LLVM_DERIVED_OSTREAM_H_
