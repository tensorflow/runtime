//===- raw_ostream.cpp - Lightweight raw output stream --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements support for lightweight stream output to file.
// The implementation is derived from llvm::raw_fd_ostream. It has mostly
// identical functionalities with the following changes:
// 1. Terminal color related functionalities are removed
// 2. Does not report_fatal_error() in its destructor. Removing such
//    functionalities helps us achieve smaller binary size.
//
//===----------------------------------------------------------------------===//

#include "llvm_derived/Support/raw_ostream.h"

#include <sys/stat.h>

// <fcntl.h> may provide O_BINARY.
#if defined(HAVE_FCNTL_H)
# include <fcntl.h>
#endif

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif

#if defined(__CYGWIN__)
#include <io.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

#ifdef _WIN32
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <iterator>
#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// raw_fd_ostream tfrt implementation
//===----------------------------------------------------------------------===//

static int getFD(llvm::StringRef Filename, std::error_code &EC,
                 llvm::sys::fs::CreationDisposition Disp,
                 llvm::sys::fs::FileAccess Access,
                 llvm::sys::fs::OpenFlags Flags) {
  assert((Access & llvm::sys::fs::FA_Write) &&
         "Cannot make a raw_ostream from a read-only descriptor!");

  // Handle "-" as stdout. Note that when we do this, we consider ourself
  // the owner of stdout and may set the "binary" flag globally based on Flags.
  if (Filename == "-") {
    EC = std::error_code();
    // If user requested binary then put stdout into binary mode if
    // possible.
    if (!(Flags & llvm::sys::fs::OF_Text)) llvm::sys::ChangeStdoutToBinary();
    return STDOUT_FILENO;
  }

  int FD;
  if (Access & llvm::sys::fs::FA_Read)
    EC = llvm::sys::fs::openFileForReadWrite(Filename, FD, Disp, Flags);
  else
    EC = llvm::sys::fs::openFileForWrite(Filename, FD, Disp, Flags);
  if (EC) return -1;

  return FD;
}

raw_fd_ostream::raw_fd_ostream(llvm::StringRef Filename, std::error_code &EC,
                               llvm::sys::fs::OpenFlags Flags)
    : raw_fd_ostream(Filename, EC, llvm::sys::fs::CD_CreateAlways,
                     llvm::sys::fs::FA_Write, Flags) {}

raw_fd_ostream::raw_fd_ostream(llvm::StringRef Filename, std::error_code &EC,
                               llvm::sys::fs::CreationDisposition Disp,
                               llvm::sys::fs::FileAccess Access,
                               llvm::sys::fs::OpenFlags Flags)
    : raw_fd_ostream(getFD(Filename, EC, Disp, Access, Flags), true) {}

/// FD is the file descriptor that this writes to.  If ShouldClose is true, this
/// closes the file when the stream is destroyed.
raw_fd_ostream::raw_fd_ostream(int fd, bool shouldClose, bool unbuffered)
    : raw_pwrite_stream(unbuffered), FD(fd), ShouldClose(shouldClose) {
  if (FD < 0) {
    ShouldClose = false;
    return;
  }

  // Do not attempt to close stdout or stderr. We used to try to maintain the
  // property that tools that support writing file to stdout should not also
  // write informational output to stdout, but in practice we were never able to
  // maintain this invariant. Many features have been added to LLVM and clang
  // (-fdump-record-layouts, optimization remarks, etc) that print to stdout, so
  // users must simply be aware that mixed output and remarks is a possibility.
  if (FD <= STDERR_FILENO) ShouldClose = false;

  // Get the starting position.
  off_t loc = ::lseek(FD, 0, SEEK_CUR);
  SupportsSeeking = loc != (off_t)-1;
  if (!SupportsSeeking)
    pos = 0;
  else
    pos = static_cast<uint64_t>(loc);
}

raw_fd_ostream::~raw_fd_ostream() {
  if (FD >= 0) {
    flush();
    if (ShouldClose) {
      if (auto EC = llvm::sys::Process::SafelyCloseFileDescriptor(FD))
        error_detected(EC);
    }
  }
}

void raw_fd_ostream::write_impl(const char *Ptr, size_t Size) {
  assert(FD >= 0 && "File already closed.");
  pos += Size;

  // The maximum write size is limited to INT32_MAX. A write
  // greater than SSIZE_MAX is implementation-defined in POSIX,
  // and Windows _write requires 32 bit input.
  size_t MaxWriteSize = INT32_MAX;

#if defined(__linux__)
  // It is observed that Linux returns EINVAL for a very large write (>2G).
  // Make it a reasonably small value.
  MaxWriteSize = 1024 * 1024 * 1024;
#endif  // defined(__linux__)

  do {
    size_t ChunkSize = std::min(Size, MaxWriteSize);
    ssize_t ret = ::write(FD, Ptr, ChunkSize);

    if (ret < 0) {
      // If it's a recoverable error, swallow it and retry the write.
      //
      // Ideally we wouldn't ever see EAGAIN or EWOULDBLOCK here, since
      // raw_ostream isn't designed to do non-blocking I/O. However, some
      // programs, such as old versions of bjam, have mistakenly used
      // O_NONBLOCK. For compatibility, emulate blocking semantics by
      // spinning until the write succeeds. If you don't want spinning,
      // don't use O_NONBLOCK file descriptors with raw_ostream.
      if (errno == EINTR || errno == EAGAIN
#ifdef EWOULDBLOCK
          || errno == EWOULDBLOCK
#endif  // EWOULDBLOCK
      )
        continue;

      // Otherwise it's a non-recoverable error. Note it and quit.
      error_detected(std::error_code(errno, std::generic_category()));
      break;
    }

    // The write may have written some or all of the data. Update the
    // size and buffer pointer to reflect the remainder that needs
    // to be written. If there are no bytes left, we're done.
    Ptr += ret;
    Size -= ret;
  } while (Size > 0);
}

void raw_fd_ostream::close() {
  assert(ShouldClose);
  ShouldClose = false;
  flush();
  if (auto EC = llvm::sys::Process::SafelyCloseFileDescriptor(FD))
    error_detected(EC);
  FD = -1;
}

uint64_t raw_fd_ostream::seek(uint64_t off) {
  assert(SupportsSeeking && "Stream does not support seeking!");
  flush();
#if defined(HAVE_LSEEK64)
  pos = ::lseek64(FD, off, SEEK_SET);
#else
  pos = ::lseek(FD, off, SEEK_SET);
#endif
  if (pos == static_cast<uint64_t>(-1))
    error_detected(std::error_code(errno, std::generic_category()));
  return pos;
}

void raw_fd_ostream::pwrite_impl(const char *Ptr, size_t Size,
                                 uint64_t Offset) {
  uint64_t Pos = tell();
  seek(Offset);
  write(Ptr, Size);
  seek(Pos);
}

size_t raw_fd_ostream::preferred_buffer_size() const {
#if !defined(__minix) && !defined(_WIN32)
  // Minix has no st_blksize.
  assert(FD >= 0 && "File not yet open!");
  struct stat statbuf;
  if (fstat(FD, &statbuf) != 0) return 0;

  // If this is a terminal, don't use buffering. Line buffering
  // would be a more traditional thing to do, but it's not worth
  // the complexity.
  if (S_ISCHR(statbuf.st_mode) && isatty(FD)) return 0;
  // Return the preferred block size.
  return statbuf.st_blksize;
#else
  return llvm::raw_ostream::preferred_buffer_size();
#endif
}

void raw_fd_ostream::anchor() {}

//===----------------------------------------------------------------------===//
//  outs(), errs()
//===----------------------------------------------------------------------===//

/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
llvm::raw_ostream &outs() {
  // Set buffer settings to model stdout behavior.
  std::error_code EC;
  static raw_fd_ostream S("-", EC, llvm::sys::fs::OF_None);  // NOLINT
  assert(!EC);
  return S;
}

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
llvm::raw_ostream &errs() {
  // Set standard error to be unbuffered by default.
  static raw_fd_ostream S(STDERR_FILENO, false, true);  // NOLINT
  return S;
}

}  // namespace tfrt
