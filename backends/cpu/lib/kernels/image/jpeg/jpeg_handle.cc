/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements a memory destination for libjpeg. The design is similar
// to jdatadst.c in libjpeg.
//
// Based on tensorflow/core/lib/jpeg/jpeg_handle.cc

#include "jpeg_handle.h"

#include <setjmp.h>

namespace tfrt {
namespace image {
namespace jpeg {

void CatchError(j_common_ptr cinfo) {
  (*cinfo->err->output_message)(cinfo);
  jmp_buf *jpeg_jmpbuf = reinterpret_cast<jmp_buf *>(cinfo->client_data);
  jpeg_destroy(cinfo);
  longjmp(*jpeg_jmpbuf, 1);
}

// *****************************************************************************
// *****************************************************************************
// *****************************************************************************
// Source functions

// -----------------------------------------------------------------------------
void MemInitSource(j_decompress_ptr cinfo) {
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.next_input_byte = src->data;
  src->pub.bytes_in_buffer = src->datasize;
}

// -----------------------------------------------------------------------------
// We emulate the same error-handling as fill_input_buffer() from jdatasrc.c,
// for coherency's sake.
boolean MemFillInputBuffer(j_decompress_ptr cinfo) {
  static const JOCTET kEOIBuffer[2] = {0xff, JPEG_EOI};
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (src->pub.bytes_in_buffer == 0 && src->pub.next_input_byte == src->data) {
    // empty file -> treated as an error.
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    return FALSE;
  } else if (src->pub.bytes_in_buffer) {
    // if there's still some data left, it's probably corrupted
    return src->try_recover_truncated_jpeg ? TRUE : FALSE;
  } else if (src->pub.next_input_byte != kEOIBuffer &&
             src->try_recover_truncated_jpeg) {
    // In an attempt to recover truncated files, we insert a fake EOI
    WARNMS(cinfo, JWRN_JPEG_EOF);
    src->pub.next_input_byte = kEOIBuffer;
    src->pub.bytes_in_buffer = 2;
    return TRUE;
  } else {
    // We already inserted a fake EOI and it wasn't enough, so this time
    // it's really an error.
    ERREXIT(cinfo, JERR_FILE_READ);
    return FALSE;
  }
}

// -----------------------------------------------------------------------------
void MemTermSource(j_decompress_ptr cinfo) {}

// -----------------------------------------------------------------------------
void MemSkipInputData(j_decompress_ptr cinfo, int64_t jump) {
  MemSourceMgr *src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  if (jump < 0) {
    return;
  }
  if (jump > src->pub.bytes_in_buffer) {
    src->pub.bytes_in_buffer = 0;
    (void)MemFillInputBuffer(cinfo);  // warn with a fake EOI or error
  } else {
    src->pub.bytes_in_buffer -= jump;
    src->pub.next_input_byte += jump;
  }
}

// -----------------------------------------------------------------------------
void SetSrc(j_decompress_ptr cinfo, const void *data, uint64_t datasize,
            bool try_recover_truncated_jpeg) {
  MemSourceMgr *src;

  cinfo->src = reinterpret_cast<struct jpeg_source_mgr *>(
      (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                 JPOOL_PERMANENT, sizeof(MemSourceMgr)));

  src = reinterpret_cast<MemSourceMgr *>(cinfo->src);
  src->pub.init_source = MemInitSource;
  src->pub.fill_input_buffer = MemFillInputBuffer;
  src->pub.skip_input_data = MemSkipInputData;
  src->pub.resync_to_restart = jpeg_resync_to_restart;
  src->pub.term_source = MemTermSource;
  src->data = reinterpret_cast<const unsigned char *>(data);
  src->datasize = datasize;
  src->pub.bytes_in_buffer = 0;
  src->pub.next_input_byte = nullptr;
  src->try_recover_truncated_jpeg = try_recover_truncated_jpeg;
}

}  // namespace jpeg
}  // namespace image
}  // namespace tfrt
