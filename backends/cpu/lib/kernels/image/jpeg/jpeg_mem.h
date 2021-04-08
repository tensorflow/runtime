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

// This file defines functions to compress and uncompress JPEG files
// to and from memory. It provides interfaces for raw images
// (data array and size fields).
// Direct manipulation of JPEG strings are supplied: Flip, Rotate, Crop.
//
// Based on tensorflow/core/lib/jpeg/jpeg_mem.h

#ifndef TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_MEM_H_
#define TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_MEM_H_

#include <functional>

extern "C" {
#include "third_party/libjpeg_turbo/jerror.h"
#include "third_party/libjpeg_turbo/jpeglib.h"
}

namespace tfrt {
namespace image {
namespace jpeg {

// Flags for Uncompress
struct UncompressFlags {
  // ratio can be 1, 2, 4, or 8 and represent the denominator for the scaling
  // factor (eg ratio = 4 means that the resulting image will be at 1/4 original
  // size in both directions).
  int ratio = 1;

  // The number of bytes per pixel (1, 3 or 4), or 0 for autodetect.
  int components = 0;

  // If true, decoder will use a slower but nicer upscaling of the chroma
  // planes (yuv420/422 only).
  bool fancy_upscaling = true;

  // If true, will attempt to fill in missing lines of truncated files
  bool try_recover_truncated_jpeg = false;

  // The minimum required fraction of lines read before the image is accepted.
  float min_acceptable_fraction = 1.0;

  // The distance in bytes from one scanline to the other.  Should be at least
  // equal to width*components*sizeof(JSAMPLE).  If 0 is passed, the stride
  // used will be this minimal value.
  int stride = 0;

  // Setting of J_DCT_METHOD enum in jpeglib.h, for choosing which
  // algorithm to use for DCT/IDCT.
  //
  // Setting this has a quality/speed trade-off implication.
  J_DCT_METHOD dct_method = JDCT_DEFAULT;

  // Settings of crop window before decompression.
  bool crop = false;
  // Vertical coordinate of the top-left corner of the result in the input.
  int crop_x = 0;
  // Horizontal coordinate of the top-left corner of the result in the input.
  int crop_y = 0;
  // Width of the output image.
  int crop_width = 0;
  // Height of the output image.
  int crop_height = 0;
};

// This function that allocates memory via a callback. The callback
// arguments are (width, height, components).  If the size is known ahead of
// time this function can return an existing buffer; passing a callback allows
// the buffer to be shaped based on the JPEG header.  The caller is responsible
// for freeing the memory *even along error paths*.
uint8_t* Uncompress(const void* srcdata, int datasize,
                    const UncompressFlags& flags, int64_t* nwarn,
                    std::function<uint8_t*(int, int, int)> allocate_output);

}  // namespace jpeg
}  // namespace image
}  // namespace tfrt

#endif  // TFRT_BACKENDS_CPU_LIB_KERNELS_IMAGE_JPEG_JPEG_MEM_H_
