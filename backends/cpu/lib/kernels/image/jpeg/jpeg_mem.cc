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

// This file defines functions to compress and uncompress JPEG data to and from
// memory, as well as some direct manipulations of JPEG string.
//
// Based on tensorflow/core/lib/jpeg/jpeg_mem.cc

#include "jpeg_mem.h"

#include <setjmp.h>

#include <string>

#include "jpeg_handle.h"
#include "llvm/Support/raw_ostream.h"

namespace tfrt {
namespace image {
namespace jpeg {

// -----------------------------------------------------------------------------
// Decompression

namespace {

enum JPEGErrors {
  JPEGERRORS_OK,
  JPEGERRORS_UNEXPECTED_END_OF_DATA,
  JPEGERRORS_BAD_PARAM
};

// Prevent bad compiler behavior in ASAN mode by wrapping most of the
// arguments in a struct struct.
class FewerArgsForCompiler {
 public:
  FewerArgsForCompiler(int datasize, const UncompressFlags& flags,
                       int64_t* nwarn,
                       std::function<uint8_t*(int, int, int)> allocate_output)
      : datasize_(datasize),
        flags_(flags),
        pnwarn_(nwarn),
        allocate_output_(std::move(allocate_output)),
        height_read_(0),
        height_(0),
        stride_(0) {
    if (pnwarn_ != nullptr) *pnwarn_ = 0;
  }

  const int datasize_;
  const UncompressFlags flags_;
  int64_t* const pnwarn_;
  std::function<uint8_t*(int, int, int)> allocate_output_;
  int height_read_;  // number of scanline lines successfully read
  int height_;
  int stride_;
};

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
// If in fuzzing mode, don't print any error message as that slows down fuzzing.
// See also http://llvm.org/docs/LibFuzzer.html#fuzzer-friendly-build-mode
void no_print(j_common_ptr cinfo) {}
#endif

uint8_t* UncompressLow(const void* srcdata, FewerArgsForCompiler* argball) {
  // unpack the argball
  const int datasize = argball->datasize_;
  const auto& flags = argball->flags_;
  const int ratio = flags.ratio;
  int components = flags.components;
  int stride = flags.stride;                // may be 0
  int64_t* const nwarn = argball->pnwarn_;  // may be NULL

  // Can't decode if the ratio is not recognized by libjpeg
  if ((ratio != 1) && (ratio != 2) && (ratio != 4) && (ratio != 8)) {
    return nullptr;
  }

  // Channels must be autodetect, grayscale, or rgb.
  if (!(components == 0 || components == 1 || components == 3)) {
    return nullptr;
  }

  // if empty image, return
  if (datasize == 0 || srcdata == nullptr) return nullptr;

  // Declare temporary buffer pointer here so that we can free on error paths
  JSAMPLE* tempdata = nullptr;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  JPEGErrors error = JPEGERRORS_OK;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jerr.error_exit = CatchError;

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  jerr.output_message = no_print;
#endif

  jmp_buf jpeg_jmpbuf;
  cinfo.client_data = &jpeg_jmpbuf;
  if (setjmp(jpeg_jmpbuf)) {
    delete[] tempdata;
    return nullptr;
  }

  jpeg_create_decompress(&cinfo);
  SetSrc(&cinfo, srcdata, datasize, flags.try_recover_truncated_jpeg);
  jpeg_read_header(&cinfo, TRUE);

  // Set components automatically if desired, autoconverting cmyk to rgb.
  if (components == 0) components = std::min(cinfo.num_components, 3);

  // set grayscale and ratio parameters
  switch (components) {
    case 1:
      cinfo.out_color_space = JCS_GRAYSCALE;
      break;
    case 3:
      if (cinfo.jpeg_color_space == JCS_CMYK ||
          cinfo.jpeg_color_space == JCS_YCCK) {
        // Always use cmyk for output in a 4 channel jpeg. libjpeg has a builtin
        // decoder.  We will further convert to rgb below.
        cinfo.out_color_space = JCS_CMYK;
      } else {
        cinfo.out_color_space = JCS_RGB;
      }
      break;
    default:
      llvm::errs() << " Invalid components value " << components << "\n";
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
  }
  cinfo.do_fancy_upsampling = boolean(flags.fancy_upscaling);
  cinfo.scale_num = 1;
  cinfo.scale_denom = ratio;
  cinfo.dct_method = flags.dct_method;

  // Determine the output image size before attempting decompress to prevent
  // OOM'ing doing the decompress
  jpeg_calc_output_dimensions(&cinfo);

  int64_t total_size = static_cast<int64_t>(cinfo.output_height) *
                       static_cast<int64_t>(cinfo.output_width) *
                       static_cast<int64_t>(cinfo.num_components);
  // Some of the internal routines do not gracefully handle ridiculously
  // large images, so fail fast.
  if (cinfo.output_width <= 0 || cinfo.output_height <= 0) {
    llvm::errs() << "Invalid image size: " << cinfo.output_width << " x "
                 << cinfo.output_height;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }
  if (total_size >= (1LL << 29)) {
    llvm::errs() << "Image too large: " << total_size;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }

  jpeg_start_decompress(&cinfo);

  JDIMENSION target_output_width = cinfo.output_width;
  JDIMENSION target_output_height = cinfo.output_height;
  JDIMENSION skipped_scanlines = 0;

  // check for compatible stride
  const int min_stride = target_output_width * components * sizeof(JSAMPLE);
  if (stride == 0) {
    stride = min_stride;
  } else if (stride < min_stride) {
    llvm::errs() << "Incompatible stride: " << stride << " < " << min_stride;
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }

  // Remember stride and height for use in Uncompress
  argball->height_ = target_output_height;
  argball->stride_ = stride;

#if !defined(LIBJPEG_TURBO_VERSION)
  uint8_t* dstdata = nullptr;
  if (flags.crop) {
    dstdata = new JSAMPLE[stride * target_output_height];
  } else {
    dstdata = argball->allocate_output_(target_output_width,
                                        target_output_height, components);
  }
#else
  uint8_t* dstdata = argball->allocate_output_(
      target_output_width, target_output_height, components);
#endif
  if (dstdata == nullptr) {
    jpeg_destroy_decompress(&cinfo);
    return nullptr;
  }
  JSAMPLE* output_line = static_cast<JSAMPLE*>(dstdata);

  // jpeg_read_scanlines requires the buffers to be allocated based on
  // cinfo.output_width, but the target image width might be different if crop
  // is enabled and crop_width is not MCU aligned. In this case, we need to
  // realign the scanline output to achieve the exact cropping.  Notably, only
  // cinfo.output_width needs to fall on MCU boundary, while cinfo.output_height
  // has no such constraint.
  const bool need_realign_cropped_scanline =
      (target_output_width != cinfo.output_width);
  const bool use_cmyk = (cinfo.out_color_space == JCS_CMYK);

  if (use_cmyk) {
    // Temporary buffer used for CMYK -> RGB conversion.
    tempdata = new JSAMPLE[cinfo.output_width * 4];
  } else if (need_realign_cropped_scanline) {
    // Temporary buffer used for MCU-aligned scanline data.
    tempdata = new JSAMPLE[cinfo.output_width * components];
  }

  // If there is an error reading a line, this aborts the reading.
  // Save the fraction of the image that has been read.
  argball->height_read_ = target_output_height;

  // These variables are just to avoid repeated computation in the loop.
  const int max_scanlines_to_read = skipped_scanlines + target_output_height;
  const int mcu_align_offset =
      (cinfo.output_width - target_output_width) * (use_cmyk ? 4 : components);
  while (cinfo.output_scanline < max_scanlines_to_read) {
    int num_lines_read = 0;
    if (use_cmyk) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        // Convert CMYK to RGB if scanline read succeeded.
        for (size_t i = 0; i < target_output_width; ++i) {
          int offset = 4 * i;
          if (need_realign_cropped_scanline) {
            // Align the offset for MCU boundary.
            offset += mcu_align_offset;
          }
          const int c = tempdata[offset + 0];
          const int m = tempdata[offset + 1];
          const int y = tempdata[offset + 2];
          const int k = tempdata[offset + 3];
          int r, g, b;
          if (cinfo.saw_Adobe_marker) {
            r = (k * c) / 255;
            g = (k * m) / 255;
            b = (k * y) / 255;
          } else {
            r = (255 - k) * (255 - c) / 255;
            g = (255 - k) * (255 - m) / 255;
            b = (255 - k) * (255 - y) / 255;
          }
          output_line[3 * i + 0] = r;
          output_line[3 * i + 1] = g;
          output_line[3 * i + 2] = b;
        }
      }
    } else if (need_realign_cropped_scanline) {
      num_lines_read = jpeg_read_scanlines(&cinfo, &tempdata, 1);
      if (num_lines_read > 0) {
        memcpy(output_line, tempdata + mcu_align_offset, min_stride);
      }
    } else {
      num_lines_read = jpeg_read_scanlines(&cinfo, &output_line, 1);
    }
    // Handle error cases
    if (num_lines_read == 0) {
      llvm::errs() << "Premature end of JPEG data. Stopped at line "
                   << cinfo.output_scanline - skipped_scanlines << "/"
                   << target_output_height;
      if (!flags.try_recover_truncated_jpeg) {
        argball->height_read_ = cinfo.output_scanline - skipped_scanlines;
        error = JPEGERRORS_UNEXPECTED_END_OF_DATA;
      } else {
        for (size_t line = cinfo.output_scanline; line < max_scanlines_to_read;
             ++line) {
          if (line == 0) {
            // If even the first line is missing, fill with black color
            memset(output_line, 0, min_stride);
          } else {
            // else, just replicate the line above.
            memcpy(output_line, output_line - stride, min_stride);
          }
          output_line += stride;
        }
        argball->height_read_ =
            target_output_height;  // consider all lines as read
        // prevent error-on-exit in libjpeg:
        cinfo.output_scanline = max_scanlines_to_read;
      }
      break;
    }
    assert(num_lines_read == 1 && "should read only 1 line");
    output_line += stride;
  }
  delete[] tempdata;
  tempdata = nullptr;

#if defined(LIBJPEG_TURBO_VERSION)
  if (flags.crop && cinfo.output_scanline < cinfo.output_height) {
    // Skip the rest of scanlines, required by jpeg_destroy_decompress.
    jpeg_skip_scanlines(&cinfo,
                        cinfo.output_height - flags.crop_y - flags.crop_height);
    // After this, cinfo.output_height must be equal to cinfo.output_height;
    // otherwise, jpeg_destroy_decompress would fail.
  }
#endif

  // Convert the RGB data to RGBA, with alpha set to 0xFF to indicate
  // opacity.
  // RGBRGBRGB... --> RGBARGBARGBA...
  if (components == 4) {
    // Start on the last line.
    JSAMPLE* scanlineptr = static_cast<JSAMPLE*>(
        dstdata + static_cast<int64_t>(target_output_height - 1) * stride);
    const JSAMPLE kOpaque = -1;  // All ones appropriate for JSAMPLE.
    const int right_rgb = (target_output_width - 1) * 3;
    const int right_rgba = (target_output_width - 1) * 4;

    for (int y = target_output_height; y-- > 0;) {
      // We do all the transformations in place, going backwards for each row.
      const JSAMPLE* rgb_pixel = scanlineptr + right_rgb;
      JSAMPLE* rgba_pixel = scanlineptr + right_rgba;
      scanlineptr -= stride;
      for (int x = target_output_width; x-- > 0;
           rgba_pixel -= 4, rgb_pixel -= 3) {
        // We copy the 3 bytes at rgb_pixel into the 4 bytes at rgba_pixel
        // The "a" channel is set to be opaque.
        rgba_pixel[3] = kOpaque;
        rgba_pixel[2] = rgb_pixel[2];
        rgba_pixel[1] = rgb_pixel[1];
        rgba_pixel[0] = rgb_pixel[0];
      }
    }
  }

  switch (components) {
    case 1:
      if (cinfo.output_components != 1) {
        error = JPEGERRORS_BAD_PARAM;
      }
      break;
    case 3:
    case 4:
      if (cinfo.out_color_space == JCS_CMYK) {
        if (cinfo.output_components != 4) {
          error = JPEGERRORS_BAD_PARAM;
        }
      } else {
        if (cinfo.output_components != 3) {
          error = JPEGERRORS_BAD_PARAM;
        }
      }
      break;
    default:
      // will never happen, should be caught by the previous switch
      llvm::errs() << "Invalid components value " << components << "\n";
      jpeg_destroy_decompress(&cinfo);
      return nullptr;
  }

  // save number of warnings if requested
  if (nwarn != nullptr) {
    *nwarn = cinfo.err->num_warnings;
  }

  // Handle errors in JPEG
  switch (error) {
    case JPEGERRORS_OK:
      jpeg_finish_decompress(&cinfo);
      break;
    case JPEGERRORS_UNEXPECTED_END_OF_DATA:
    case JPEGERRORS_BAD_PARAM:
      jpeg_abort(reinterpret_cast<j_common_ptr>(&cinfo));
      break;
    default:
      llvm::errs() << "Unhandled case " << error;
      break;
  }

  jpeg_destroy_decompress(&cinfo);
  return dstdata;
}

}  // namespace

// -----------------------------------------------------------------------------
//  We do the apparently silly thing of packing 5 of the arguments
//  into a structure that is then passed to another routine
//  that does all the work.  The reason is that we want to catch
//  fatal JPEG library errors with setjmp/longjmp, and g++ and
//  associated libraries aren't good enough to guarantee that 7
//  parameters won't get clobbered by the longjmp.  So we help
//  it out a little.
uint8_t* Uncompress(const void* srcdata, int datasize,
                    const UncompressFlags& flags, int64_t* nwarn,
                    std::function<uint8_t*(int, int, int)> allocate_output) {
  FewerArgsForCompiler argball(datasize, flags, nwarn,
                               std::move(allocate_output));
  uint8_t* const dstdata = UncompressLow(srcdata, &argball);

  const float fraction_read =
      argball.height_ == 0
          ? 1.0
          : (static_cast<float>(argball.height_read_) / argball.height_);
  if (dstdata == nullptr ||
      fraction_read < std::min(1.0f, flags.min_acceptable_fraction)) {
    // Major failure, none or too-partial read returned; get out
    return nullptr;
  }

  // If there was an error in reading the jpeg data,
  // set the unread pixels to black
  if (argball.height_read_ != argball.height_) {
    const int first_bad_line = argball.height_read_;
    uint8_t* start = dstdata + first_bad_line * argball.stride_;
    const int nbytes = (argball.height_ - first_bad_line) * argball.stride_;
    memset(static_cast<void*>(start), 0, nbytes);
  }

  return dstdata;
}

}  // namespace jpeg
}  // namespace image
}  // namespace tfrt
