/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===- partial_packets.h ----------------------------------------*- C++ -*-===//
//
// This file defines helper functions for loading partial packets from Eigen
// expression evaluators.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_PARTIAL_PACKETS_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_PARTIAL_PACKETS_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace Eigen {
namespace internal {

// TensorEvaluatorHasPartialPacket provides `value` that is true if
// TensorEvaluatorType supports partial packet access:
//
//   using mask_t = unpacket_traits<PacketType>::mask_t;
//   PacketType partialPacket<PacketType>(IndexType, mask_t) const;
//
// and if the PacketType supports masked load (on CPUs supporting AVX and
// above).
//
// Partial packets are used to:
//
// 1) Split the packet over two columns in eigen based spatial convolution and
//    use partial loads for each individual part before combining them to get
//    the required packet. This class is used to pick the correct implementation
//    of loadPacketStandard function.
//
// 2) Split the packet over two rows (within the same column) in eigen based
//    cuboid convolution and use partial loads for each individual part before
//    combining them to get the required packet. This class is used to pick the
//    correct implementation of loadPacketStandard function. This usage is
//    similar to the usage in eigen based spatial convolution described above.
//
// 3) Finalize packing of columns in `gemm_pack_colmajor` after processing
//    vectorized part with full packets (see spatial_convolution_data_mapper.h).
template <typename TensorEvaluatorType, typename PacketType, typename IndexType>
class TensorEvaluatorHasPartialPacket {
 public:
  template <typename TensorEvaluatorT, typename PacketT, typename IndexT>
  static auto functionExistsSfinae(
      typename std::enable_if<
          unpacket_traits<PacketT>::masked_load_available &&
          std::is_same<PacketT,
                       decltype(std::declval<const TensorEvaluatorT>()
                                    .template partialPacket<PacketT>(
                                        std::declval<IndexT>(),
                                        std::declval<typename unpacket_traits<
                                            PacketT>::mask_t>()))>::value>::
          type*) -> std::true_type;

  template <typename TensorEvaluatorT, typename PacketT, typename IndexT>
  static auto functionExistsSfinae(...) -> std::false_type;

  using status =
      decltype(functionExistsSfinae<TensorEvaluatorType, PacketType, IndexType>(
          nullptr));

  static constexpr bool value = status::value;
};

// Compute a mask for loading/storing coefficients in/from a packet in a
// [from, to) range. If the mask bit is 1, element will be loaded/stored.
template <typename Packet>
EIGEN_ALWAYS_INLINE
    typename std::enable_if<unpacket_traits<Packet>::masked_load_available,
                            typename unpacket_traits<Packet>::mask_t>::type
    mask(int from, int to) {
  const Index packet_size = internal::unpacket_traits<Packet>::size;
  assert(0 <= from && to <= (packet_size + 1) && from < to);

  using Mask = typename internal::unpacket_traits<Packet>::mask_t;
  const Mask mask_max = std::numeric_limits<Mask>::max();

  return (mask_max >> (packet_size - to)) ^ (mask_max >> (packet_size - from));
}

}  // namespace internal
}  // namespace Eigen

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_PARTIAL_PACKETS_H_
