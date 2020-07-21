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

//===- spatial_convolution_data_mapper.h ------------------------*- C++ -*-===//
//
// Provide optimized Eigen contraction data mappers for extracting image patches
// from the underlying tensor expression. We do this by "pattern matching" on a
// spatial convolution tensor expression AST (defined in spatial_convolution.h)
// via Eigen template specialization.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_DATA_MAPPER_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_DATA_MAPPER_H_

#include "partial_packets.h"

#if defined(TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL)
#include "contraction_kernel.h"
#endif  // TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL

namespace Eigen {
namespace internal {

// WARNING: Most of the code here implicitly assumes that the matrix is in
// ColMajor layout. This is guaranteed by the tensor contraction (see
// Eigen TensorContraction.h).
//
// Inside Eigen a tensor contraction is represented by a matrix multiplication.
// We don't want to actually extract image patches and reshape the result into
// a matrix (this involves allocating huge extra memory), so the patch
// extraction and reshape operations are implicit.
//
// TensorContractionInputMapper takes a matrix index and returns the coefficient
// (or the packet) of the "virtual tensor", that would be at that index if we
// were to actually reshape the result of patch extraction.
//
// TensorContractionSubMapper provides a similar view into the "virtual matrix"
// at the given vertical and horizontal offsets.
//
// "Virtual matrix" dimensions:
//   *0: kernelChannels * kernelRows * kernelCols;
//    1: out_height * out_width; * OTHERS (e.g batches, etc...)
//
// *) Extracted patches are continuous in memory (innermost dimension assuming
//    col major layout).
//
// With this dimensions:
//   row - offset within a single patch (in code: patchId)
//   col - index of the extracted patch (in code: patchIndex)
//         patchIndex ∈ [0..num_patches * OTHERS] (batch and other dimensions)

template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename ScalarType,
          typename IndexType, typename NoContractDims, typename ContractDims,
          int side, int packet_size, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment>
class TensorContractionInputMapper<
    ScalarType, IndexType, side,
    TensorEvaluator<
        const TensorReshapingOp<PreContractDimensions,
                                const TensorImagePatchOp<rows, cols, ArgType>>,
        Device>,
    NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
  // Sanity checking that we are actually matching the correct spatial
  // convolution expression defined in `spatial_convolution.h`.
  static constexpr int kArgRank = 4;
  static constexpr int kReshapedRank = 2;

  static_assert(traits<ArgType>::Layout == RowMajor,
                "Eigen optimizer spatial convolution data mapper supports only "
                "RowMajor layout");
  static_assert(traits<ArgType>::NumDimensions == kArgRank,
                "Eigen optimizer spatial convolution data mapper supports only "
                "4 dimensional argument in NHWC data format");
  static_assert(array_size<PreContractDimensions>::value == kReshapedRank,
                "Eigen optimizer spatial convolution data mapper supports only "
                "2 dimensional pre-contraction dimensions");

 public:
  using Scalar = ScalarType;
  using Packet = typename packet_traits<Scalar>::type;

  static_assert(unpacket_traits<Packet>::size == packet_size,
                "Input mapper packet size does not match the real packet size");

  using TensorReshapeImagePatchEvaluator = TensorEvaluator<
      const TensorReshapingOp<PreContractDimensions,
                              const TensorImagePatchOp<rows, cols, ArgType>>,
      Device>;

  using SubMapper = TensorContractionSubMapper<
      Scalar, IndexType, side, TensorReshapeImagePatchEvaluator, NoContractDims,
      ContractDims, packet_size, inner_dim_contiguous, inner_dim_reordered,
      Alignment>;

  using VectorMapper = SubMapper;
  using LinearMapper = SubMapper;

  using ArgTypeTensorEvaluator = TensorEvaluator<ArgType, Device>;

  TensorContractionInputMapper(const TensorReshapeImagePatchEvaluator& tensor,
                               const NoContractDims&, const NoContractDims&,
                               const ContractDims&, const ContractDims&)
      : m_impl(tensor.impl().impl()) {
    // Number of dimensions after extracting image patches.
    const int image_patch_dims = tensor.impl().dimensions().size();
    assert(image_patch_dims == kArgRank + 1);

    Index patch_depth = tensor.impl().dimensions()[image_patch_dims - 1];
    Index patch_rows = tensor.impl().dimensions()[image_patch_dims - 2];
    m_patch_cols = tensor.impl().dimensions()[image_patch_dims - 3];
    m_num_patches = tensor.impl().dimensions()[image_patch_dims - 4];

    // Strides for navigating through the single patch.
    m_patch_row_stride = patch_depth;
    m_patch_col_stride = patch_rows * m_patch_row_stride;

    m_patch_row_inflate_strides = tensor.impl().rowInflateStride();
    m_patch_col_inflate_strides = tensor.impl().colInflateStride();

    m_colStride = patch_rows;

    m_outputRows = tensor.impl().outputRows();
    m_outputCols = tensor.impl().outputCols();
    m_row_strides = tensor.impl().userRowStride();
    m_col_strides = tensor.impl().userColStride();

    m_in_row_strides = tensor.impl().userInRowStride();
    m_in_col_strides = tensor.impl().userInColStride();

    // Number of dimensions in the input tensor prior to extracting patches.
    const int num_input_dims = tensor.impl().impl().dimensions().size();
    assert(num_input_dims == kArgRank);

    m_inputRows = tensor.impl().impl().dimensions()[num_input_dims - 2];
    m_inputCols = tensor.impl().impl().dimensions()[num_input_dims - 3];

    m_rowInputStride = patch_depth;
    m_colInputStride = patch_depth * m_inputRows;
    m_patchInputStride = patch_depth * m_inputRows * m_inputCols;

    m_rowPaddingTop = tensor.impl().rowPaddingTop();
    m_colPaddingLeft = tensor.impl().colPaddingLeft();

    m_fastPatchRowStride = TensorIntDivisor<Index>(m_patch_row_stride);
    m_fastPatchColStride = TensorIntDivisor<Index>(m_patch_col_stride);
    m_fastInputRowStride = TensorIntDivisor<Index>(m_patch_row_inflate_strides);
    m_fastInputColStride = TensorIntDivisor<Index>(m_patch_col_inflate_strides);
    m_fastNumPatches = TensorIntDivisor<Index>(m_num_patches);
    m_fastColStride = TensorIntDivisor<Index>(m_colStride);
    m_fastOutputRows = TensorIntDivisor<Index>(m_outputRows);
    m_fastDimZero = TensorIntDivisor<Index>(patch_depth);
  }

  TensorContractionInputMapper(const TensorContractionInputMapper& base_mapper)
      : m_impl(base_mapper.m_impl) {
    m_patch_cols = base_mapper.m_patch_cols;
    m_num_patches = base_mapper.m_num_patches;

    m_patch_row_stride = base_mapper.m_patch_row_stride;
    m_patch_col_stride = base_mapper.m_patch_col_stride;

    m_patch_row_inflate_strides = base_mapper.m_patch_row_inflate_strides;
    m_patch_col_inflate_strides = base_mapper.m_patch_col_inflate_strides;

    m_colStride = base_mapper.m_colStride;

    m_rowInputStride = base_mapper.m_rowInputStride;
    m_colInputStride = base_mapper.m_colInputStride;
    m_patchInputStride = base_mapper.m_patchInputStride;

    m_inputRows = base_mapper.m_inputRows;
    m_inputCols = base_mapper.m_inputCols;

    m_outputRows = base_mapper.m_outputRows;
    m_outputCols = base_mapper.m_outputCols;
    m_row_strides = base_mapper.m_row_strides;
    m_col_strides = base_mapper.m_col_strides;

    m_in_row_strides = base_mapper.m_in_row_strides;
    m_in_col_strides = base_mapper.m_in_col_strides;

    m_rowPaddingTop = base_mapper.m_rowPaddingTop;
    m_colPaddingLeft = base_mapper.m_colPaddingLeft;

    m_fastPatchRowStride = base_mapper.m_fastPatchRowStride;
    m_fastPatchColStride = base_mapper.m_fastPatchColStride;
    m_fastInputRowStride = base_mapper.m_fastInputRowStride;
    m_fastInputColStride = base_mapper.m_fastInputColStride;
    m_fastNumPatches = base_mapper.m_fastNumPatches;
    m_fastColStride = base_mapper.m_fastColStride;
    m_fastOutputRows = base_mapper.m_fastOutputRows;
    m_fastDimZero = base_mapper.m_fastDimZero;
  }

  // If true, turns off some optimizations for loading packets since the image
  // patches are "non-standard" such as there are non-trivial strides or
  // inflations in the input.
  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_in_row_strides != 1 || m_in_col_strides != 1 ||
           m_patch_row_inflate_strides != 1 || m_patch_col_inflate_strides != 1;
  }

  EIGEN_STRONG_INLINE SubMapper getSubMapper(Index i, Index j) const {
    return SubMapper(*this, i, j);
  }

  EIGEN_STRONG_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(*this, i, j);
  }

  // Load the coefficient at the patchIndex location instead of the usual
  // m_rowIndex, m_colIndex, m_otherIndex.
  EIGEN_STRONG_INLINE Scalar operator()(Index row, Index patchIndex) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadCoeff(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_ALWAYS_INLINE Packet loadPacket(Index row, Index patchIndex) const {
    Index rowIndex, colIndex, otherIndex;
    computeBaseIndices(patchIndex, rowIndex, colIndex, otherIndex);
    return loadPacket(row, rowIndex, colIndex, otherIndex);
  }

  EIGEN_ALWAYS_INLINE const TensorEvaluator<ArgType, Device>& impl() const {
    return m_impl;
  }

  EIGEN_ALWAYS_INLINE Index patchDepth() const { return m_rowInputStride; }

  EIGEN_ALWAYS_INLINE Index patchRows() const { return m_colStride; }

  EIGEN_ALWAYS_INLINE Index patchCols() const { return m_patch_cols; }

 private:
  friend class TensorContractionSubMapper<
      Scalar, IndexType, side, TensorReshapeImagePatchEvaluator, NoContractDims,
      ContractDims, packet_size, inner_dim_contiguous, inner_dim_reordered,
      Alignment>;

  // Load coefficient from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_STRONG_INLINE Scalar loadCoeff(Index patchId, Index rowIndex,
                                       Index colIndex, Index otherIndex) const {
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;

    const Index colOffset = patchOffset / m_fastColStride;
    const Index inputCol = colIndex + colOffset * m_in_col_strides;
    const Index origInputCol =
        (m_patch_col_inflate_strides == 1)
            ? inputCol
            : ((inputCol >= 0) ? (inputCol / m_fastInputColStride) : 0);

    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputRow = rowIndex + rowOffset * m_in_row_strides;
    const Index origInputRow =
        (m_patch_row_inflate_strides == 1)
            ? inputRow
            : ((inputRow >= 0) ? (inputRow / m_fastInputRowStride) : 0);
    if (origInputCol < 0 || origInputRow < 0 || origInputCol >= m_inputCols ||
        origInputRow >= m_inputRows ||
        (inputCol != origInputCol * m_patch_col_inflate_strides) ||
        (inputRow != origInputRow * m_patch_row_inflate_strides)) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + origInputRow * m_rowInputStride +
                             origInputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  // This is the same as loadCoeff(...), but optimized for all `inflate_strides`
  // and `in_strides` equal to 1 (template specialization without templates).
  EIGEN_STRONG_INLINE Scalar loadCoeffStandard(Index patchId, Index rowIndex,
                                               Index colIndex,
                                               Index otherIndex) const {
    assert(!nonStandardPatches());

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputCol >= m_inputCols || inputRow < 0 ||
        inputRow >= m_inputRows) {
      return Scalar(0);
    }
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.coeff(inputIndex);
  }

  // Load packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index patchId, Index rowIndex,
                                        Index colIndex,
                                        Index otherIndex) const {
    static_assert(packet_size > 1, "Packet size must be > 1");
    assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    if (nonStandardPatches()) {
      return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
    }
    return loadPacketStandard<Packet, ArgTypeTensorEvaluator>(
        patchId, rowIndex, colIndex, otherIndex);
  }

  // Helper function to load a 'partial' packet - this is the single column
  // part of a packet that is split across two columns. In the 'partial' packet,
  // the elements corresponding to the column (specified through colOffset) are
  // loaded and the rest of the elements are zero-filled into the 'partial'
  // packet. This function is called from loadPacketStandardFromTwoColumns().
  // This code path is exercised only when the packet type supports masked load
  // and when the partial packet load is available in the TensorEvaluator.
  EIGEN_ALWAYS_INLINE Packet loadPartialPacketStandard(
      Index rowIndex, Index colIndex, Index otherIndex, Index patchId,
      const Index span[], const Index patchOffsets[], Index colOffset) const {
    const Index inputCol = colIndex + colOffset;
    const Index rowOffsets[2] = {patchOffsets[0] - colOffset * m_colStride,
                                 patchOffsets[1] - colOffset * m_colStride};
    const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                rowIndex + rowOffsets[1]};

    if (inputRows[0] >= m_inputRows || inputRows[1] < 0 ||
        inputCol >= m_inputCols || inputCol < 0) {
      // Partial packet is all zeros.
      return internal::pset1<Packet>(Scalar(0));
    } else if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
      // From inputIndex-span[0], we need to load elements starting from index
      // span[0] all the way upto (and including) span[1].
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputRows[0] * m_rowInputStride +
                               inputCol * m_colInputStride + otherIndex;
      return m_impl.template partialPacket<Packet>(
          inputIndex - span[0], mask<Packet>(span[0], span[1] + 1));
    } else {
      // Using slow path for this partial packet. We need to load elements
      // starting from index span[0] all the way up to (and including) span[1].
      //
      // We split this load into 3 parts:
      //   [0         , span[0]-1    ] - fill with zeroes
      //   [span[0]   , span[1]      ] - load elements from values
      //   [span[1]+1 , packet_size-1] - fill with zeroes
      EIGEN_ALIGN_MAX
      typename internal::remove_const<Scalar>::type values[packet_size];
      for (Index i = 0; i < span[0]; ++i) values[i] = Scalar(0);
      for (Index i = span[0]; i < span[1] + 1; ++i)
        values[i] =
            loadCoeff(patchId - span[0] + i, rowIndex, colIndex, otherIndex);
      for (Index i = span[1] + 1; i < packet_size; ++i) values[i] = Scalar(0);
      return internal::pload<Packet>(values);
    }
  }

  // Helper function to load a packet that is split across two columns.
  // If required, this function is called from loadPacketStandard() when the
  // packet type supports masked load and when the partial packet load is
  // available in the TensorEvaluator.
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromTwoColumns(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex,
      const Index patchOffsets[], const Index colOffsets[]) const {
    assert(colOffsets[1] == colOffsets[0] + 1);

    // Packet to load will be split into 2 parts where each part spans a single
    // column. First determine where to split.
    const Index patchIdSplit =
        ((colOffsets[1] * m_colStride) * m_rowInputStride) - 1;
    const Index patchOffsetSplit = patchIdSplit / m_fastDimZero;

    // patchIds[i]:          patchId corresponding to partial packet i
    // spans[i]:             Start and end indices corresponding to the elements
    //                       to be loaded for partial packet i
    // patchOffsets2Cols[i]: patchOffsets corresponding to partial packet i
    const Index patchIds[2] = {patchId, patchIdSplit + 1};
    const Index spans[2][2] = {{0, patchIdSplit - patchId},
                               {patchIdSplit - patchId + 1, packet_size - 1}};
    const Index patchOffsets2Cols[2][2] = {
        {patchOffsets[0], patchOffsetSplit},
        {patchOffsetSplit + 1, patchOffsets[1]}};

    // Load partial packets and do bit-wise OR to generate required packet.
    return internal::por<Packet>(
        loadPartialPacketStandard(rowIndex, colIndex, otherIndex, patchIds[0],
                                  spans[0], patchOffsets2Cols[0],
                                  colOffsets[0]),
        loadPartialPacketStandard(rowIndex, colIndex, otherIndex, patchIds[1],
                                  spans[1], patchOffsets2Cols[1],
                                  colOffsets[1]));
  }

  // Helper function to load a packet that is present in a single column.
  // If required, this function is called from loadPacketStandard().
  EIGEN_ALWAYS_INLINE Packet loadPacketStandardFromSingleColumn(
      Index patchId, Index rowIndex, Index colIndex, Index otherIndex,
      const Index patchOffsets[], const Index colOffsets[],
      const Index inputCols[]) const {
    assert(colOffsets[0] == colOffsets[1]);
    const Index rowOffsets[2] = {patchOffsets[0] - colOffsets[0] * m_colStride,
                                 patchOffsets[1] - colOffsets[1] * m_colStride};
    assert(rowOffsets[0] <= rowOffsets[1]);
    const Index inputRows[2] = {rowIndex + rowOffsets[0],
                                rowIndex + rowOffsets[1]};

    if (inputRows[0] >= m_inputRows || inputRows[1] < 0) {
      // All zeros.
      return internal::pset1<Packet>(Scalar(0));
    }

    if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
      // No padding.
      const Index depth = patchId - patchOffsets[0] * patchDepth();
      const Index inputIndex = depth + inputRows[0] * m_rowInputStride +
                               inputCols[0] * m_colInputStride + otherIndex;
      return m_impl.template packet<Unaligned>(inputIndex);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is not available for
  // the TensorEvaluator or if the packet type does not support masked load.
  template <typename ArgTypeTensorEvaluator>
  EIGEN_ALWAYS_INLINE
      typename std::enable_if<!TensorEvaluatorHasPartialPacket<
                                  ArgTypeTensorEvaluator, Packet, Index>::value,
                              Packet>::type
      loadPacketStandard(Index patchId, Index rowIndex, Index colIndex,
                         Index otherIndex) const {
    static_assert(packet_size > 1, "Packet size must be > 1");

    assert(patchId < patchDepth() * patchRows() * m_patch_cols);
    assert(!nonStandardPatches());

    if ((patchDepth() % packet_size) == 0) {
      return loadPacketFast(patchId, rowIndex, colIndex, otherIndex);
    }

    // Offsets and input calculation here are identical to
    // loadCoeffStandard(...), but repeated twice.
    const Index patchOffsets[2] = {patchId / m_fastDimZero,
                                   (patchId + packet_size - 1) / m_fastDimZero};
    const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                 patchOffsets[1] / m_fastColStride};
    const Index inputCols[2] = {colIndex + colOffsets[0],
                                colIndex + colOffsets[1]};

    if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
      // All zeros.
      return internal::pset1<Packet>(Scalar(0));
    }
    if (inputCols[0] == inputCols[1]) {
      return loadPacketStandardFromSingleColumn(patchId, rowIndex, colIndex,
                                                otherIndex, patchOffsets,
                                                colOffsets, inputCols);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  // Load standard packet from a patch specified by the "within patch offset"
  // (patchId) and the precomputed indices of the first element of the patch.
  // This function will be called if partial packet loading is available for
  // the TensorEvaluator and if the packet type supports masked load.
  // The only difference between this and the other case is that if the packet
  // to load is split across two columns, then in this case instead of going to
  // the slow (element-by-element) load, we load two packets - each containing
  // elements from one of the columns (rest of the elements of the packets are
  // zeroes), and then combine these two packets to generate the required
  // packet. The idea is to enable fast load (if possible) of these 'partial'
  // packets.
  template <typename ArgTypeTensorEvaluator>
  EIGEN_ALWAYS_INLINE
      typename std::enable_if<TensorEvaluatorHasPartialPacket<
                                  ArgTypeTensorEvaluator, Packet, Index>::value,
                              Packet>::type
      loadPacketStandard(Index patchId, Index rowIndex, Index colIndex,
                         Index otherIndex) const {
    static_assert(packet_size > 1, "Packet size must be > 1");

    assert(patchId < patchDepth() * patchRows() * m_patch_cols);
    assert(!nonStandardPatches());

    if ((patchDepth() % packet_size) == 0) {
      return loadPacketFast(patchId, rowIndex, colIndex, otherIndex);
    }

    // Offsets and input calculation here are identical to
    // loadCoeffStandard(...), but repeated twice.
    const Index patchOffsets[2] = {patchId / m_fastDimZero,
                                   (patchId + packet_size - 1) / m_fastDimZero};
    const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride,
                                 patchOffsets[1] / m_fastColStride};
    const Index inputCols[2] = {colIndex + colOffsets[0],
                                colIndex + colOffsets[1]};

    if (inputCols[0] >= m_inputCols || inputCols[1] < 0) {
      // All zeros.
      return internal::pset1<Packet>(Scalar(0));
    }
    if (inputCols[0] == inputCols[1]) {
      return loadPacketStandardFromSingleColumn(patchId, rowIndex, colIndex,
                                                otherIndex, patchOffsets,
                                                colOffsets, inputCols);
    }
    if (inputCols[1] == inputCols[0] + 1) {
      return loadPacketStandardFromTwoColumns(
          patchId, rowIndex, colIndex, otherIndex, patchOffsets, colOffsets);
    }
    return packetWithPossibleZero(patchId, rowIndex, colIndex, otherIndex);
  }

  EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index patchId, Index rowIndex,
                                            Index colIndex,
                                            Index otherIndex) const {
    static_assert(packet_size > 1, "Packet size must be > 1");
    assert(patchId < patchDepth() * patchRows() * m_patch_cols);

    assert(!nonStandardPatches());
    assert((patchDepth() % packet_size) == 0);
    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = patchId / m_fastDimZero;
    assert((patchId + packet_size - 1) / m_fastDimZero == patchOffset);

    const Index colOffset = patchOffset / m_fastColStride;
    const Index rowOffset = patchOffset - colOffset * m_colStride;
    const Index inputCol = colIndex + colOffset;
    const Index inputRow = rowIndex + rowOffset;
    if (inputCol < 0 || inputRow < 0 || inputCol >= m_inputCols ||
        inputRow >= m_inputRows) {
      // All zeros.
      return internal::pset1<Packet>(Scalar(0));
    }
    // No padding.
    const Index depth = patchId - patchOffset * patchDepth();
    const Index inputIndex = depth + inputRow * m_rowInputStride +
                             inputCol * m_colInputStride + otherIndex;
    return m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_ALWAYS_INLINE Packet packetWithPossibleZero(Index patchId,
                                                    Index rowIndex,
                                                    Index colIndex,
                                                    Index otherIndex) const {
    EIGEN_ALIGN_MAX
    typename internal::remove_const<Scalar>::type values[packet_size];
    for (int i = 0; i < packet_size; ++i) {
      values[i] = loadCoeff(patchId + i, rowIndex, colIndex, otherIndex);
    }
    Packet rslt = internal::pload<Packet>(values);
    return rslt;
  }

  EIGEN_STRONG_INLINE void computeBaseIndices(Index patchIndex, Index& rowIndex,
                                              Index& colIndex,
                                              Index& otherIndex) const {
    static constexpr size_t kNumInputDims =
        array_size<typename ArgTypeTensorEvaluator::Dimensions>::value;

    otherIndex = (kNumInputDims == 3) ? 0 : patchIndex / m_fastNumPatches;
    const Index patch2DIndex = (kNumInputDims == 3)
                                   ? patchIndex
                                   : (patchIndex - otherIndex * m_num_patches);
    otherIndex *= m_patchInputStride;
    colIndex = patch2DIndex / m_fastOutputRows;
    rowIndex = patch2DIndex - colIndex * m_outputRows;
    colIndex = colIndex * m_col_strides - m_colPaddingLeft;
    rowIndex = rowIndex * m_row_strides - m_rowPaddingTop;
  }

  Index m_patch_cols;   // Number of columns in the patch.  // NOLINT
  Index m_num_patches;  // Number of patches to extract.  // NOLINT

  // Strides for navigating through the single patch.
  Index m_patch_row_stride;                      // NOLINT
  Index m_patch_col_stride;                      // NOLINT
  TensorIntDivisor<Index> m_fastPatchRowStride;  // NOLINT
  TensorIntDivisor<Index> m_fastPatchColStride;  // NOLINT

  // The strides for row inflation in the image patch.
  Index m_patch_row_inflate_strides;  // NOLINT
  // The strides for col inflation in the image patch.
  Index m_patch_col_inflate_strides;  // NOLINT
  // Fast representation of inflation strides.
  TensorIntDivisor<Index> m_fastInputRowStride;  // NOLINT
  TensorIntDivisor<Index> m_fastInputColStride;  // NOLINT

  Index m_colStride;
  TensorIntDivisor<Index> m_fastNumPatches;  // NOLINT
  TensorIntDivisor<Index> m_fastColStride;   // NOLINT

  Index m_rowInputStride;    // Row stride in the input tensor.  // NOLINT
  Index m_colInputStride;    // Col stride in the input tensor.  // NOLINT
  Index m_patchInputStride;  // Patch stride in the input tensor.  // NOLINT

  Index m_inputRows;  // Number of rows in the input tensor.  // NOLINT
  Index m_inputCols;  // Number of cols in the input tensor.  // NOLINT

  Index m_outputRows;  // Number of convolution output rows.  // NOLINT
  Index m_outputCols;  // Number of convolution output column.  // NOLINT

  Index m_row_strides;  // User specified row stride.  // NOLINT
  Index m_col_strides;  // User specified col stride.  // NOLINT

  Index m_in_row_strides;  // User specified input row stride.  // NOLINT
  Index m_in_col_strides;  // User specified input col stride.  // NOLINT

  Index m_rowPaddingTop;   // Row padding.  // NOLINT
  Index m_colPaddingLeft;  // Column padding.  // NOLINT

  TensorIntDivisor<Index> m_fastOutputRows;  // NOLINT
  TensorIntDivisor<Index> m_fastDimZero;     // NOLINT

  const TensorEvaluator<ArgType, Device> m_impl;  // NOLINT
};

template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename Scalar,
          typename IndexType, typename NoContractDims, typename ContractDims,
          int side, int packet_size, bool inner_dim_contiguous,
          bool inner_dim_reordered, int Alignment>
class TensorContractionSubMapper<
    Scalar, IndexType, side,
    TensorEvaluator<
        const TensorReshapingOp<PreContractDimensions,
                                const TensorImagePatchOp<rows, cols, ArgType>>,
        Device>,
    NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
    inner_dim_reordered, Alignment> {
 public:
  using Packet = typename packet_traits<Scalar>::type;
  using HalfPacket = typename packet_traits<Scalar>::half;

  using TensorReshapeImagePatchEvaluator = TensorEvaluator<
      const TensorReshapingOp<PreContractDimensions,
                              const TensorImagePatchOp<rows, cols, ArgType>>,
      Device>;

  using ParentMapper = TensorContractionInputMapper<
      Scalar, IndexType, side, TensorReshapeImagePatchEvaluator, NoContractDims,
      ContractDims, packet_size, inner_dim_contiguous, inner_dim_reordered,
      Alignment>;

  using LinearMapper = TensorContractionSubMapper;
  using ArgTypeTensorEvaluator = typename ParentMapper::ArgTypeTensorEvaluator;

  EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const ParentMapper& base_mapper, Index vert_offset, Index horiz_offset)
      : m_depth_offset(vert_offset),
        m_col_offset(horiz_offset),
        m_base_mapper(base_mapper) {
    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_STRONG_INLINE TensorContractionSubMapper(
      const TensorContractionSubMapper& base_mapper, Index vert_offset,
      Index horiz_offset)
      : m_depth_offset(vert_offset + base_mapper.m_depth_offset),
        m_col_offset(horiz_offset + base_mapper.m_col_offset),
        m_base_mapper(base_mapper.m_base_mapper) {
    m_base_mapper.computeBaseIndices(m_col_offset, m_rowIndex, m_colIndex,
                                     m_otherIndex);
  }
  EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_base_mapper.loadCoeff(i + m_depth_offset, m_rowIndex, m_colIndex,
                                   m_otherIndex);
  }
  template <typename PacketT>
  EIGEN_ALWAYS_INLINE Packet loadPacket(Index i) const {
    return m_base_mapper.loadPacket(i + m_depth_offset, m_rowIndex, m_colIndex,
                                    m_otherIndex);
  }
  EIGEN_ALWAYS_INLINE Scalar loadCoeffStandard(Index i) const {
    return m_base_mapper.loadCoeffStandard(i + m_depth_offset, m_rowIndex,
                                           m_colIndex, m_otherIndex);
  }

  EIGEN_ALWAYS_INLINE Packet loadPacketFast(Index i) const {
    return m_base_mapper.loadPacketFast(i + m_depth_offset, m_rowIndex,
                                        m_colIndex, m_otherIndex);
  }
  EIGEN_ALWAYS_INLINE Packet loadPacketStandard(Index i) const {
    return m_base_mapper.template loadPacketStandard<ArgTypeTensorEvaluator>(
        i + m_depth_offset, m_rowIndex, m_colIndex, m_otherIndex);
  }

  EIGEN_ALWAYS_INLINE bool nonStandardPatches() const {
    return m_base_mapper.nonStandardPatches();
  }

  EIGEN_ALWAYS_INLINE bool hasPadding() const {
    // TODO(ezhulenev): It seems that for inflated filter it's still possible to
    // guarantee "no padding or skipping" for non-standard packing.
    if (nonStandardPatches()) return true;

    // Non zero padding before.
    if (m_base_mapper.m_rowPaddingTop > 0) return true;
    if (m_base_mapper.m_colPaddingLeft > 0) return true;

    // Non zero padding after in rows.
    const Index last_row =
        (m_base_mapper.m_outputRows - 1) * m_base_mapper.m_row_strides;
    if (last_row + (patchRows() - 1) >= m_base_mapper.m_inputRows) return true;

    // Non zero padding after in cols.
    const Index last_col =
        (m_base_mapper.m_outputCols - 1) * m_base_mapper.m_col_strides;
    if (last_col + (patchCols() - 1) >= m_base_mapper.m_inputCols) return true;

    return false;
  }

  // Max(Col|Row|Depth): compute the upper limit for the column, row and depth
  // index respectively that fits into the peeled_k elements starting at
  // m_depth_offset.
  EIGEN_ALWAYS_INLINE Index maxCol(const Index peeled_k) const {
    const Index max_col =
        (m_depth_offset + (peeled_k == 0 ? 0 : peeled_k - 1)) /
        fastPatchColStride();
    return std::min<Index>(1 + max_col, patchCols());
  }

  EIGEN_ALWAYS_INLINE Index maxRow(const Index peeled_k,
                                   const Index col) const {
    const Index max_row = (m_depth_offset + (peeled_k == 0 ? 0 : peeled_k - 1) -
                           col * patchColStride()) /
                          fastPatchRowStride();
    return std::min<Index>(1 + max_row, patchRows());
  }

  EIGEN_ALWAYS_INLINE Index maxDepth(const Index peeled_k, const Index col,
                                     Index row) const {
    // clang-format off
    const Index max_depth = m_depth_offset + peeled_k -
                            col * patchColStride() -
                            row * patchRowStride();
    // clang-format on
    return std::min<Index>(max_depth, patchDepth());
  }

  // MaxDepth uses only the remaining number of elements in the peeled_k.
  EIGEN_ALWAYS_INLINE Index maxDepth(const Index num_elements,
                                     const Index start_depth) const {
    return std::min<Index>(start_depth + num_elements, patchDepth());
  }

  // Every register matters in this code, so sometimes to prevent register
  // spilling, instead of the variable that you would expect to see, we use
  // another one, that is guaranteed to have the same value. E.g. patch depth is
  // always the same as input depth, and it's also the same as input row stride.
  // Bunch of other parameters have similar relations.
  EIGEN_ALWAYS_INLINE Index patchDepth() const {
    return m_base_mapper.m_rowInputStride;
  }

  EIGEN_ALWAYS_INLINE Index patchRows() const {
    return m_base_mapper.m_colStride;
  }

  EIGEN_ALWAYS_INLINE Index patchCols() const {
    return m_base_mapper.m_patch_cols;
  }

  EIGEN_ALWAYS_INLINE Index patchRowStride() const {
    assert(patchDepth() == m_base_mapper.m_patch_row_stride &&
           "Patch depth must be equal to patch row stride.");
    return patchDepth();
  }

  EIGEN_ALWAYS_INLINE Index patchColStride() const {
    return m_base_mapper.m_patch_col_stride;
  }

  EIGEN_ALWAYS_INLINE TensorIntDivisor<Index> fastPatchRowStride() const {
    assert(patchDepth() == m_base_mapper.m_patch_row_stride &&
           "Patch depth must be equal to patch row stride.");
    return m_base_mapper.m_fastDimZero;  // patch_depth
  }

  EIGEN_ALWAYS_INLINE TensorIntDivisor<Index> fastPatchColStride() const {
    return m_base_mapper.m_fastPatchColStride;
  }

  EIGEN_ALWAYS_INLINE Packet packetNoPadding(const Index depth,
                                             const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template packet<Unaligned>(inputIndex);
  }

  EIGEN_ALWAYS_INLINE Scalar coeffNoPadding(const Index depth,
                                            const Index baseIndex) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.coeff(inputIndex);
  }
  template <typename PacketT = Packet>
  EIGEN_ALWAYS_INLINE typename std::enable_if<
      TensorEvaluatorHasPartialPacket<ArgTypeTensorEvaluator, PacketT,
                                      Index>::value,
      PacketT>::type
  partialPacketNoPadding(const Index depth, const Index baseIndex,
                         Index num_coeffs) const {
    const Index inputIndex = depth + baseIndex;
    return m_base_mapper.m_impl.template partialPacket<PacketT>(
        inputIndex, mask<PacketT>(0, num_coeffs));
  }

  EIGEN_ALWAYS_INLINE bool padRow(const Index row) const {
    const Index r = m_rowIndex + row;
    return r < 0 || r >= m_base_mapper.m_inputRows;
  }

  EIGEN_ALWAYS_INLINE bool padAnyRow(const Index first_row,
                                     const Index last_row) const {
    return m_rowIndex + first_row < 0 ||
           m_rowIndex + last_row >= m_base_mapper.m_inputRows;
  }

  EIGEN_ALWAYS_INLINE bool padOrSkipRow(const Index row,
                                        Index* orig_row) const {
    assert(nonStandardPatches());

    const Index input_row = m_rowIndex + row * m_base_mapper.m_in_row_strides;
    *orig_row = (m_base_mapper.m_patch_row_inflate_strides == 1)
                    ? input_row
                    : ((input_row >= 0)
                           ? (input_row / m_base_mapper.m_fastInputRowStride)
                           : 0);

    return (*orig_row < 0 || *orig_row >= m_base_mapper.m_inputRows) ||
           (input_row != *orig_row * m_base_mapper.m_patch_row_inflate_strides);
  }

  EIGEN_ALWAYS_INLINE bool padCol(const Index col) const {
    const Index c = m_colIndex + col;
    return c < 0 || c >= m_base_mapper.m_inputCols;
  }

  EIGEN_ALWAYS_INLINE bool padOrSkipCol(const Index col,
                                        Index* orig_col) const {
    assert(nonStandardPatches());

    const Index input_col = m_colIndex + col * m_base_mapper.m_in_col_strides;
    *orig_col = (m_base_mapper.m_patch_col_inflate_strides == 1)
                    ? input_col
                    : ((input_col >= 0)
                           ? (input_col / m_base_mapper.m_fastInputColStride)
                           : 0);

    return (*orig_col < 0 || *orig_col >= m_base_mapper.m_inputCols) ||
           (input_col != *orig_col * m_base_mapper.m_patch_col_inflate_strides);
  }

  EIGEN_ALWAYS_INLINE Index baseIndex(const Index row, const Index col) const {
    const Index r = m_rowIndex + row;
    const Index c = m_colIndex + col;
    return r * m_base_mapper.m_rowInputStride +
           c * m_base_mapper.m_colInputStride + m_otherIndex;
  }

  // Compute a base index when original input row and column were precomputed
  // using padOrSkipRow and padOrSkipCol. Used only for non standard patches.
  EIGEN_ALWAYS_INLINE Index origBaseIndex(const Index orig_row,
                                          const Index orig_col) const {
    return orig_row * m_base_mapper.m_rowInputStride +
           orig_col * m_base_mapper.m_colInputStride + m_otherIndex;
  }

  EIGEN_ALWAYS_INLINE Index rowStride() const {
    return m_base_mapper.m_row_strides;
  }

  EIGEN_ALWAYS_INLINE Index colStride() const {
    return m_base_mapper.m_col_strides;
  }

  EIGEN_ALWAYS_INLINE Index rowOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return patchOffset - colOffset * m_base_mapper.m_colStride;
  }

  EIGEN_ALWAYS_INLINE Index colOffset() const {
    const Index patchOffset = m_depth_offset / m_base_mapper.m_fastDimZero;
    const Index colOffset = patchOffset / m_base_mapper.m_fastColStride;
    return colOffset;
  }

  EIGEN_ALWAYS_INLINE Index depthOffset() const {
    return m_depth_offset % patchDepth();
  }

  EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(m_base_mapper, i + m_depth_offset, j + m_col_offset);
  }

 private:
  Index m_depth_offset;  // First row in the input matrix.  // NOLINT
  Index m_col_offset;    // First col in the input matrix.  // NOLINT

  // Knowing that: col_offset == patchIndex * OTHERS, we keep precomputed base
  // indices for the first element in a patch specified by col_offset
  // (see computeBaseIndices(...) for details).
  Index m_rowIndex;    // NOLINT
  Index m_colIndex;    // NOLINT
  Index m_otherIndex;  // NOLINT

  // Keeping a copy instead of a reference performs better in benchmarks.
  const ParentMapper m_base_mapper;  // NOLINT
};

// Arrange a block of the right input matrix (in our case it's always a "virtual
// matrix" constructed from extracted image patches) in contiguous memory.
//
// Given column major input (A0 beside A1 in memory):
// A0 B0 C0 D0  E0 F0 G0 H0 ... Z0
// A1 B1 C1 D1  E1 F1 G1 H1 ... Z1
// A2 B2 C2 D2  E2 F2 G2 H2 ... Z2
// A3 B3 C3 D3  E3 F3 G3 H3 ... Z3
// A4 B4 C4 D4  E4 F4 G4 H4 ... Z4
// A5 B5 C5 D5  E5 F5 G5 H5 ... Z5
// A6 B6 C6 D6  E6 F6 G6 H6 ... Z6
// A7 B7 C7 D7  E7 F7 G7 H7 ... Z7
// A8 ...
// ...
//
// *) A, B, C, ... - patches extracted from the original input.
// *) A0, A1, A2 ... - values from the same patch at different offsets.
//
// The traversal (packed rhs memory) order (B0 besides A0 in memory):
// A0 B0 C0 D0 A1 B1 C1 D1 ...
// E0 F0 G0 H0 E1 F1 G1 H1 ...
// ...
// Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 ... <- doesn't belong to any block (nr = 4)
//
// This traversal order must be the same as in default gemm_pack_rhs defined in
// GeneralBlockPanelKernel.h.
//
// *) nr - number of registers along the 'n' dimension.
//    See GeneralBlockPanelKernel.h and "Anatomy of High-Performance Matrix
//    Multiplication" paper.
template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename Scalar,
          typename IndexType, typename NoContractDims, typename ContractDims,
          int packet_size, bool inner_dim_contiguous, bool inner_dim_reordered,
          int Alignment, int nr>
struct gemm_pack_rhs<
    Scalar, IndexType,
    TensorContractionSubMapper<
        Scalar, IndexType, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            PreContractDimensions,
                            const TensorImagePatchOp<rows, cols, ArgType>>,
                        Device>,
        NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  using SubMapper = TensorContractionSubMapper<
      Scalar, IndexType, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          PreContractDimensions,
                          const TensorImagePatchOp<rows, cols, ArgType>>,
                      Device>,
      NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;

  using DataMapper = SubMapper;
  using Packet = typename packet_traits<Scalar>::type;

  static_assert(nr == 4, "nr must be equal to 4");

  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index num_cols,
                                    Index stride = 0, Index offset = 0) const {
    assert(stride == 0);
    assert(offset == 0);

    const Index packet_cols4 = (num_cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if ((packet_size % 4) == 0 && !non_standard_patches) {
        // FAST PATH: Iterate over patch columns and rows, if we know that a
        // single packet does not span across multiple rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            // Check if we can squeeze reads along the `row` and `depth`
            // dimensions (two innermost dimensions).
            if (!pad_col0 && !pad_col1 && !pad_col2 && !pad_col3 &&
                !dm0.padRow(start_row) && !dm0.padRow(max_row - 1) &&
                !dm1.padRow(start_row) && !dm1.padRow(max_row - 1) &&
                !dm2.padRow(start_row) && !dm2.padRow(max_row - 1) &&
                !dm3.padRow(start_row) && !dm3.padRow(max_row - 1)) {
              // Compute how many elements we can squeeze read.
              const Index start_depth =
                  (c == start_col) ? rhs.depthOffset() : 0;

              // Upper bound for the number of elements in the depth dimension
              // that we can squeeze read.
              const Index squeeze_length =
                  (max_row - start_row) * rhs.patchDepth() - start_depth;

              // Do not overshoot beyond the block size.
              const Index max_depth =
                  start_depth + std::min<Index>(peeled_k - k, squeeze_length);
              assert((max_depth - start_depth) % packet_size == 0);

              const Index idx0 = dm0.baseIndex(start_row, c);
              const Index idx1 = dm1.baseIndex(start_row, c);
              const Index idx2 = dm2.baseIndex(start_row, c);
              const Index idx3 = dm3.baseIndex(start_row, c);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                assert(k < peeled_k);
                PacketBlock<Packet, 4> kernel;
                kernel.packet[0] = rhs.packetNoPadding(d, idx0);
                kernel.packet[1] = rhs.packetNoPadding(d, idx1);
                kernel.packet[2] = rhs.packetNoPadding(d, idx2);
                kernel.packet[3] = rhs.packetNoPadding(d, idx3);
                ptranspose(kernel);
                pstoreu(block + 0 * packet_size, kernel.packet[0]);
                pstoreu(block + 1 * packet_size, kernel.packet[1]);
                pstoreu(block + 2 * packet_size, kernel.packet[2]);
                pstoreu(block + 3 * packet_size, kernel.packet[3]);
                block += 4 * packet_size;
                k += packet_size;
              }

              // Go to the next column.
              continue;
            }

            // If we can't squeeze reads, process rows one by one.
            for (Index r = start_row; r < max_row; ++r) {
              assert(k <= peeled_k);

              const bool pad0 = pad_col0 || dm0.padRow(r);
              const bool pad1 = pad_col1 || dm1.padRow(r);
              const bool pad2 = pad_col2 || dm2.padRow(r);
              const bool pad3 = pad_col3 || dm3.padRow(r);

              const Index idx0 = dm0.baseIndex(r, c);
              const Index idx1 = dm1.baseIndex(r, c);
              const Index idx2 = dm2.baseIndex(r, c);
              const Index idx3 = dm3.baseIndex(r, c);

              const Index start_depth = ((c == start_col) && (r == start_row))
                                            ? rhs.depthOffset()
                                            : 0;
              const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
              assert((max_depth - start_depth) % packet_size == 0);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                assert(k < peeled_k);
                PacketBlock<Packet, 4> kernel;
                kernel.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx0);
                kernel.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx1);
                kernel.packet[2] = pad2 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx2);
                kernel.packet[3] = pad3 ? pset1<Packet>(Scalar(0))
                                        : rhs.packetNoPadding(d, idx3);
                ptranspose(kernel);
                pstoreu(block + 0 * packet_size, kernel.packet[0]);
                pstoreu(block + 1 * packet_size, kernel.packet[1]);
                pstoreu(block + 2 * packet_size, kernel.packet[2]);
                pstoreu(block + 3 * packet_size, kernel.packet[3]);
                block += 4 * packet_size;
                k += packet_size;
              }
            }
          }

          // The loop above should fill peeled_k elements.
          assert(peeled_k == k);

        } else {
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 4> kernel;
            kernel.packet[0] = dm0.loadPacketStandard(k);
            kernel.packet[1] = dm1.loadPacketStandard(k);
            kernel.packet[2] = dm2.loadPacketStandard(k);
            kernel.packet[3] = dm3.loadPacketStandard(k);
            ptranspose(kernel);
            pstoreu(block + 0 * packet_size, kernel.packet[0]);
            pstoreu(block + 1 * packet_size, kernel.packet[1]);
            pstoreu(block + 2 * packet_size, kernel.packet[2]);
            pstoreu(block + 3 * packet_size, kernel.packet[3]);
            block += 4 * packet_size;
          }
        }
      }

      // Copy the remaining coefficients of the column block after the peeled_k.
      if (!rhs.nonStandardPatches()) {
        for (; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < num_cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Template specialization for packet_size = 2. We must special-case packet
// blocks with nr > packet_size, e.g. PacketBlock<Packet2d, 4>.
template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename Scalar,
          typename IndexType, typename NoContractDims, typename ContractDims,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment,
          int nr>
struct gemm_pack_rhs<
    Scalar, IndexType,
    TensorContractionSubMapper<
        Scalar, IndexType, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            PreContractDimensions,
                            const TensorImagePatchOp<rows, cols, ArgType>>,
                        Device>,
        NoContractDims, ContractDims, 2, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  using SubMapper = TensorContractionSubMapper<
      Scalar, IndexType, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          PreContractDimensions,
                          const TensorImagePatchOp<rows, cols, ArgType>>,
                      Device>,
      NoContractDims, ContractDims, 2, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;
  using DataMapper = SubMapper;
  using Packet = typename packet_traits<Scalar>::type;

  static_assert(nr == 4, "nr must be equal to 4");

  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index num_cols,
                                    Index stride = 0, Index offset = 0) const {
    assert(stride == 0);
    assert(offset == 0);

    const int packet_size = 2;
    const Index packet_cols4 = (num_cols / 4) * 4;
    const Index peeled_k = (depth / packet_size) * packet_size;
    const bool non_standard_patches = rhs.nonStandardPatches();

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      Index k = 0;
      if (!non_standard_patches) {
        // FAST PATH: Iterate over patch columns and rows if we know that a
        // single packet does not span across multiple rows or columns.
        if ((rhs.patchDepth() % packet_size) == 0) {
          const Index start_col = rhs.colOffset();
          const Index max_col = rhs.maxCol(peeled_k);

          for (Index c = start_col; c < max_col; ++c) {
            assert(k <= peeled_k);

            const Index start_row = (c == start_col) ? rhs.rowOffset() : 0;
            const Index max_row = rhs.maxRow(peeled_k, c);

            const bool pad_col0 = dm0.padCol(c);
            const bool pad_col1 = dm1.padCol(c);
            const bool pad_col2 = dm2.padCol(c);
            const bool pad_col3 = dm3.padCol(c);

            // We can squeeze reads along the `row` and `depth` dimensions if
            // the row stride is `1`, which means that `row` and `depth`
            // dimensions are contiguous (two innermost dimensions).
            // clang-format off
            if (rhs.rowStride() == 1 &&
                !pad_col0 && !pad_col1 && !pad_col2 && !pad_col3 &&
                !dm0.padRow(start_row) && !dm0.padRow(max_row - 1) &&
                !dm1.padRow(start_row) && !dm1.padRow(max_row - 1) &&
                !dm2.padRow(start_row) && !dm2.padRow(max_row - 1) &&
                !dm3.padRow(start_row) && !dm3.padRow(max_row - 1)) {
              // clang-format on
              // Compute how many elements we can squeeze read.
              const Index start_depth =
                  (c == start_col) ? rhs.depthOffset() : 0;

              // Upper bound for the number of elements in the depth dimension
              // that we can squeeze read.
              const Index squeeze_length =
                  (max_row - start_row) * rhs.patchDepth() - start_depth;

              // Do not overshoot beyond the block size.
              const Index max_depth =
                  start_depth + std::min<Index>(peeled_k - k, squeeze_length);
              assert((max_depth - start_depth) % packet_size == 0);

              const Index idx0 = dm0.baseIndex(start_row, c);
              const Index idx1 = dm1.baseIndex(start_row, c);
              const Index idx2 = dm2.baseIndex(start_row, c);
              const Index idx3 = dm3.baseIndex(start_row, c);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                PacketBlock<Packet, 2> kernel0;
                PacketBlock<Packet, 2> kernel1;
                kernel0.packet[0] = rhs.packetNoPadding(d, idx0);
                kernel0.packet[1] = rhs.packetNoPadding(d, idx1);
                kernel1.packet[0] = rhs.packetNoPadding(d, idx2);
                kernel1.packet[1] = rhs.packetNoPadding(d, idx3);
                ptranspose(kernel0);
                ptranspose(kernel1);
                pstoreu(block + 0 * packet_size, kernel0.packet[0]);
                pstoreu(block + 1 * packet_size, kernel1.packet[0]);
                pstoreu(block + 2 * packet_size, kernel0.packet[1]);
                pstoreu(block + 3 * packet_size, kernel1.packet[1]);
                block += 4 * packet_size;
                k += packet_size;
              }

              // Go to the next column.
              continue;
            }

            // If we can't squeeze reads, process rows one by one.
            for (Index r = start_row; r < max_row; ++r) {
              assert(k <= peeled_k);

              const bool pad0 = pad_col0 || dm0.padRow(r);
              const bool pad1 = pad_col1 || dm1.padRow(r);
              const bool pad2 = pad_col2 || dm2.padRow(r);
              const bool pad3 = pad_col3 || dm3.padRow(r);

              const Index idx0 = dm0.baseIndex(r, c);
              const Index idx1 = dm1.baseIndex(r, c);
              const Index idx2 = dm2.baseIndex(r, c);
              const Index idx3 = dm3.baseIndex(r, c);

              const Index start_depth = ((c == start_col) && (r == start_row))
                                            ? rhs.depthOffset()
                                            : 0;
              const Index max_depth = rhs.maxDepth(peeled_k - k, start_depth);
              assert((max_depth - start_depth) % packet_size == 0);

              for (Index d = start_depth; d < max_depth; d += packet_size) {
                assert(k < peeled_k);
                PacketBlock<Packet, 2> kernel0;
                PacketBlock<Packet, 2> kernel1;
                kernel0.packet[0] = pad0 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx0);
                kernel0.packet[1] = pad1 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx1);
                kernel1.packet[0] = pad2 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx2);
                kernel1.packet[1] = pad3 ? pset1<Packet>(Scalar(0))
                                         : rhs.packetNoPadding(d, idx3);
                ptranspose(kernel0);
                ptranspose(kernel1);
                pstoreu(block + 0 * packet_size, kernel0.packet[0]);
                pstoreu(block + 1 * packet_size, kernel1.packet[0]);
                pstoreu(block + 2 * packet_size, kernel0.packet[1]);
                pstoreu(block + 3 * packet_size, kernel1.packet[1]);
                block += 4 * packet_size;
                k += packet_size;
              }
            }
          }

          // The loop above should fill peeled_k elements.
          assert(peeled_k == k);

        } else {
          // Packet can span multiple rows or columns, so we have to go
          // though the slower "standard" path.
          for (; k < peeled_k; k += packet_size) {
            PacketBlock<Packet, 2> kernel0;
            PacketBlock<Packet, 2> kernel1;
            kernel0.packet[0] = dm0.loadPacketStandard(k);
            kernel0.packet[1] = dm1.loadPacketStandard(k);
            kernel1.packet[0] = dm2.loadPacketStandard(k);
            kernel1.packet[1] = dm3.loadPacketStandard(k);
            ptranspose(kernel0);
            ptranspose(kernel1);
            pstoreu(block + 0 * packet_size, kernel0.packet[0]);
            pstoreu(block + 1 * packet_size, kernel1.packet[0]);
            pstoreu(block + 2 * packet_size, kernel0.packet[1]);
            pstoreu(block + 3 * packet_size, kernel1.packet[1]);
            block += 4 * packet_size;
          }
        }
      }

      // Copy the remaining coefficients of the column block after the peeled_k.
      if (!non_standard_patches) {
        for (; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < num_cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

// Special case for non-vectorized types such as float16.
template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename Scalar,
          typename IndexType, typename NoContractDims, typename ContractDims,
          bool inner_dim_contiguous, bool inner_dim_reordered, int Alignment,
          int nr>
struct gemm_pack_rhs<
    Scalar, IndexType,
    TensorContractionSubMapper<
        Scalar, IndexType, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            PreContractDimensions,
                            const TensorImagePatchOp<rows, cols, ArgType>>,
                        Device>,
        NoContractDims, ContractDims, 1, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    nr, ColMajor, false, false> {
  using SubMapper = TensorContractionSubMapper<
      Scalar, IndexType, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          PreContractDimensions,
                          const TensorImagePatchOp<rows, cols, ArgType>>,
                      Device>,
      NoContractDims, ContractDims, 1, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;
  using DataMapper = SubMapper;

  static_assert(nr == 4, "nr must be equal to 4");

  EIGEN_DONT_INLINE void operator()(Scalar* block, const DataMapper& rhs,
                                    Index depth, Index num_cols,
                                    Index stride = 0, Index offset = 0) const {
    assert(stride == 0);
    assert(offset == 0);

    const Index packet_cols4 = (num_cols / 4) * 4;

    for (Index j2 = 0; j2 < packet_cols4; j2 += 4) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2 + 0);
      const SubMapper dm1 = rhs.getLinearMapper(0, j2 + 1);
      const SubMapper dm2 = rhs.getLinearMapper(0, j2 + 2);
      const SubMapper dm3 = rhs.getLinearMapper(0, j2 + 3);

      if (!rhs.nonStandardPatches()) {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0.loadCoeffStandard(k);
          block[1] = dm1.loadCoeffStandard(k);
          block[2] = dm2.loadCoeffStandard(k);
          block[3] = dm3.loadCoeffStandard(k);
          block += 4;
        }
      } else {
        for (Index k = 0; k < depth; k++) {
          block[0] = dm0(k);
          block[1] = dm1(k);
          block[2] = dm2(k);
          block[3] = dm3(k);
          block += 4;
        }
      }
    }

    // Copy the remaining columns one at a time (nr==1).
    for (Index j2 = packet_cols4; j2 < num_cols; ++j2) {
      const SubMapper dm0 = rhs.getLinearMapper(0, j2);
      for (Index k = 0; k < depth; k++) {
        *block = dm0(k);
        block += 1;
      }
    }
  }
};

#if defined(TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL)

// After we vectorized all loads from the underlying tensor using Packet ops, we
// have to finalize coefficients that do not fit into a packet.
template <typename Scalar, typename DataMapper, int packet_size,
          bool masked_load_store>
struct FinalizeDataMapperCoeffs {
  EIGEN_ALWAYS_INLINE static Index finalize(Scalar* block,
                                            const DataMapper& rhs,
                                            Index base_idx, Index depth,
                                            Index max_depth, bool pad = false) {
    const Index num_coeffs = max_depth - depth;
    assert(num_coeffs <= packet_size);

    for (; depth < max_depth; ++depth) {
      *block = pad ? Scalar(0) : rhs.coeffNoPadding(depth, base_idx);
      ++block;
    }

    return num_coeffs;
  }
};

template <typename Scalar, typename DataMapper, int packet_size>
struct FinalizeDataMapperCoeffs<Scalar, DataMapper, packet_size,
                                /*masked_load_store=*/true> {
  EIGEN_ALWAYS_INLINE static Index finalize(Scalar* block,
                                            const DataMapper& rhs,
                                            Index base_idx, Index depth,
                                            Index max_depth, bool pad = false) {
    Index num_coeffs = max_depth - depth;
    assert(num_coeffs <= packet_size);
    if (num_coeffs == 0) return 0;

    using Packet = typename packet_traits<Scalar>::type;
    Packet p = pad ? pset1<Packet>(Scalar(0))
                   : rhs.partialPacketNoPadding(depth, base_idx, num_coeffs);
    internal::pstoreu(block, p, mask<Packet>(0, num_coeffs));

    return num_coeffs;
  }
};

// Pack a block of the right input matrix (in our case it's always a
// "virtual matrix" constructed from extracted image patches) in a contiguous
// block in column-major storage order. Knowing the properties of the
// original patch op we can do it more efficient than the default
// gemm_pack_colmajor_block.
template <typename PreContractDimensions, Index rows, Index cols,
          typename ArgType, typename Device, typename Scalar,
          typename IndexType, typename NoContractDims, typename ContractDims,
          int packet_size, bool inner_dim_contiguous, bool inner_dim_reordered,
          int Alignment>
struct gemm_pack_colmajor_block<
    Scalar, IndexType,
    TensorContractionSubMapper<
        Scalar, IndexType, Rhs,
        TensorEvaluator<const TensorReshapingOp<
                            PreContractDimensions,
                            const TensorImagePatchOp<rows, cols, ArgType>>,
                        Device>,
        NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
        inner_dim_reordered, Alignment>,
    ColMajor> {
  using SubMapper = TensorContractionSubMapper<
      Scalar, IndexType, Rhs,
      TensorEvaluator<const TensorReshapingOp<
                          PreContractDimensions,
                          const TensorImagePatchOp<rows, cols, ArgType>>,
                      Device>,
      NoContractDims, ContractDims, packet_size, inner_dim_contiguous,
      inner_dim_reordered, Alignment>;

  using DataMapper = SubMapper;
  using Packet = typename packet_traits<Scalar>::type;

  using CoeffFinalizer = FinalizeDataMapperCoeffs<
      Scalar, DataMapper, packet_size,
      TensorEvaluatorHasPartialPacket<
          typename DataMapper::ArgTypeTensorEvaluator, Packet, Index>::value &&
          unpacket_traits<Packet>::masked_store_available>;

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper rhs, IndexType num_rows,
                  IndexType num_cols) {
    const bool standard_patches = !rhs.nonStandardPatches();

    if (standard_patches && (rhs.patchDepth() % packet_size == 0)) {
      // Single packet always belong to single patch (row, col).
      if (rhs.hasPadding()) {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/true,
                            /*has_padding=*/true>(block, rhs, num_rows,
                                                  num_cols);
      } else {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/true,
                            /*has_padding=*/false>(block, rhs, num_rows,
                                                   num_cols);
      }

    } else if (standard_patches) {
      // Single packet can span across multiple patch rows or columns.
      if (rhs.hasPadding()) {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/false,
                            /*has_padding=*/true>(block, rhs, num_rows,
                                                  num_cols);
      } else {
        packStandardPatches</*patch_depth_is_multiple_of_packet_size=*/false,
                            /*has_padding=*/false>(block, rhs, num_rows,
                                                   num_cols);
      }

    } else if (rhs.patchDepth() % packet_size == 0) {
      // Single packet always belong to single patch (row, col).
      packNonStandardPatches</*patch_depth_is_multiple_of_packet_size*/
                             true>(block, rhs, num_rows, num_cols);

    } else {
      // Single packet can span across multiple patch rows or columns.
      packNonStandardPatches</*patch_depth_is_multiple_of_packet_size*/
                             false>(block, rhs, num_rows, num_cols);
    }
  }

 private:
  // (A) Standard image patches:
  //
  // (1) in_row_stride = 1 && in_col_stride == 1
  // (2) patch_row_inflate_strides == 1 && patch_col_inflate_strides == 1
  //
  // Standard patches guarantee that two inner most dimensions (depth and rows)
  // are contiguous in memory and we can try to squeeze reads from them.
  //
  // (B) Non standard image patches: in_row/in_col and patch_row/patch_col
  // strides can be not equal to 1, and for each [row, col] inside a patch we
  // have to do additional computations to find corresponding row and col in the
  // input tensor. Also we can no longer squeeze reads from inner dimensions.
  //
  // Additional parameters:
  // - patch_depth_is_multiple_of_packet_size=true: We are guaranteed to have
  //   depth dimension size to be a multiple of packet size, so we can skip all
  //   non vectorized loads and checks, because it's guaranteed that block size
  //   will be a multiple of a packet size (see TensorContractionBlocking).
  //
  // - has_padding: Input tensor has non-zero padding. In this case for each
  //   patch col and row we need to check that it doesn't correspond to the
  //   padded region of original input.
  template <bool patch_depth_is_multiple_of_packet_size, bool has_padding>
  EIGEN_ALWAYS_INLINE void packStandardPatches(Scalar* block,
                                               const DataMapper rhs,
                                               IndexType num_rows,
                                               IndexType num_cols) {
    assert(!rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const IndexType peeled_k = (num_rows / packet_size) * packet_size;

    const IndexType start_col = rhs.colOffset();
    const IndexType max_col = rhs.maxCol(peeled_k);
    const IndexType depth_offset = rhs.depthOffset();

    for (IndexType col = 0; col < num_cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      IndexType k = 0;
      for (Index c = start_col; c < max_col; ++c) {
        assert(k <= peeled_k);

        const IndexType start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const IndexType max_row = rhs.maxRow(peeled_k, c);
        const bool pad_col = has_padding && lm.padCol(c);

        eigen_assert(has_padding || !lm.padCol(c));
        eigen_assert(has_padding || !lm.padAnyRow(start_row, max_row - 1));

        // We can squeeze reads for all rows in [start_row, max_row) range.
        if (!has_padding ||
            (!pad_col && !lm.padAnyRow(start_row, max_row - 1))) {
          const IndexType start_depth = (c == start_col) ? depth_offset : 0;

          const IndexType max_depth =
              std::min<IndexType>(start_depth + (peeled_k - k),
                                  (max_row - start_row) * rhs.patchDepth());

          const IndexType base_idx = lm.baseIndex(start_row, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            assert((max_depth - start_depth) % packet_size == 0);
            IndexType d = start_depth;

            const IndexType unrolled_depth = max_depth - 4 * packet_size;
            for (; d <= unrolled_depth; d += 4 * packet_size) {
              assert(k < peeled_k);

              Packet p0 = rhs.packetNoPadding(d + 0 * packet_size, base_idx);
              Packet p1 = rhs.packetNoPadding(d + 1 * packet_size, base_idx);
              Packet p2 = rhs.packetNoPadding(d + 2 * packet_size, base_idx);
              Packet p3 = rhs.packetNoPadding(d + 3 * packet_size, base_idx);

              internal::pstoreu(block + 0 * packet_size, p0);
              internal::pstoreu(block + 1 * packet_size, p1);
              internal::pstoreu(block + 2 * packet_size, p2);
              internal::pstoreu(block + 3 * packet_size, p3);

              block += 4 * packet_size;
              k += 4 * packet_size;
            }

            for (; d < max_depth; d += packet_size) {
              assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }

          } else {
            IndexType d = start_depth;

            const IndexType unrolled_depth = max_depth - 4 * packet_size;
            for (; d <= unrolled_depth; d += 4 * packet_size) {
              eigen_assert(k < peeled_k);

              Packet p0 = rhs.packetNoPadding(d + 0 * packet_size, base_idx);
              Packet p1 = rhs.packetNoPadding(d + 1 * packet_size, base_idx);
              Packet p2 = rhs.packetNoPadding(d + 2 * packet_size, base_idx);
              Packet p3 = rhs.packetNoPadding(d + 3 * packet_size, base_idx);

              internal::pstoreu(block + 0 * packet_size, p0);
              internal::pstoreu(block + 1 * packet_size, p1);
              internal::pstoreu(block + 2 * packet_size, p2);
              internal::pstoreu(block + 3 * packet_size, p3);

              block += 4 * packet_size;
              k += 4 * packet_size;
            }

            const IndexType vectorized_depth = max_depth - packet_size;
            for (; d <= vectorized_depth; d += packet_size) {
              assert(k < peeled_k);
              internal::pstoreu(block, rhs.packetNoPadding(d, base_idx));
              block += packet_size;
              k += packet_size;
            }

            assert(k <= peeled_k);
            const Index num_coeffs =
                CoeffFinalizer::finalize(block, rhs, base_idx, d, max_depth);

            k += num_coeffs;
            block += num_coeffs;
            assert(k <= peeled_k);
          }

          // Go to the next column.
          continue;
        }

        // If we are not allowed to squeeze reads along the `row` and `depth`
        // dimensions, we must process rows one by one.
        for (IndexType r = start_row; r < max_row; ++r) {
          assert(k <= peeled_k);

          const IndexType start_depth =
              ((c == start_col) && (r == start_row)) ? depth_offset : 0;
          const IndexType max_depth = rhs.maxDepth(peeled_k - k, start_depth);

          const bool pad = has_padding && (pad_col || lm.padRow(r));
          eigen_assert(has_padding || !lm.padRow(r));

          const IndexType base_idx = lm.baseIndex(r, c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            assert((max_depth - start_depth) % packet_size == 0);
            IndexType d = start_depth;

            for (; d < max_depth; d += packet_size) {
              assert(k < peeled_k);
              const Packet p = (has_padding && pad)
                                   ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

          } else {
            IndexType d = start_depth;

            const IndexType vectorized_depth = max_depth - packet_size;
            for (; d <= vectorized_depth; d += packet_size) {
              assert(k < peeled_k);
              const Packet p = (has_padding && pad)
                                   ? pset1<Packet>(Scalar(0))
                                   : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

            assert(k <= peeled_k);
            const Index num_coeffs = CoeffFinalizer::finalize(
                block, rhs, base_idx, d, max_depth, has_padding && pad);

            k += num_coeffs;
            block += num_coeffs;
            assert(k <= peeled_k);
          }
        }
      }

      // The loop above should fill peeled_k elements.
      assert(peeled_k == k);

      // Fill remaining elements using loadCoeffStandard.
      for (; k < num_rows; ++k) {
        *block = lm.loadCoeffStandard(k);
        ++block;
      }
    }
  }

  template <bool patch_depth_is_multiple_of_packet_size>
  EIGEN_ALWAYS_INLINE void packNonStandardPatches(Scalar* block,
                                                  const DataMapper rhs,
                                                  IndexType num_rows,
                                                  IndexType num_cols) {
    assert(rhs.nonStandardPatches());

    // Give vectorized_rows the name used in all other gemm_pack_rhs above.
    const IndexType peeled_k = (num_rows / packet_size) * packet_size;

    const IndexType start_col = rhs.colOffset();
    const IndexType max_col = rhs.maxCol(peeled_k);
    const IndexType depth_offset = rhs.depthOffset();

    // Original input column and row after applying all non-standard strides and
    // dilations. Computed by padOrSkip{Row,Col}.
    IndexType orig_c;
    IndexType orig_r;

    for (IndexType col = 0; col < num_cols; ++col) {
      SubMapper lm = rhs.getLinearMapper(0, col);

      IndexType k = 0;
      for (IndexType c = start_col; c < max_col; ++c) {
        assert(k <= peeled_k);

        const IndexType start_row = (c == start_col) ? rhs.rowOffset() : 0;
        const IndexType max_row = rhs.maxRow(peeled_k, c);
        const bool pad_or_skip_col = lm.padOrSkipCol(c, &orig_c);

        for (IndexType r = start_row; r < max_row; ++r) {
          assert(k <= peeled_k);

          const IndexType start_depth =
              ((c == start_col) && (r == start_row)) ? depth_offset : 0;
          const IndexType max_depth = rhs.maxDepth(peeled_k - k, start_depth);

          const bool pad_or_skip =
              pad_or_skip_col || lm.padOrSkipRow(r, &orig_r);
          const IndexType base_idx = lm.origBaseIndex(orig_r, orig_c);

          if (patch_depth_is_multiple_of_packet_size) {
            // If patch depth is a multiple of packet size, it's guaranteed that
            // we can process all values in depth dimension with packets.
            assert((max_depth - start_depth) % packet_size == 0);
            IndexType d = start_depth;

            for (; d < max_depth; d += packet_size) {
              assert(k < peeled_k);
              const Packet p = pad_or_skip ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

          } else {
            const IndexType vectorized_depth = max_depth - packet_size;
            IndexType d = start_depth;
            for (; d <= vectorized_depth; d += packet_size) {
              assert(k < peeled_k);
              const Packet p = pad_or_skip ? pset1<Packet>(Scalar(0))
                                           : rhs.packetNoPadding(d, base_idx);
              internal::pstoreu(block, p);
              block += packet_size;
              k += packet_size;
            }

            assert(k <= peeled_k);
            const IndexType num_coeffs = CoeffFinalizer::finalize(
                block, rhs, base_idx, d, max_depth, pad_or_skip);

            k += num_coeffs;
            block += num_coeffs;
            assert(k <= peeled_k);
          }
        }
      }

      // The loop above should fill peeled_k elements.
      assert(peeled_k == k);

      // Fill remaining elements using loadCoeff.
      for (; k < num_rows; ++k) {
        *block = lm(k);
        ++block;
      }
    }
  }
};

#endif  // TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL

}  // namespace internal
}  // namespace Eigen

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_SPATIAL_CONVOLUTION_DATA_MAPPER_H_
