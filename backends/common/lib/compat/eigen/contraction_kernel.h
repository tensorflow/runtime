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

//===- contraction_kernel.h -------------------------------------*- C++ -*-===//
//
// Depending on a build configuration this header provides a custom kernel for
// Eigen tensor contractions (small matrix multiplication kernel used to
// multiply together blocks of the original tensors).
//
// 1) Default.
//    Use MKL-DNN single threaded sgemm for float, and Eigen for everything
//    else. The MKL-DNN kernels are generated at runtime and use
//    avx/avx2/fma/avx512 based on cpu status registers
//    (https://en.wikipedia.org/wiki/CPUID).
//
// 2) No MKL-DNN: --define disable_eigen_mkldnn_contraction_kernel=true.
//    This header will not define any custom contraction kernels, and Eigen will
//    use the default Eigen::internal::gebp_kernel.
//
// If you use `tensor.contract(other_tensor)` in your code, you must include
// this header to get the benefit of custom contraction kernel.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_KERNEL_H_
#define TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_KERNEL_H_

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#if defined(TFRT_EIGEN_USE_MKLDNN_CONTRACTION_KERNEL)
#include "mkldnn.h"  // from @mkl_dnn
#endif

#include "tfrt/support/msan.h"

namespace Eigen {
namespace internal {

#if defined(TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL)

// Returns `true` iff we can use custom contraction kernels. This is a runtime
// check that uses environment variables.
// TODO(b/152346987): Rename this back to UseCustomContractionKernels when TFRT
// is open sourced.
bool UseCustomContractionKernelsTFRT();

// Pack a 2D block of a Tensor expression into a contiguous block of memory with
// col-major storage order. We do not have access to the underlying Tensor
// expression, we only have a DataMapper (TensorContractionInputMapper for
// tensor contractions, or blas_data_mapper for plain tensors), that provides a
// two-dimensional view into the Tensor expression.
//
// Default Eigen gemm_pack_rhs and gemm_pack_lhs packs blocks of tensor
// expressions into the packed format described in "Anatomy of High-Performance
// Matrix Multiplication" paper (1). Eigen::internal::gebp_kernel relies on this
// packing format for efficient micro-panel multiplication.
//
// This simple packing can be used with any '?gemm' function from BLAS
// libraries, that work with col-major matrices.
//
// (1) http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf
//
// IMPORTANT: `gemm_pack_colmajor_block` always packs the block in column major
// order, `data_mapper_storage_order` specifies the storage order of the
// underlying Tensor expression.
template <typename Scalar, typename IndexType, typename DataMapper,
          int data_mapper_storage_order>
struct gemm_pack_colmajor_block;

// gemm_pack_colmajor_block for ColMajor storage order.
template <typename Scalar, typename IndexType, typename DataMapper>
struct gemm_pack_colmajor_block<Scalar, IndexType, DataMapper,
                                /*data_mapper_storage_order=*/ColMajor> {
  using Packet = typename internal::packet_traits<Scalar>::type;
  using LinearMapper = typename DataMapper::LinearMapper;

  enum { PacketSize = internal::packet_traits<Scalar>::size };

  EIGEN_DONT_INLINE
  void operator()(Scalar* block, const DataMapper& data_mapper, IndexType rows,
                  IndexType cols) {
    const IndexType unrolled_rows = rows - 4 * PacketSize;
    const IndexType vectorized_rows = rows - PacketSize;

    for (IndexType col = 0; col < cols; ++col) {
      LinearMapper lm = data_mapper.getLinearMapper(0, col);

      IndexType row = 0;
      // Give compiler a strong possibility to unroll the loop.
      for (; row <= unrolled_rows; row += 4 * PacketSize) {
        for (IndexType j = 0; j < 4; ++j) {
          const Packet p = lm.template loadPacket<Packet>(row + j * PacketSize);
          internal::pstoreu(block + j * PacketSize, p);
        }
        block += 4 * PacketSize;
      }
      // Process remaining rows with packets.
      for (; row <= vectorized_rows; row += PacketSize) {
        const Packet p = lm.template loadPacket<Packet>(row);
        internal::pstoreu(block, p);
        block += PacketSize;
      }
      // Finalize with coefficients.
      for (; row < rows; ++row) {
        *block = lm(row);
        ++block;
      }
    }
  }
};

#endif  // TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL

// Use MKL-DNN sgemm as an Eigen contraction kernel.
//
// To disable it build with:
//   "--define disable_eigen_mkldnn_contraction_kernel=true"
#if defined(TFRT_EIGEN_USE_MKLDNN_CONTRACTION_KERNEL)

template <typename Scalar, typename IndexType, typename OutputMapper,
          bool ConjugateLhs = false, bool ConjugateRhs = false>
struct mkldnn_gemm_kernel;

// mkldnn_gemm_kernel for floats defined as a thin layer on top of mkldnn_sgemm.
template <typename IndexType, typename OutputMapper, bool ConjugateLhs,
          bool ConjugateRhs>
struct mkldnn_gemm_kernel</*Scalar=*/float, IndexType, OutputMapper,
                          ConjugateLhs, ConjugateRhs> {
  static_assert(!ConjugateLhs, "MKL-DNN kernel doesn't support ConjugateLhs");
  static_assert(!ConjugateRhs, "MKL-DNN kernel doesn't support ConjugateRhs");

  static constexpr int kComputeStrideFromBlockDimensions = -1;

  using LhsScalar = float;
  using RhsScalar = float;
  using ResScalar = float;

  EIGEN_DONT_INLINE
  void operator()(const OutputMapper& output, const LhsScalar* blockA,
                  const RhsScalar* blockB, const IndexType rows,
                  const IndexType depth, const IndexType cols, float alpha,
                  float beta, int ldA = kComputeStrideFromBlockDimensions,
                  int ldB = kComputeStrideFromBlockDimensions,
                  char transposeA = 'N', char transposeB = 'N') {
    static const int max_index = (std::numeric_limits<int>::max)();

    assert(max_index >= rows);
    assert(max_index >= cols);
    assert(max_index >= depth);
    assert(max_index >= output.stride());

    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(cols);
    const int k = static_cast<int>(depth);

    ldA = ldA == kComputeStrideFromBlockDimensions ? m : ldA;
    ldB = ldB == kComputeStrideFromBlockDimensions ? k : ldB;
    const int ldC = static_cast<int>(output.stride());

    mkldnn_status_t st = mkldnn_sgemm(
        &transposeA, &transposeB, &m, &n, &k, &alpha, blockA, &ldA, blockB,
        &ldB, &beta, const_cast<ResScalar*>(output.data()), &ldC);
    assert(st == 0);

#if defined(MEMORY_SANITIZER)
    for (IndexType col = 0; col < cols; ++col) {
      ResScalar* row_base = &output(0, col);
      TFRT_MSAN_MEMORY_IS_INITIALIZED(row_base, sizeof(ResScalar) * rows);
    }
#endif

    // assert is a no-op in optimized mode so we add these to avoid
    // compiler's unused-variable errors.
    EIGEN_UNUSED_VARIABLE(max_index);
    EIGEN_UNUSED_VARIABLE(st);
  }
};

// For mkldnn_sgemm having the right dimensions (especially for small matrices)
// is more important than fitting all the working set in L1/L2 caches.
// TODO(ezhulenev): Develop better heuristics.
template <typename IndexType, int sharding_type>
class TensorContractionBlocking<float, float, float, IndexType, sharding_type> {
  // For now MKL-DNN has only mkldnn_sgemm (gemm for floats).
  using Scalar = float;

  // Adjust the block sizes to work well with MKL-DNN kernels.

  // Multiply default choice of block size along M and N dimensions.
  // TODO(ezhulenev): Explore if this can work in general (kScaleM=2.0 worked
  // well for some models).
  static constexpr float kScaleM = 1.5;
  static constexpr float kScaleN = 1.0;

  // MKL-DNN Avx/Avx2/Avx512 unroll factors are: 8/16/48.
  static constexpr IndexType kUnrollM = 48;

  // MKL-DNN Avx/Avx2/Avx512 unroll factors are: 6/6/8.
  static constexpr IndexType kUnrollN = 24;

 public:
  TensorContractionBlocking(IndexType k, IndexType m, IndexType n,
                            IndexType num_threads = 1)
      : kc_(k), mc_(m), nc_(n) {
    // 1. Compute block sizes using default Eigen heuristics.
    if (sharding_type == ShardByCol) {
      computeProductBlockingSizes<Scalar, Scalar, 1>(kc_, mc_, nc_,
                                                     num_threads);
    } else {
      computeProductBlockingSizes<Scalar, Scalar, 1>(kc_, nc_, mc_,
                                                     num_threads);
    }

    // If dimensions do not pass basic sanity checks, return immediately.
    if (kc_ <= 0 || mc_ <= 0 || nc_ <= 0) return;

    // If we are using default Eigen gebp kernel, there is no need to adjust the
    // block sizes for MKL-DNN.
    if (!UseCustomContractionKernelsTFRT()) return;

    // 2. And refine them to work well with mkldnn sgemm.
    mc_ = (std::min)(
        m, Eigen::divup(static_cast<IndexType>(mc_ * kScaleM), kUnrollM) *
               kUnrollM);
    nc_ = (std::min)(
        n, Eigen::divup(static_cast<IndexType>(nc_ * kScaleN), kUnrollN) *
               kUnrollN);

    // We split Kth dimensions in roughly equal slices.
    IndexType target_k_slices = (std::max)(IndexType(1), Eigen::divup(k, kc_));
    IndexType packet_size = internal::packet_traits<Scalar>::size;
    if (packet_size < 8) packet_size = 8;
    IndexType target_bk =
        Eigen::divup(k / target_k_slices, packet_size) * packet_size;
    kc_ = (std::min)(k, target_bk);
  }

  EIGEN_ALWAYS_INLINE IndexType kc() const { return kc_; }
  EIGEN_ALWAYS_INLINE IndexType mc() const { return mc_; }
  EIGEN_ALWAYS_INLINE IndexType nc() const { return nc_; }

 private:
  IndexType kc_;
  IndexType mc_;
  IndexType nc_;
};

// If the Lhs or Rhs Tensor expressions are already evaluated and have access to
// raw data, we can skip the packing step, and setup pointers and a stride to
// the underlying memory buffer and pass them directly to Gemm.
template <typename Scalar, typename IndexType>
struct ColMajorBlock {
  bool is_direct_access;

  // Valid iff `is_direct_access == false`.
  Scalar* packed_data;

  // Valid iff `is_direct_access == true`.
  Scalar* raw_data;
  IndexType stride;
  char transpose;
};

template <typename DataMapper>
struct DirectColMajorAccess {
  enum { value = false };

  template <typename Scalar, typename IndexType>
  static bool block(const typename DataMapper::SubMapper& data_mapper,
                    const IndexType rows, const IndexType cols,
                    const IndexType num_kernels,
                    ColMajorBlock<Scalar, IndexType>* block) {
    return false;
  }
};

// If we have an access to raw memory of the contraction input, we can safely
// skip packing if:
//
// (1) Packing is a no-op.
// (2) Packed block will be used just once.
// (3) Packed block will be used a small number of times, and accounting for the
//     strides it does not touch a large region of memory (otherwise it seems
//     that the cost of TLB misses will dominate).
//
// If a packed block is used many times, it's more efficient to pack it into
// contiguous block of memory to reduce TLB pressure.
//
// TODO(ezhulenev): Constants for (3) were picked from running benchmarks on
// a dedicated Skylake desktop, figure out if they are different for realistic
// workload in a shared environment.
//
// TODO(ezhulenev): Add support for more tensor expressions that matter.
#define REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_EXPR)                          \
  template <typename Scalar, typename IndexType, int Side, typename Device,    \
            typename nocontract_t, typename contract_t, int packet_size,       \
            int Alignment>                                                     \
  struct DirectColMajorAccess<TensorContractionInputMapper<                    \
      Scalar, IndexType, Side, TensorEvaluator<TENSOR_EXPR, Device>,           \
      nocontract_t, contract_t, packet_size, /*inner_dim_contiguous=*/true,    \
      /*inner_dim_reordered=*/false, Alignment>> {                             \
    enum { value = true };                                                     \
                                                                               \
    using DataMapper = TensorContractionInputMapper<                           \
        Scalar, IndexType, Side, TensorEvaluator<TENSOR_EXPR, Device>,         \
        nocontract_t, contract_t, packet_size, /*inner_dim_contiguous=*/true,  \
        /*inner_dim_reordered=*/false, Alignment>;                             \
                                                                               \
    static bool block(const typename DataMapper::SubMapper& data_mapper,       \
                      const IndexType rows, const IndexType cols,              \
                      const IndexType num_kernels,                             \
                      ColMajorBlock<Scalar, IndexType>* block) {               \
      /* Heuristics for case (3) above. */                                     \
      static constexpr IndexType kMaxBlockKernels = 2;                         \
      static constexpr IndexType kMaxAddressedMemory = 256 << 10; /* 256 Kb */ \
                                                                               \
      static_assert(DataMapper::DirectOffsets == true,                         \
                    "DataMapper must support direct offsets");                 \
                                                                               \
      const IndexType vert_offset = data_mapper.vert_offset();                 \
      const IndexType horiz_offset = data_mapper.horiz_offset();               \
      const IndexType stride =                                                 \
          Side == Lhs ? data_mapper.base_mapper().stride()                     \
                      : data_mapper.base_mapper().nocontract_strides()[0];     \
      const Scalar* data = data_mapper.base_mapper().tensor().data();          \
      data = Side == Lhs ? data : data + vert_offset + horiz_offset * stride;  \
                                                                               \
      const bool is_no_op_packing = stride == rows;                            \
      const IndexType addressed_mem = (stride * cols * sizeof(Scalar));        \
      const bool use_direct_access = is_no_op_packing ||                       \
                                     num_kernels == 1 /* used once */ ||       \
                                     ((num_kernels <= kMaxBlockKernels) &&     \
                                      (addressed_mem < kMaxAddressedMemory));  \
                                                                               \
      if (use_direct_access) {                                                 \
        block->is_direct_access = true;                                        \
        block->raw_data = const_cast<Scalar*>(data);                           \
        block->stride = stride;                                                \
        block->transpose = 'N';                                                \
        return true;                                                           \
      }                                                                        \
      return false;                                                            \
    }                                                                          \
  }

#define SIMPLE_TENSOR const Tensor<Scalar, 2, Eigen::ColMajor, IndexType>

#define TENSOR_MAP_ROWMAJOR \
  const TensorMap<Tensor<Scalar, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>

#define TENSOR_MAP_COLMAJOR \
  const TensorMap<Tensor<Scalar, 2, Eigen::ColMajor, IndexType>, Eigen::Aligned>

#define TENSOR_MAP_CONST_ROWMAJOR                                      \
  const TensorMap<const Tensor<Scalar, 2, Eigen::RowMajor, IndexType>, \
                  Eigen::Aligned>

#define TENSOR_MAP_CONST_COLMAJOR                                      \
  const TensorMap<const Tensor<Scalar, 2, Eigen::ColMajor, IndexType>, \
                  Eigen::Aligned>

// This is the reshaped convolution filter from `spatial_convolution.h`.
#define TENSOR_RESHAPE                                                     \
  const TensorReshapingOp<                                                 \
      const Eigen::DSizes<IndexType, 2>,                                   \
      const TensorMap<const Tensor<Scalar, 4, Eigen::RowMajor, IndexType>, \
                      Eigen::Aligned>>

REGISTER_DIRECT_COL_MAJOR_ACCESS(SIMPLE_TENSOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_ROWMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_COLMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_CONST_ROWMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_MAP_CONST_COLMAJOR);
REGISTER_DIRECT_COL_MAJOR_ACCESS(TENSOR_RESHAPE);

#undef SIMPLE_TENSOR
#undef TENSOR_MAP_ROWMAJOR
#undef TENSOR_MAP_COLMAJOR
#undef TENSOR_MAP_CONST_ROWMAJOR
#undef TENSOR_MAP_CONST_COLMAJOR
#undef TENSOR_RESHAPE
#undef REGISTER_DIRECT_COL_MAJOR_ACCESS

template <typename ResScalar, typename LhsScalar, typename RhsScalar,
          typename IndexType, typename OutputMapper>
struct GemmKernelProvider {
  enum { Defined = 0 };
  using GemmKernel = void;
};

template <typename IndexType, typename OutputMapper>
struct GemmKernelProvider<float, float, float, IndexType, OutputMapper> {
  enum { Defined = 1 };
  using GemmKernel = mkldnn_gemm_kernel<float, IndexType, OutputMapper>;
};

// Tensor contraction kernel that can fallback on Eigen gebp_kernel at runtime.
//
// For some data types we can't fallback on Eigen, e.g. Eigen does not have
// packing and kernel implementations for quantized types.
#define REGISTER_TENSOR_CONTRACTION_KERNEL_WITH_FALLBACK(                      \
    RES_SCALAR, LHS_SCALAR, RHS_SCALAR)                                        \
                                                                               \
  template <typename IndexType, typename OutputMapper, typename LhsMapper,     \
            typename RhsMapper>                                                \
  struct TensorContractionKernel<RES_SCALAR, LHS_SCALAR, RHS_SCALAR,           \
                                 IndexType, OutputMapper, LhsMapper,           \
                                 RhsMapper> {                                  \
    TensorContractionKernel(IndexType m, IndexType k, IndexType n,             \
                            IndexType bm, IndexType bk, IndexType bn)          \
        : m(m),                                                                \
          k(k),                                                                \
          n(n),                                                                \
          bm(bm),                                                              \
          bk(bk),                                                              \
          bn(bn),                                                              \
          nm0(bm > 0 ? divup(m, bm) : 0),                                      \
          nn0(bn > 0 ? divup(n, bn) : 0) {}                                    \
                                                                               \
    enum { HasBeta = true };                                                   \
                                                                               \
    using ResScalar = RES_SCALAR;                                              \
    using LhsScalar = LHS_SCALAR;                                              \
    using RhsScalar = RHS_SCALAR;                                              \
                                                                               \
    using Traits = typename internal::gebp_traits<LhsScalar, RhsScalar>;       \
                                                                               \
    using LhsBlock = ColMajorBlock<LhsScalar, IndexType>;                      \
    using RhsBlock = ColMajorBlock<RhsScalar, IndexType>;                      \
                                                                               \
    using DirectLhsAccess = DirectColMajorAccess<LhsMapper>;                   \
    using DirectRhsAccess = DirectColMajorAccess<RhsMapper>;                   \
                                                                               \
    /* Packed Lhs/Rhs block memory allocator. */                               \
    using BlockMemAllocator =                                                  \
        TensorContractionBlockMemAllocator<LhsScalar, RhsScalar>;              \
    using BlockMemHandle = typename BlockMemAllocator::BlockMemHandle;         \
                                                                               \
    using LhsPacker =                                                          \
        gemm_pack_colmajor_block<LhsScalar, IndexType,                         \
                                 typename LhsMapper::SubMapper, ColMajor>;     \
    using RhsPacker =                                                          \
        gemm_pack_colmajor_block<RhsScalar, IndexType,                         \
                                 typename RhsMapper::SubMapper, ColMajor>;     \
                                                                               \
    using GemmKernelProviderType =                                             \
        GemmKernelProvider<ResScalar, LhsScalar, RhsScalar, IndexType,         \
                           OutputMapper>;                                      \
    static_assert(                                                             \
        GemmKernelProviderType::Defined,                                       \
        "Custom GEMM kernel is not registered for given scalar types");        \
    using GemmKernel = typename GemmKernelProviderType::GemmKernel;            \
                                                                               \
    /* Fallback on default Eigen pack and GEBP kernel if custom contraction */ \
    /* kernels disabled at runtime.                                         */ \
    using EigenLhsPacker =                                                     \
        gemm_pack_lhs<LhsScalar, IndexType, typename LhsMapper::SubMapper,     \
                      Traits::mr, Traits::LhsProgress,                         \
                      typename Traits::LhsPacket4Packing, ColMajor>;           \
    using EigenRhsPacker =                                                     \
        gemm_pack_rhs<RhsScalar, IndexType, typename RhsMapper::SubMapper,     \
                      Traits::nr, ColMajor>;                                   \
    using GebpKernel =                                                         \
        gebp_kernel<LhsScalar, RhsScalar, IndexType, OutputMapper, Traits::mr, \
                    Traits::nr, /*ConjugateLhs=*/false,                        \
                    /*ConjugateRhs=*/false>;                                   \
                                                                               \
    template <typename Device>                                                 \
    BlockMemHandle allocate(Device& d, LhsBlock* lhs_block,                    \
                            RhsBlock* rhs_block) {                             \
      return BlockMemAllocator::allocate(                                      \
          d, bm, bk, bn, &lhs_block->packed_data, &rhs_block->packed_data);    \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    BlockMemHandle allocateSlices(Device& d, const int num_lhs,                \
                                  const int num_rhs, const int num_slices,     \
                                  std::vector<LhsBlock>* lhs_blocks,           \
                                  std::vector<RhsBlock>* rhs_blocks) {         \
      assert(num_slices > 0);                                                  \
      std::vector<std::vector<LhsScalar*>> lhs_mem(num_slices);                \
      std::vector<std::vector<RhsScalar*>> rhs_mem(num_slices);                \
                                                                               \
      BlockMemHandle block_mem = BlockMemAllocator::allocateSlices(            \
          d, bm, bk, bn, num_lhs, num_rhs, num_slices, lhs_mem.data(),         \
          rhs_mem.data());                                                     \
                                                                               \
      for (Index x = 0; x < num_slices; x++) {                                 \
        if (num_lhs > 0) lhs_blocks[x].resize(num_lhs);                        \
        for (Index m = 0; m < num_lhs; m++) {                                  \
          lhs_blocks[x][m].packed_data = lhs_mem[x][m];                        \
        }                                                                      \
        if (num_rhs > 0) rhs_blocks[x].resize(num_rhs);                        \
        for (Index n = 0; n < num_rhs; n++) {                                  \
          rhs_blocks[x][n].packed_data = rhs_mem[x][n];                        \
        }                                                                      \
      }                                                                        \
                                                                               \
      return block_mem;                                                        \
    }                                                                          \
                                                                               \
    template <typename Device>                                                 \
    static void deallocate(Device& d, BlockMemHandle handle) {                 \
      BlockMemAllocator::deallocate(d, handle);                                \
    }                                                                          \
                                                                               \
    EIGEN_DONT_INLINE void packLhs(                                            \
        LhsBlock* lhsBlock, const typename LhsMapper::SubMapper& data_mapper,  \
        const IndexType depth, const IndexType rows) {                         \
      if (UseCustomContractionKernelsTFRT()) {                                 \
        const bool is_direct_access =                                          \
            DirectLhsAccess::value &&                                          \
            DirectLhsAccess::block(data_mapper, rows, depth, nn0, lhsBlock);   \
                                                                               \
        if (!is_direct_access) {                                               \
          lhsBlock->is_direct_access = false;                                  \
          LhsPacker()(lhsBlock->packed_data, data_mapper, rows, depth);        \
        }                                                                      \
      } else {                                                                 \
        lhsBlock->is_direct_access = false;                                    \
        EigenLhsPacker()(lhsBlock->packed_data, data_mapper, depth, rows,      \
                         /*stride=*/0, /*offset=*/0);                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DONT_INLINE void packRhs(                                            \
        RhsBlock* rhsBlock, const typename RhsMapper::SubMapper& data_mapper,  \
        const IndexType depth, const IndexType cols) {                         \
      if (UseCustomContractionKernelsTFRT()) {                                 \
        const bool is_direct_access =                                          \
            DirectRhsAccess::value &&                                          \
            DirectRhsAccess::block(data_mapper, depth, cols, nm0, rhsBlock);   \
                                                                               \
        if (!is_direct_access) {                                               \
          rhsBlock->is_direct_access = false;                                  \
          RhsPacker()(rhsBlock->packed_data, data_mapper, depth, cols);        \
        }                                                                      \
      } else {                                                                 \
        rhsBlock->is_direct_access = false;                                    \
        EigenRhsPacker()(rhsBlock->packed_data, data_mapper, depth, cols);     \
      }                                                                        \
    }                                                                          \
                                                                               \
    EIGEN_DONT_INLINE void invoke(const OutputMapper& output_mapper,           \
                                  const LhsBlock& lhsBlock,                    \
                                  const RhsBlock& rhsBlock,                    \
                                  const IndexType rows, const IndexType depth, \
                                  const IndexType cols, const float alpha,     \
                                  const float beta) {                          \
      if (UseCustomContractionKernelsTFRT()) {                                 \
        if ((DirectLhsAccess::value && lhsBlock.is_direct_access) &&           \
            (DirectRhsAccess::value && rhsBlock.is_direct_access)) {           \
          GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.raw_data,    \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/lhsBlock.stride, /*ldB=*/rhsBlock.stride,       \
                       /*transposeA=*/lhsBlock.transpose,                      \
                       /*transposeB=*/rhsBlock.transpose);                     \
                                                                               \
        } else if (DirectLhsAccess::value && lhsBlock.is_direct_access) {      \
          GemmKernel()(output_mapper, lhsBlock.raw_data, rhsBlock.packed_data, \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/lhsBlock.stride,                                \
                       /*ldB=*/GemmKernel::kComputeStrideFromBlockDimensions,  \
                       /*transposeA=*/lhsBlock.transpose, /*transposeB=*/'N'); \
                                                                               \
        } else if (DirectRhsAccess::value && rhsBlock.is_direct_access) {      \
          GemmKernel()(output_mapper, lhsBlock.packed_data, rhsBlock.raw_data, \
                       rows, depth, cols, alpha, beta,                         \
                       /*ldA=*/GemmKernel::kComputeStrideFromBlockDimensions,  \
                       /*ldB=*/rhsBlock.stride, /*transposeA=*/'N',            \
                       /*transposeB=*/rhsBlock.transpose);                     \
                                                                               \
        } else {                                                               \
          GemmKernel()(output_mapper, lhsBlock.packed_data,                    \
                       rhsBlock.packed_data, rows, depth, cols, alpha, beta);  \
        }                                                                      \
      } else {                                                                 \
        /* Gebp kernel does not support beta, so we have to clear memory in */ \
        /* the output mapper manually.                                      */ \
        /* WARNING(ezhulenev): This is optimized into a memset in a loop,   */ \
        /* could be much slower for small matrices. Currently this code     */ \
        /* path used only for testing, and performance does not matter.     */ \
        if (beta == 0.0) {                                                     \
          for (IndexType col = 0; col < cols; ++col) {                         \
            ResScalar* output_base = &output_mapper(0, col);                   \
            using OutputRow = Array<ResScalar, Dynamic, 1>;                    \
            using OutputRowMap = Map<OutputRow, 0, InnerStride<1>>;            \
            OutputRowMap(output_base, rows).setZero();                         \
          }                                                                    \
        }                                                                      \
                                                                               \
        GebpKernel()(                                                          \
            output_mapper, lhsBlock.packed_data, rhsBlock.packed_data, rows,   \
            depth, cols, alpha,                                                \
            /*strideA=*/GemmKernel::kComputeStrideFromBlockDimensions,         \
            /*strideB=*/GemmKernel::kComputeStrideFromBlockDimensions,         \
            /*offsetA=*/0, /*offsetB=*/0);                                     \
      }                                                                        \
    }                                                                          \
                                                                               \
   private:                                                                    \
    /* These are dimensions of the original Tensors, and selected block     */ \
    /* sizes. The actual block sizes passed to all function above might be  */ \
    /* smaller because of the partial blocks at the end.                    */ \
    const IndexType m;                                                         \
    const IndexType k;                                                         \
    const IndexType n;                                                         \
    const IndexType bm;                                                        \
    const IndexType bk;                                                        \
    const IndexType bn;                                                        \
                                                                               \
    /* Number of kernels for each dimension. */                                \
    const IndexType nm0;                                                       \
    const IndexType nn0;                                                       \
  }

REGISTER_TENSOR_CONTRACTION_KERNEL_WITH_FALLBACK(float, float, float);

#undef REGISTER_TENSOR_CONTRACTION_KERNEL_WITH_FALLBACK

#endif  // defined(TFRT_EIGEN_USE_MKLDNN_CONTRACTION_KERNEL)

}  // namespace internal
}  // namespace Eigen

#endif  // TFRT_BACKENDS_COMMON_LIB_COMPAT_EIGEN_CONTRACTION_KERNEL_H_
