/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// StreamAnalysis: Given an MLIR module in TFRT dialects and a cost_threshold as
// a module attribute, it produces stream assignment for each operations. The
// runtime can use this stream assignment as a hint for more efficient execution
// (eg. successive operations in the same stream can be executed inline).
//
// Cost Threshold: The cost threshold is used to decide whether to merge two
// streams. When there are independent streams, if the cost of a stream is
// smaller than this threshold, then it would be worth merging this stream with
// others, while if the cost is larger or equal to the threshold, then it is not
// worth doing so. Note that dependent streams can still be merged regardless of
// the cost. It is set through the module attribute `tfrt.cost_threshold`.
//
// The algorithm can be summarized as follows:
//
// 1. Build a naive stream tree where each stream contains only one operation:
//    1.1 Traverse the DAG in a topological order.
//    1.2 Create a stream for the current op.
//    1.3 Among streams that produce this op’s operands, choose the one with the
//        largest cost from root as the parent. This is because it is likely the
//        path that triggers the op execution.
// 2. Try merging streams
//    2.1 Traverse the stream tree in post-order (which is also a reverse
//        topological order of the original DAG).
//    2.2 Merge children that are smaller than the cost threshold together until
//        exceeding the threshold. At this point, there is at most one child
//        left whose cost is smaller than the threshold. If there is, merge it
//        with the current stream
//    2.3 Choose the child with the highest cost and merge it with the current
//        stream.
//
// TODO(chky): Add g3doc and link it here.

#ifndef TFRT_COMPILER_STREAM_ANALYSIS_H_
#define TFRT_COMPILER_STREAM_ANALYSIS_H_

#include <optional>

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace tfrt {
namespace compiler {

// Stream describes a sequence of operations that are executed sequentially.
class Stream {
 public:
  Stream(int id, int64_t cost, int parent_id, mlir::Operation* parent_op)
      : id_(id), cost_(cost), parent_id_(parent_id), parent_op_(parent_op) {}

  // The total cost of this stream, which is the sum of costs of all operations
  // in this stream.
  int64_t cost() const { return cost_; }

  // The id for this stream, which is non-negative and unique in a function.
  int id() const { return id_; }

  // The id for the parent stream. If the current stream is the root, then its
  // parent_id is -1.
  int parent_id() const { return parent_id_; }

  // The `parent_op` is the op in the parent stream that triggers the execution
  // of this stream. It can be nullptr in the case of root stream.
  mlir::Operation* parent_op() const { return parent_op_; }

  // The operations in this stream in the original topological order.
  llvm::ArrayRef<mlir::Operation*> ops() const { return ops_; }

  // Returns the child streams for the `op` in this stream.
  const llvm::SmallSetVector<const Stream*, 4>& GetChildStreams(
      mlir::Operation* op) const {
    auto iter = child_streams_.find(op);
    if (iter != child_streams_.end()) return iter->second;

    static const auto* empty_set = new llvm::SmallSetVector<const Stream*, 4>();
    return *empty_set;
  }

  // Returns the child streams for the root op if this is a root stream.
  const llvm::SmallSetVector<const Stream*, 4>& GetChildStreamsForRootOp()
      const {
    assert(parent_id_ == -1);
    return GetChildStreams(nullptr);
  }

 private:
  int id_ = -1;
  int64_t cost_ = 0;
  int parent_id_ = -1;
  mlir::Operation* parent_op_ = nullptr;
  // The operations in this stream in the original topological order.
  llvm::SmallVector<mlir::Operation*> ops_;
  llvm::DenseMap<mlir::Operation*, llvm::SmallSetVector<const Stream*, 4>>
      child_streams_;

  friend class StreamAnalysis;

  // TODO(chky): Add more information for a stream, such as incoming and
  // outgoing data dependencies, and child streams.
};

// StreamAnalysis is a per-function analysis that produces the stream tree of a
// function.
class StreamAnalysis {
 public:
  class CostModelInterface {
   public:
    virtual ~CostModelInterface();

    // The implementation is expected to return a positive value or
    // std::nullopt if a cost cannot be computed. If std::nullopt is returned,
    // stream analysis will use cost threshold as the cost for this op.
    virtual std::optional<int64_t> GetOperationCost(
        mlir::Operation* op) const = 0;
  };

  explicit StreamAnalysis(mlir::func::FuncOp op,
                          const CostModelInterface* cost_model = nullptr);
  explicit StreamAnalysis(mlir::Block& block,
                          const CostModelInterface* cost_model = nullptr);

  // Return the stream that contains `op`. An operation can only belong to one
  // stream.
  const Stream& GetStream(mlir::Operation* op) const {
    const auto* stream = stream_map_.lookup(op);
    assert(stream != nullptr);
    return *stream;
  }

  // Return the root stream of the current function, which is the entry of the
  // function.
  const Stream& GetRootStream() const;

  size_t GetNumStreams() const { return streams_.size(); }

  // The cost threshold is used to decide whether to merge two streams. When
  // there are independent streams, if the cost of a stream is smaller than this
  // threshold, then it would be worth merging this stream with others, while if
  // the cost is larger or equal to the threshold, then it is not worth doing
  // so. Note that dependent streams can still be merged regardless of the cost.
  // It is set through the module attribute `tfrt.cost_threshold`.
  int64_t GetCostThreshold() const { return options_.cost_threshold; }

 private:
  void GetOptionsForBlock(mlir::Block& block);
  void AnalyzeBlock(mlir::Block& block);
  void ScheduleOpForwardPass(mlir::Block& block);
  void BuildStreamBackwardPass(mlir::Block& block);
  void BuildStreamForOp(mlir::Operation* op);
  void MergeInterDependentStreams(llvm::SmallVector<int, 4>& child_stream_ids);
  void MergeStreams(int from_id, int to_id);
  void FinalizeStreams(mlir::Block& block);
  int64_t GetOperationCost(mlir::Operation* op) const;

  // BuildInfo is a temporary data structure for keeping stream and op
  // information during building the stream tree. It is used for efficient
  // building but not user-friendly. When building is done, it will be finalized
  // to a more user-friendly format in FinalizeStreams().
  struct BuildInfo {
    struct StreamInfo {
      int64_t cost = 0;
      mlir::Operation* parent_op = nullptr;
      // `merge_to_stream_id` is the id of the stream that this stream should be
      // merged into.
      int merge_to_stream_id = -1;
      // `side_deps` are ids of streams that are ancestors of the current stream
      // and that have data dependencies with the current stream.
      llvm::SmallSetVector<int, 4> side_deps;
      // Whether this stream contains the return op.
      bool contains_return_op = false;
    };

    struct OpInfo {
      int stream_id = -1;
      int64_t cost = 0;

      // `scheduled_users` are a subset of users of the current operation. Some
      // of the users of the current operation might be ready for execution
      // after the current operation finishes, while other users might be
      // triggered by other operations. So stream analysis performs a
      // compile-time guess to determine such a subset that are likely to be
      // triggered by the current operation. This is done in
      // ScheduleOpForwardPass().
      llvm::SmallVector<mlir::Operation*, 4> scheduled_users;

      // `side_uses` are the user ops of this op's results that are not in
      // `scheduled_users`.
      llvm::SmallDenseSet<mlir::Operation*, 2> side_uses;
    };

    // `stream_infos` is a temporary data structure to keep stream information
    // during building the stream tree.
    llvm::SmallVector<StreamInfo, 4> stream_infos;
    // `op_map` is a temporary data structure to keep per-op information like
    // cost, its stream_id during building the stream tree.
    llvm::DenseMap<mlir::Operation*, OpInfo> op_map;
    // Resolve the stream of `op` to the latest one. During the analysis,
    // streams might be merged to others, so the stream_id of an op might be
    // stale.
    void ResolveStreamId(mlir::Operation* op);
    // Find the latest stream id of `stream_id` if it is merged to another
    // stream.
    int FindLatestStreamId(int stream_id) const;
  };

  void AssignOpToStream(mlir::Operation* op, BuildInfo::OpInfo& op_info,
                        int stream_id, BuildInfo::StreamInfo& stream_info);

  struct Options {
    // `cost_threshold` is the lower threshold for streams costs. We try to
    // merge independent streams to one stream with a cost that is barely
    // exceeding this threshold.
    int64_t cost_threshold = 1;
    // If `merge_inter_dependent_streams` is true, when merging independent
    // streams, it will try to merge those with inter data dependencies first.
    bool merge_inter_dependent_streams = false;
  };

  BuildInfo build_info_;
  Options options_;

  // `streams_` contains the finalized Stream objects that contain information
  // for users to query.
  llvm::SmallVector<std::unique_ptr<Stream>, 4> streams_;

  // `stream_map_` contains the finalized op-to-stream mapping.
  llvm::DenseMap<mlir::Operation*, Stream*> stream_map_;

  const CostModelInterface* cost_model_ = nullptr;
};

}  // namespace compiler
}  // namespace tfrt

#endif  // TFRT_COMPILER_STREAM_ANALYSIS_H_
