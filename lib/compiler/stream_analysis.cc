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

// This implements StreamAnalysis that analyzes sequential regions of a graph.

#include "tfrt/compiler/stream_analysis.h"

#include "tfrt/basic_kernels/opdefs/tfrt_traits.h"

namespace tfrt {
namespace compiler {
namespace {

constexpr mlir::Operation* kRootOperation = nullptr;

int64_t GetCostThresholdFromModule(mlir::ModuleOp op) {
  // default cost threshold is set to a lowest possible value, to disable any
  // merging of streams.
  static constexpr int64_t kDefaultCostThreshold = 1;

  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("tfrt.cost_threshold")) {
    return std::max(kDefaultCostThreshold, attr.getInt());
  }

  return kDefaultCostThreshold;
}

}  // namespace

int64_t StreamAnalysis::GetOperationCost(mlir::Operation* op) const {
  // Root has the lowest cost.
  if (op == kRootOperation) return 1;

  if (op->hasTrait<mlir::OpTrait::tfrt::CostTrait>()) {
    int64_t cost = op->getAttrOfType<mlir::IntegerAttr>("_tfrt_cost").getInt();
    // Operation costs should be verified by the verifier in the CostTrait to
    // have positive values.
    assert(cost > 0);
    return cost;
  }

  // If there is no cost specified for this operation, We conservatively return
  // the cost threshold as its cost. So we treat operations without cost as
  // expensive ops, but not too expensive to outweigh any other operations.
  return GetCostThreshold();
}

// ScheduleOpForwardPass traverses the operations in a topological order, and
// makes compile-time scheduling decisions: when there are more than one inputs
// to an operation, it chooses the path with the largest cost from root. This
// creates a directed tree that is a subgraph of the original DAG.
void StreamAnalysis::ScheduleOpForwardPass(mlir::Block& block) {
  // `cost_from_root_map` is to keep the total cost for each operation from
  // root.
  llvm::DenseMap<mlir::Operation*, int64_t> cost_from_root_map;

  // Set up the root operation.
  build_info_.op_map[kRootOperation].cost = GetOperationCost(kRootOperation);
  cost_from_root_map[kRootOperation] = build_info_.op_map[kRootOperation].cost;

  // For each operation, we try to decide its parent op.
  for (auto& op : block) {
    auto& current_op_info = build_info_.op_map[&op];
    current_op_info.cost = GetOperationCost(&op);
    int64_t cost_from_root = current_op_info.cost;

    if (op.getNumOperands() == 0) {
      // If it is an op with no operands, make it a child of the root.
      build_info_.op_map[kRootOperation].scheduled_users.push_back(&op);
    } else {
      // For ops with operands, pick the operand's defining operation with the
      // highest cost path from root as the parent. This is because the highest
      // cost path from the root to this operation should also be the path that
      // trigger the execution of this operation.

      mlir::Operation* parent_op = nullptr;

      // Note that only positive integers can be valid costs.
      int64_t parent_cost_from_root = 0;

      for (auto operand : op.getOperands()) {
        auto* def = operand.getDefiningOp();
        assert(build_info_.op_map.count(def) > 0);

        assert(cost_from_root_map.count(def) > 0);
        int64_t operand_cost_from_root = cost_from_root_map[def];
        assert(operand_cost_from_root > 0);

        if (operand_cost_from_root > parent_cost_from_root) {
          parent_op = def;
          parent_cost_from_root = operand_cost_from_root;
        }
      }
      build_info_.op_map[parent_op].scheduled_users.push_back(&op);

      cost_from_root += parent_cost_from_root;
    }

    cost_from_root_map[&op] = cost_from_root;
  }
}

// BuildStreamForOp first tries to merge the child streams of the current `op`
// (ie. the streams led by its scheduled users that are decided in
// ScheduleOpForwardPass), and then assigns the current `op` to one of the child
// streams. This function is supposed be to called in the reverse topological
// order of operations to get an optimal stream tree.
void StreamAnalysis::BuildStreamForOp(mlir::Operation* op) {
  auto& op_info = build_info_.op_map[op];

  if (op_info.scheduled_users.empty()) {
    // Create a new stream for ops without scheduled users.
    op_info.stream_id = build_info_.stream_infos.size();
    BuildInfo::StreamInfo stream_info;
    stream_info.cost = op_info.cost;
    build_info_.stream_infos.push_back(stream_info);
    return;
  }

  // If there are scheduled users, then each of them comes from different
  // streams. And we will assign the current op to the stream with the highest
  // cost.

  int max_cost_stream_id = -1;
  int64_t max_cost = 0;

  auto update_max_cost_stream = [&](int64_t cost, int stream_id) {
    if (max_cost < cost) {
      max_cost = cost;
      max_cost_stream_id = stream_id;
    }
  };

  auto merge_streams = [this](int from_id, int to_id) {
    assert(from_id != to_id);
    // Add the cost of from_id to to_id.
    build_info_.stream_infos[to_id].cost +=
        build_info_.stream_infos[from_id].cost;
    assert(build_info_.stream_infos[from_id].merge_to_stream_id < 0);
    // Set to_id in from_id's stream to indicate that they are merged.
    build_info_.stream_infos[from_id].merge_to_stream_id = to_id;
  };

  // The loop below does the following:
  //  1) try merging child streams that are cheaper than cost threshold.
  //  2) find out the child stream with the highest cost.

  // `merged_child_stream_ids` collects the stream ids for merged child streams.
  llvm::SmallVector<int, 4> merged_child_stream_ids;

  for (auto* child_op : op_info.scheduled_users) {
    auto& child_op_info = build_info_.op_map[child_op];
    assert(child_op_info.stream_id >= 0);
    assert(child_op_info.stream_id < build_info_.stream_infos.size());

    auto& child_stream_info = build_info_.stream_infos[child_op_info.stream_id];
    assert(child_stream_info.parent_op == nullptr);
    child_stream_info.parent_op = op;

    if (child_stream_info.cost >= GetCostThreshold()) {
      update_max_cost_stream(child_stream_info.cost, child_op_info.stream_id);
      continue;
    }

    // Below logic merges the cheap child streams, so that each of the merged
    // streams is barely over the cost threshold. The reason is that if there
    // are multiple child streams that are cheaper than the threshold, then we
    // want to merge them to reduce the number of streams. On the other hand, we
    // don't want too expensive streams due to merging as it increases the
    // latency.
    if (merged_child_stream_ids.empty() ||
        build_info_.stream_infos[merged_child_stream_ids.back()].cost >=
            GetCostThreshold()) {
      merged_child_stream_ids.push_back(child_op_info.stream_id);
    } else {
      merge_streams(child_op_info.stream_id, merged_child_stream_ids.back());
      // Update the stream_id in the op_info to the merged stream_id.
      child_op_info.stream_id = merged_child_stream_ids.back();
    }
    update_max_cost_stream(
        build_info_.stream_infos[merged_child_stream_ids.back()].cost,
        merged_child_stream_ids.back());
  }

  // Assign the current op to the stream with the highest cost found.
  assert(op_info.stream_id < 0);
  assert(max_cost_stream_id >= 0);
  op_info.stream_id = max_cost_stream_id;
  build_info_.stream_infos[max_cost_stream_id].cost += op_info.cost;
  // Reset parent_op because we haven't found the parent_op for this newly
  // merged stream yet.
  build_info_.stream_infos[max_cost_stream_id].parent_op = nullptr;

  // There is at most one single cheap stream (ie. the stream whose cost is
  // below cost_threshold) left, and it must be the last one in
  // `merged_child_stream_ids`. If there is, merge it into the current stream.
  //
  // TODO(chky): This effectively merges this cheap stream with the highest cost
  // child stream for the simplicity in implementation. Consider if it should be
  // merged to the next lowest cost child stream if there are use cases that
  // shows the latter approach is better..
  if (!merged_child_stream_ids.empty() &&
      build_info_.stream_infos[merged_child_stream_ids.back()].cost <
          GetCostThreshold() &&
      max_cost_stream_id != merged_child_stream_ids.back()) {
    merge_streams(merged_child_stream_ids.back(), op_info.stream_id);
  }
}

// BuildStreamBackwardPass traverse the graph in reversed topological order,
// create streams for ops and assign ops to existing streams.
void StreamAnalysis::BuildStreamBackwardPass(mlir::Block& block) {
  // Build streams for each op in reversed topological order.
  for (auto& op : llvm::reverse(block)) {
    BuildStreamForOp(&op);
  }

  // Lastly, build the stream for the root.
  BuildStreamForOp(kRootOperation);
}

// FinalizeStreams creates user-friendly data structure so that the stream
// information can be easily queried.
void StreamAnalysis::FinalizeStreams(mlir::Block& block) {
  llvm::DenseMap<int, Stream*> stream_id_map;

  // Finalize root first.
  const auto& root_op_info = build_info_.op_map[kRootOperation];
  const auto& root_stream_info =
      build_info_.stream_infos[root_op_info.stream_id];
  assert(root_stream_info.merge_to_stream_id == -1);
  assert(root_stream_info.parent_op == nullptr);
  streams_.push_back(
      std::make_unique<Stream>(root_op_info.stream_id, root_stream_info.cost,
                               /*parent_id=*/-1, root_stream_info.parent_op));
  stream_map_[kRootOperation] = streams_.back().get();
  stream_id_map[root_op_info.stream_id] = streams_.back().get();

  // Then finalize the rest ops.
  for (auto& op : block) {
    auto& op_info = build_info_.op_map[&op];
    int stream_id = op_info.stream_id;
    assert(stream_id >= 0);

    // Find out the final stream_id it belongs to.
    while (build_info_.stream_infos[stream_id].merge_to_stream_id >= 0) {
      stream_id = build_info_.stream_infos[stream_id].merge_to_stream_id;
    }

    int final_stream_id = stream_id;

    // Update the relevant stream infos.
    stream_id = op_info.stream_id;
    while (build_info_.stream_infos[stream_id].merge_to_stream_id >= 0) {
      int next_id = build_info_.stream_infos[stream_id].merge_to_stream_id;
      build_info_.stream_infos[stream_id].merge_to_stream_id = final_stream_id;
      stream_id = next_id;
    }

    op_info.stream_id = final_stream_id;

    Stream*& stream = stream_id_map[op_info.stream_id];

    if (stream == nullptr) {
      auto* parent_op = build_info_.stream_infos[op_info.stream_id].parent_op;
      // Note that parent_id is already finalized, as we are processing in the
      // topological order.
      int parent_id = build_info_.op_map[parent_op].stream_id;
      assert(build_info_.stream_infos[parent_id].merge_to_stream_id < 0);
      streams_.push_back(std::make_unique<Stream>(
          op_info.stream_id, build_info_.stream_infos[op_info.stream_id].cost,
          parent_id, parent_op));
      stream = streams_.back().get();
    }

    stream_map_[&op] = stream;
  }
}

void StreamAnalysis::AnalyzeBlock(mlir::Block& block) {
  cost_threshold_ = GetCostThresholdFromModule(
      block.getParentOp()->getParentOfType<mlir::ModuleOp>());
  ScheduleOpForwardPass(block);
  BuildStreamBackwardPass(block);
  FinalizeStreams(block);
}

const Stream& StreamAnalysis::GetRootStream() const {
  return GetStream(kRootOperation);
}

}  // namespace compiler
}  // namespace tfrt
