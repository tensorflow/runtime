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

#include "tfrt/basic_kernels/opdefs/basic_kernels.h"
#include "tfrt/basic_kernels/opdefs/types.h"
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"

namespace tfrt {
namespace compiler {
namespace {

constexpr mlir::Operation* kRootOperation = nullptr;

mlir::Attribute GetOptionAttribute(mlir::Block& block,
                                   llvm::StringRef attr_name) {
  auto* parent = block.getParentOp();

  // Try to use function-level option first.
  if (auto func = llvm::dyn_cast<mlir::FuncOp>(parent)) {
    if (auto attr = func->getAttr(attr_name)) {
      return attr;
    }
  }

  // If there is no function-level option, use module-level option.
  auto module = parent->getParentOfType<mlir::ModuleOp>();
  return module->getAttr(attr_name);
}

int64_t GetCostThresholdForBlock(mlir::Block& block) {
  // default cost threshold is set to a lowest possible value, to disable any
  // merging of streams.
  static constexpr int64_t kDefaultCostThreshold = 1;

  if (auto attr = GetOptionAttribute(block, "tfrt.cost_threshold")
                      .dyn_cast_or_null<mlir::IntegerAttr>()) {
    return std::max(kDefaultCostThreshold, attr.getInt());
  }

  // Otherwise, use default cost threshold.
  return kDefaultCostThreshold;
}

int64_t GetUpperCostThresholdForBlock(mlir::Block& block) {
  // Use -1 as the default, which means there is no limit on the cost of a
  // stream.
  static constexpr int64_t kDefaultMaxStreamCost = -1;

  if (auto attr = GetOptionAttribute(block, "tfrt.upper_cost_threshold")
                      .dyn_cast_or_null<mlir::IntegerAttr>()) {
    return attr.getInt();
  }

  return kDefaultMaxStreamCost;
}

bool GetMergeInterDependentStreams(mlir::Block& block) {
  if (auto attr =
          GetOptionAttribute(block, "tfrt.merge_inter_dependent_streams")
              .dyn_cast_or_null<mlir::BoolAttr>()) {
    return attr.getValue();
  }
  return false;
}

}  // namespace

int64_t StreamAnalysis::GetOperationCost(mlir::Operation* op) const {
  // Root has the lowest cost.
  if (op == kRootOperation) return 1;

  // A few TFRT kernels are guaranteed to be cheap.
  if (llvm::isa<ReturnOp, MergeChainsOp>(op)) return 1;

  // Check if operations defines a cost function.
  if (auto cost_function = mlir::dyn_cast<CostFunctionInterface>(op)) {
    int64_t cost = cost_function.cost();
    assert(cost > 0 && "cost must be a positive value");
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

        // Record the data dependencies even if they are not on the path that
        // triggers the execution. This will be used later when we are trying to
        // merge parallel streams. Parallel streams with data dependencies will
        // be preferred to be merged. Note that control dependencies are skipped
        // as we prefer to merge for data dependencies.
        if (!operand.getType().isa<ChainType>()) {
          current_op_info.side_defs.insert(def);
        }

        if (operand_cost_from_root > parent_cost_from_root) {
          parent_op = def;
          parent_cost_from_root = operand_cost_from_root;
        }
      }
      current_op_info.side_defs.erase(parent_op);
      build_info_.op_map[parent_op].scheduled_users.push_back(&op);

      cost_from_root += parent_cost_from_root;
    }

    cost_from_root_map[&op] = cost_from_root;
  }
}

void StreamAnalysis::MergeStreams(int from_id, int to_id) {
  assert(from_id != to_id);

  auto& from_stream = build_info_.stream_infos[from_id];
  auto& to_stream = build_info_.stream_infos[to_id];
  assert(from_stream.merge_to_stream_id < 0);
  assert(to_stream.merge_to_stream_id < 0);

  // First, we merge inter-stream data dependencies. This is done by 1) removing
  // dependencies in `to_stream` that are produced by `from_stream`; and 2) copy
  // dependencies in `from_stream` that are not produced by `to_stream` to
  // `to_stream`.
  llvm::SmallVector<mlir::Operation*, 2> stale_side_defs;
  for (auto* side_def : to_stream.side_deps) {
    auto& side_def_info = build_info_.op_map[side_def];
    // If the defining op of this dep has not been assigned to a stream yet, we
    // don't need to remove it.
    if (side_def_info.stream_id < 0) continue;
    // Try update the stream id of the defining op as its original stream might
    // be merged.
    build_info_.ResolveStreamId(side_def);
    if (side_def_info.stream_id == from_id) {
      // Remove if the defining op is in `from_stream`.
      stale_side_defs.push_back(side_def);
    }
  }
  for (auto* side_def : stale_side_defs) {
    to_stream.side_deps.erase(side_def);
  }

  for (auto* side_def : from_stream.side_deps) {
    auto& side_def_info = build_info_.op_map[side_def];
    if (side_def_info.stream_id < 0) {
      // If the defining op of this dep has not been assigned to a stream yet,
      // we should keep it in the `to_stream`.
      to_stream.side_deps.insert(side_def);
    } else {
      // Try update the stream id of the defining op as its original stream
      // might be merged.
      build_info_.ResolveStreamId(side_def);
      if (side_def_info.stream_id != to_id) {
        // Keep if the defining op is not in `to_stream`.
        to_stream.side_deps.insert(side_def);
      }
    }
  }

  // Add the cost of from_id to to_id.
  to_stream.cost += from_stream.cost;
  // Set to_id in from_id's stream to indicate that they are merged.
  from_stream.merge_to_stream_id = to_id;
}

void StreamAnalysis::MergeInterDependentStreams(
    llvm::SmallVector<int, 4>& child_stream_ids) {
  llvm::SmallDenseSet<int, 4> child_stream_id_set(child_stream_ids.begin(),
                                                  child_stream_ids.end());

  // Keep streams that are merged during the following loop to avoid double
  // merging.
  llvm::SmallDenseSet<int, 4> stale_child_stream_ids;
  for (int child_stream_id : child_stream_ids) {
    auto& child_stream_info = build_info_.stream_infos[child_stream_id];

    // Skip if it is already merged in a previous iteration.
    if (stale_child_stream_ids.count(child_stream_id) > 0) {
      assert(child_stream_info.merge_to_stream_id >= 0);
      continue;
    }

    assert(child_stream_info.merge_to_stream_id < 0);

    // Iterate through all side dependencies and try to merge those that are in
    // the candidate stream pool.
    llvm::SmallDenseSet<int, 2> side_def_streams_to_merge;
    for (auto* side_def : child_stream_info.side_deps) {
      auto& side_def_info = build_info_.op_map[side_def];
      // Skip if the defining op has not been assigned to a stream yet.
      if (side_def_info.stream_id < 0) continue;
      // Update the stream id of the defining op.
      build_info_.ResolveStreamId(side_def);
      assert(side_def_info.stream_id != child_stream_id);

      // If the stream is in the candidate stream pool, and the its cost is
      // below the threshold, try merge it into the current stream.
      if (child_stream_id_set.count(side_def_info.stream_id) > 0) {
        auto& side_def_stream_info =
            build_info_.stream_infos[side_def_info.stream_id];
        if (side_def_stream_info.cost >= GetCostThreshold()) continue;
        side_def_streams_to_merge.insert(side_def_info.stream_id);
      }
    }

    for (int stream_id : side_def_streams_to_merge) {
      MergeStreams(/*from_id=*/stream_id,
                   /*to_id=*/child_stream_id);
      stale_child_stream_ids.insert(stream_id);
      if (child_stream_info.cost >= GetCostThreshold()) break;
    }
  }

  // Update the candidate stream pool as some streams may have been merged into
  // others.
  child_stream_ids.erase(
      std::remove_if(
          child_stream_ids.begin(), child_stream_ids.end(),
          [&](int id) { return stale_child_stream_ids.count(id) > 0; }),
      child_stream_ids.end());
}

void StreamAnalysis::AssignOpToStream(mlir::Operation* op,
                                      BuildInfo::OpInfo& op_info, int stream_id,
                                      BuildInfo::StreamInfo& stream_info) {
  assert(stream_info.merge_to_stream_id == -1);
  op_info.stream_id = stream_id;
  stream_info.cost += op_info.cost;
  // Update the side dependencies of the stream by adding the side dependencies
  // of the op. The op might be a side dep of some child streams itself, remove
  // in this case.
  stream_info.side_deps.erase(op);
  for (auto* side_def : op_info.side_defs) {
    // The defining op of side dependencies of the current op must not have
    // been assigned to a stream yet, because we are building streams in a
    // reverse topological order.
    assert(build_info_.op_map[side_def].stream_id < 0);
    stream_info.side_deps.insert(side_def);
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
    int new_stream_id = build_info_.stream_infos.size();
    build_info_.stream_infos.push_back({});
    auto& new_stream_info = build_info_.stream_infos.back();
    AssignOpToStream(op, op_info, new_stream_id, new_stream_info);
    return;
  }

  // If there are scheduled users, then each of them comes from different
  // streams. And we will assign the current op to the stream with the highest
  // cost.

  // Find all stream ids of candidates and keep them in a vector to have stable
  // order.
  llvm::SmallVector<int, 4> child_stream_ids;
  for (auto* child_op : op_info.scheduled_users) {
    auto& child_op_info = build_info_.op_map[child_op];
    assert(child_op_info.stream_id >= 0);
    assert(child_op_info.stream_id < build_info_.stream_infos.size());

    auto& child_stream_info = build_info_.stream_infos[child_op_info.stream_id];
    assert(child_stream_info.parent_op == nullptr);
    assert(child_stream_info.merge_to_stream_id == -1);

    // Blindly make the current op the parent of child streams. It will be
    // updated if any are merged later.
    child_stream_info.parent_op = op;

    child_stream_ids.push_back(child_op_info.stream_id);
  }

  // First, we try merging child streams that have inter-dependencies.
  if (options_.merge_inter_dependent_streams)
    MergeInterDependentStreams(child_stream_ids);

  // After we merge streams with inter-dependencies, there might be still small
  // streams left. Then we just merge them in a random order. The loop below
  // does the following:
  //  1) try merging child streams that are cheaper than cost threshold.
  //  2) find out the child stream with the highest cost.
  int max_cost_stream_id = -1;
  int64_t max_cost = 0;

  auto update_max_cost_stream = [&](int64_t cost, int stream_id) {
    // Try to find the stream with the largest cost that is also smaller than
    // the upper_cost_threshold among the candidates.
    if ((options_.upper_cost_threshold < 0 ||
         cost < options_.upper_cost_threshold) &&
        max_cost < cost) {
      max_cost = cost;
      max_cost_stream_id = stream_id;
    }
  };

  // `merged_child_stream_ids` collects the stream ids for merged child streams.
  llvm::SmallVector<int, 4> merged_child_stream_ids;
  for (int child_stream_id : child_stream_ids) {
    auto& child_stream_info = build_info_.stream_infos[child_stream_id];
    assert(child_stream_info.merge_to_stream_id == -1);

    if (child_stream_info.cost >= GetCostThreshold()) {
      update_max_cost_stream(child_stream_info.cost, child_stream_id);
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
      merged_child_stream_ids.push_back(child_stream_id);
    } else {
      MergeStreams(child_stream_id, merged_child_stream_ids.back());
      // Update the stream_id in the op_info to the merged stream_id.
      child_stream_id = merged_child_stream_ids.back();
    }
    update_max_cost_stream(
        build_info_.stream_infos[merged_child_stream_ids.back()].cost,
        merged_child_stream_ids.back());
  }

  assert(op_info.stream_id < 0);

  if (max_cost_stream_id < 0) {
    // Create a new stream for ops without if it fails to find an appropriate
    // child stream.
    int new_stream_id = build_info_.stream_infos.size();
    build_info_.stream_infos.push_back({});
    auto& new_stream_info = build_info_.stream_infos.back();
    AssignOpToStream(op, op_info, new_stream_id, new_stream_info);
  } else {
    // Otherwise assign the current op to the stream with the highest cost
    // found.
    auto& current_op_stream = build_info_.stream_infos[max_cost_stream_id];
    assert(current_op_stream.merge_to_stream_id == -1);
    // Reset parent_op because we haven't found the parent_op for this newly
    // merged stream yet.
    current_op_stream.parent_op = nullptr;
    AssignOpToStream(op, op_info, max_cost_stream_id, current_op_stream);

    // There is at most one single cheap stream (ie. the stream whose cost is
    // below cost_threshold) left, and it must be the last one in
    // `merged_child_stream_ids`. If there is, merge it into the current stream.
    //
    // TODO(chky): This effectively merges this cheap stream with the highest
    // cost child stream for the simplicity in implementation. Consider if it
    // should be merged to the next lowest cost child stream if there are use
    // cases that shows the latter approach is better..
    if (!merged_child_stream_ids.empty() &&
        build_info_.stream_infos[merged_child_stream_ids.back()].cost <
            GetCostThreshold() &&
        max_cost_stream_id != merged_child_stream_ids.back()) {
      MergeStreams(merged_child_stream_ids.back(), op_info.stream_id);
    }
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
    build_info_.ResolveStreamId(&op);

    auto& op_info = build_info_.op_map[&op];
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
  GetOptionsForBlock(block);
  ScheduleOpForwardPass(block);
  BuildStreamBackwardPass(block);
  FinalizeStreams(block);
}

const Stream& StreamAnalysis::GetRootStream() const {
  return GetStream(kRootOperation);
}

void StreamAnalysis::BuildInfo::ResolveStreamId(mlir::Operation* op) {
  assert(op_map.count(op) > 0);
  auto& op_info = op_map[op];
  int stream_id = op_info.stream_id;
  assert(stream_id >= 0);

  // Find out the final stream_id it belongs to.
  while (stream_infos[stream_id].merge_to_stream_id >= 0) {
    stream_id = stream_infos[stream_id].merge_to_stream_id;
  }
  int final_stream_id = stream_id;

  // Update the relevant stream infos.
  stream_id = op_info.stream_id;
  while (stream_infos[stream_id].merge_to_stream_id >= 0) {
    int next_id = stream_infos[stream_id].merge_to_stream_id;
    stream_infos[stream_id].merge_to_stream_id = final_stream_id;
    stream_id = next_id;
  }
  op_info.stream_id = final_stream_id;
}

void StreamAnalysis::GetOptionsForBlock(mlir::Block& block) {
  options_.cost_threshold = GetCostThresholdForBlock(block);
  options_.upper_cost_threshold = GetUpperCostThresholdForBlock(block);
  options_.merge_inter_dependent_streams = GetMergeInterDependentStreams(block);
}

}  // namespace compiler
}  // namespace tfrt
