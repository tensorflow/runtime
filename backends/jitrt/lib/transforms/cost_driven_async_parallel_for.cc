/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tfrt/jitrt/transforms/codegen_passes.h"

namespace tfrt {
namespace jitrt {
namespace {

namespace vector = mlir::vector;
namespace math = mlir::math;
namespace arith = mlir::arith;
namespace scf = mlir::scf;
namespace memref = mlir::memref;

using llvm::Optional;
using mlir::dyn_cast;
using mlir::ImplicitLocOpBuilder;
using mlir::isa;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::Region;
using mlir::Value;

#define GEN_PASS_CLASSES
#include "tfrt/jitrt/transforms/codegen_gen_passes.h.inc"

struct CostDrivenAsyncParallelForPass
    : public CostDrivenAsyncParallelForBase<CostDrivenAsyncParallelForPass> {
  CostDrivenAsyncParallelForPass() = default;

  CostDrivenAsyncParallelForPass(bool async_dispatch,
                                 int32_t num_worker_threads,
                                 bool legacy_behavior) {
    this->async_dispatch_ = async_dispatch;
    this->num_worker_threads_ = num_worker_threads;
    this->legacy_behavior_ = legacy_behavior;
  }

  void runOnOperation() override;
};

// Cost measuring unit.
//
// For CPU roughly corresponds to cycles. For RAM roughly corresponds to bytes.
using InverseThroughput = Value;

// Approximate cost of executing an op on a modern CPU.
//
// Encapsulates the leaf nodes of the IR that is being emitted as the cost
// estimation progresses. The leaf nodes correspond to CPU and RAM
// inverse-throughputs.
class Cost {
 public:
  explicit Cost(ImplicitLocOpBuilder &builder, size_t ram_cost = 0,
                size_t cpu_cost = 0);

  Cost(ImplicitLocOpBuilder &builder, InverseThroughput ram_cost,
       InverseThroughput cpu_cost);

  InverseThroughput &ram();
  InverseThroughput &cpu();

  Cost &operator*=(const size_t &rhs);
  Cost operator*(const size_t &rhs);
  Cost &operator+=(const Cost &rhs);
  Cost operator+(const Cost &rhs);

 private:
  ImplicitLocOpBuilder builder_;
  InverseThroughput ram_;
  InverseThroughput cpu_;
};

// Estimates execution time for an op on a modern CPU.
//
// Errs on the lower end, but is not strictly a lower bound estimate. Targeting
// being within an order of magnitude of the correct value.
class CostModel {
 public:
  CostModel(mlir::ImplicitLocOpBuilder &builder, mlir::Operation &root_op,
            bool emitRemarks);

  ImplicitLocOpBuilder &builder();
  bool IsBeforeRoot(Operation &op);
  bool IsBeforeRoot(Value &value);
  Optional<Value> GetIterations(Operation &op, Value lower_bound,
                                Value upper_bound, Value step);
  Cost NewCost(size_t ram_cost, size_t cpu_cost);
  Cost NewCost(InverseThroughput ram_cost, InverseThroughput cpu_cost);
  Cost ZeroCost();
  Cost EstimateCost(Operation &op);
  Cost EstimateCost(Region &region);
  Value CostToNanoseconds(Cost cost);

 private:
  ImplicitLocOpBuilder builder_;
  Operation &root_op_;
  bool emit_remarks_;

  Cost EstimateCostSwitch(Operation &op);
  void ConditionallyEmitRemark(Operation &op, Cost cost);
};

Cost::Cost(ImplicitLocOpBuilder &builder, size_t ram_cost, size_t cpu_cost)
    : builder_(builder),
      ram_(builder.create<arith::ConstantIndexOp>(ram_cost)),
      cpu_(builder.create<arith::ConstantIndexOp>(cpu_cost)) {}

Cost::Cost(ImplicitLocOpBuilder &builder, InverseThroughput ram_cost,
           InverseThroughput cpu_cost)
    : builder_(builder), ram_(ram_cost), cpu_(cpu_cost) {}

InverseThroughput &Cost::ram() { return ram_; }
InverseThroughput &Cost::cpu() { return cpu_; }

Cost &Cost::operator*=(const size_t &rhs) {
  auto factor = builder_.create<arith::ConstantIndexOp>(rhs);
  auto scale = [&](InverseThroughput resource) {
    return builder_.create<arith::MulIOp>(resource, factor);
  };
  ram_ = scale(ram_);
  cpu_ = scale(cpu_);
  return *this;
}

Cost Cost::operator*(const size_t &rhs) {
  Cost ret = *this;
  ret *= rhs;
  return ret;
}

Cost &Cost::operator+=(const Cost &rhs) {
  ram_ = builder_.create<arith::AddIOp>(ram_, rhs.ram_);
  cpu_ = builder_.create<arith::AddIOp>(cpu_, rhs.cpu_);
  return *this;  // return the result by reference
}

Cost Cost::operator+(const Cost &rhs) {
  Cost ret = *this;
  ret += rhs;
  return ret;
}

Cost EstimateCostVector(CostModel &helper, Operation &op) {
  auto new_cost = [&](Value value) {
    if (auto type = value.getType().dyn_cast<mlir::ShapedType>()) {
      if (type.hasStaticShape()) {
        return helper.NewCost(
            /* ram */ type.getNumElements() * type.getElementTypeBitWidth() / 8,
            /* cpu */ 0);
      }
    }
    return helper.ZeroCost();
  };

  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return new_cost(loadOp.getResult());
  }

  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return new_cost(storeOp.getOperand(0));
  }

  return helper.ZeroCost();
}

Cost EstimateCostMemref(CostModel &helper, Operation &op) {
  auto new_cost = [&](Value value) {
    return helper.NewCost(
        /* ram */ value.getType().getIntOrFloatBitWidth() / 8,
        /* cpu */ 0);
  };

  if (auto load_op = dyn_cast<memref::LoadOp>(op)) {
    return new_cost(load_op.getResult());
  }

  if (auto store_op = dyn_cast<memref::StoreOp>(op)) {
    return new_cost(store_op.getOperand(0));
  }

  return helper.ZeroCost();
}

Cost EstimateCostArith(CostModel &helper, Operation &op) {
  if (isa<arith::DivUIOp, arith::DivSIOp, arith::CeilDivSIOp,
          arith::FloorDivSIOp, arith::RemSIOp, arith::RemUIOp>(op)) {
    return helper.NewCost(/* ram */ 0, /* cpu */ 3);
  }

  return helper.NewCost(/* ram */ 0, /* cpu */ 1);
}

Cost EstimateCostMath(CostModel &helper, Operation &op) {
  if (isa<math::AtanOp, math::Atan2Op, math::CosOp, math::SinOp, math::ExpOp,
          math::Exp2Op, math::ExpM1Op, math::LogOp, math::Log10Op,
          math::Log1pOp, math::Log2Op, math::PowFOp, math::RsqrtOp,
          math::SqrtOp, math::TanhOp>(op)) {
    return helper.NewCost(/* ram */ 0, /* cpu */ 100);
  }

  return helper.NewCost(/* ram */ 0, /* cpu */ 1);
}

Cost EstimateCostSCF(CostModel &helper, Operation &op) {
  auto scale_cost = [&](Value iterations, Cost cost) {
    auto scale = [&](Value scaled) -> Value {
      return helper.builder().create<arith::MulIOp>(scaled, iterations);
    };
    return helper.NewCost(scale(cost.ram()), scale(cost.cpu()));
  };

  if (auto for_op = dyn_cast<scf::ForOp>(op)) {
    Cost cost = helper.EstimateCost(for_op.getLoopBody());
    if (auto iterations =
            helper.GetIterations(*for_op, for_op.getLowerBound(),
                                 for_op.getUpperBound(), for_op.getStep())) {
      return scale_cost(*iterations, cost);
    }
    return cost;
  }

  if (auto parallel_op = dyn_cast<scf::ParallelOp>(op)) {
    Cost cost = helper.EstimateCost(parallel_op.getLoopBody());
    for (auto &inductionVariableDomain : llvm::enumerate(
             llvm::zip(parallel_op.getLowerBound(), parallel_op.getUpperBound(),
                       parallel_op.getStep()))) {
      Value lb, ub, step;
      std::tie(lb, ub, step) = inductionVariableDomain.value();

      if (auto iterations = helper.GetIterations(*parallel_op, lb, ub, step)) {
        cost = scale_cost(*iterations, cost);
      }
    }
    return cost;
  }

  return helper.ZeroCost();
}

CostModel::CostModel(ImplicitLocOpBuilder &builder, Operation &root_op,
                     bool emitRemarks)
    : builder_(builder), root_op_(root_op), emit_remarks_(emitRemarks) {}

ImplicitLocOpBuilder &CostModel::builder() { return builder_; }

bool CostModel::IsBeforeRoot(Operation &op) {
  return op.getBlock() == root_op_.getBlock() && op.isBeforeInBlock(&root_op_);
}

bool CostModel::IsBeforeRoot(Value &value) {
  return IsBeforeRoot(*value.getDefiningOp());
}

Optional<Value> CostModel::GetIterations(Operation &op, Value lower_bound,
                                         Value upper_bound, Value step) {
  if (IsBeforeRoot(lower_bound) && IsBeforeRoot(upper_bound) &&
      IsBeforeRoot(step)) {
    Value val = builder_.create<arith::CeilDivSIOp>(
        builder_.create<arith::SubIOp>(upper_bound, lower_bound), step);
    if (emit_remarks_) {
      mlir::emitRemark(op.getLoc()) << "iterations: " << val;
    }
    return val;
  }
  return llvm::None;
}

Cost CostModel::NewCost(size_t ram_cost, size_t cpu_cost) {
  return Cost(builder_, ram_cost, cpu_cost);
}

Cost CostModel::NewCost(InverseThroughput ram_cost,
                        InverseThroughput cpu_cost) {
  return Cost(builder_, ram_cost, cpu_cost);
}

Cost CostModel::ZeroCost() { return NewCost(0, 0); }

Cost CostModel::EstimateCostSwitch(Operation &op) {
  MLIRContext *ctx = op.getContext();
  if (op.getDialect() == ctx->getLoadedDialect("scf")) {
    return EstimateCostSCF(*this, op);
  }
  if (op.getDialect() == ctx->getLoadedDialect("memref")) {
    return EstimateCostMemref(*this, op);
  }
  if (op.getDialect() == ctx->getLoadedDialect("vector")) {
    return EstimateCostVector(*this, op);
  }
  if (op.getDialect() == ctx->getLoadedDialect("arith")) {
    return EstimateCostArith(*this, op);
  }
  if (op.getDialect() == ctx->getLoadedDialect("math")) {
    return EstimateCostMath(*this, op);
  }
  return ZeroCost();
}

Cost CostModel::EstimateCost(Operation &op) {
  Cost cost = EstimateCostSwitch(op);
  ConditionallyEmitRemark(op, cost);
  return cost;
}

void CostModel::ConditionallyEmitRemark(Operation &op, Cost cost) {
  if (emit_remarks_) {
    mlir::emitRemark(op.getLoc()) << "ramCost: { " << cost.ram() << " } "
                                  << "cpuCost: { " << cost.cpu() << " }";
  }
}

Cost CostModel::EstimateCost(Region &region) {
  llvm::SmallVector<Cost> costs;
  for (auto &op : region.getOps()) {
    costs.push_back(EstimateCost(op));
  }
  // Sum the costs of individual ops in the region. This is not interspersed
  // with the computation of these costs for improved readability of the IR and
  // the remarks.
  Cost cumulative_cost = ZeroCost();
  auto emit_cumulative_cost = [&]() {
    ConditionallyEmitRemark(*region.getParentOp(), cumulative_cost);
  };
  emit_cumulative_cost();
  for (auto &cost : costs) {
    cumulative_cost += cost;
    emit_cumulative_cost();
  }
  return cumulative_cost;
}

Value CostModel::CostToNanoseconds(Cost cost) {
  // Assume that RAM throughput is 16 GB/s and that CPU runs at 3 GHz.
  auto ram_runtime = builder_.create<arith::DivUIOp>(
      cost.ram(), builder_.create<arith::ConstantIndexOp>(16));
  auto cpu_runtime = builder_.create<arith::DivUIOp>(
      cost.cpu(), builder_.create<arith::ConstantIndexOp>(3));
  auto max = [&](Value x, Value y) {
    return builder_.create<arith::MaxSIOp>(x, y);
  };
  return max(max(ram_runtime, cpu_runtime),
             builder_.create<arith::ConstantIndexOp>(1));
}

}  // namespace

void CostDrivenAsyncParallelForPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  mlir::RewritePatternSet patterns(ctx);

  mlir::async::populateAsyncParallelForPatterns(
      patterns, async_dispatch_, num_worker_threads_,
      [&](ImplicitLocOpBuilder builder, scf::ParallelOp op) -> Value {
        if (legacy_behavior_) {
          return (Value)builder.create<arith::ConstantIndexOp>(16 * 1024);
        }
        CostModel costModel(builder, *op, true);
        return builder.create<arith::CeilDivSIOp>(
            builder.create<arith::ConstantIndexOp>(512 * 1024),
            costModel.CostToNanoseconds(
                costModel.EstimateCost(op.getLoopBody())));
      });

  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> CreateCostDrivenAsyncParallelForPass() {
  return std::make_unique<CostDrivenAsyncParallelForPass>();
}

std::unique_ptr<mlir::Pass> CreateCostDrivenAsyncParallelForPass(
    bool async_dispatch, int32_t num_worker_threads, bool legacy_behavior) {
  return std::make_unique<CostDrivenAsyncParallelForPass>(
      async_dispatch, num_worker_threads, legacy_behavior);
}

}  // namespace jitrt
}  // namespace tfrt
