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

#include <iterator>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/jitrt/conversion/rt_passes.h"
#include "tfrt/jitrt/opdefs/rt_ops.h"
#include "tfrt/jitrt/types.h"

namespace tfrt {
namespace jitrt {
namespace {

using mlir::Attribute;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::failure;
using mlir::FailureOr;
using mlir::FunctionType;
using mlir::ImplicitLocOpBuilder;
using mlir::IntegerType;
using mlir::LLVMTypeConverter;
using mlir::Location;
using mlir::LogicalResult;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::OperationPass;
using mlir::RewritePatternSet;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::TypeConverter;
using mlir::TypeID;
using mlir::TypeRange;
using mlir::UnrealizedConversionCastOp;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::ConstantOp;
using mlir::func::CallOp;
using mlir::func::FuncOp;

namespace LLVM = mlir::LLVM;

#define GEN_PASS_CLASSES
#include "tfrt/jitrt/conversion/rt_gen_passes.h.inc"

//===----------------------------------------------------------------------===//
// Runtime C API declaration (see runtime.h header file).
//===----------------------------------------------------------------------===//

static constexpr const char *kGetResultStorage = "runtimeGetResultStorage";
static constexpr const char *kSetError = "runtimeSetError";
static constexpr const char *kCustomCall = "runtimeCustomCall";

struct RuntimeAPI {
  static LLVM::LLVMPointerType OpaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMPointerType CustomCallArgumentsType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(RuntimeAPI::OpaquePointerType(ctx));
  }

  static FunctionType GetResultStorageFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto i64 = IntegerType::get(ctx, 64);
    auto storage = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, i64}, {storage});
  }

  static FunctionType SetErrorFunctionType(MLIRContext *ctx) {
    auto kernel_context = OpaquePointerType(ctx);
    auto error_msg = OpaquePointerType(ctx);
    return FunctionType::get(ctx, {kernel_context, error_msg}, {});
  }

  static FunctionType CustomCallFunctionType(MLIRContext *ctx) {
    auto callee = OpaquePointerType(ctx);
    auto args = CustomCallArgumentsType(ctx);
    auto i1 = IntegerType::get(ctx, 1);
    return FunctionType::get(ctx, {callee, args}, {i1});
  }
};

// Adds Runtime C API declarations to the module.
static void AddRuntimeApiDeclarations(ModuleOp module) {
  auto b = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  auto addDecl = [&](StringRef name, FunctionType type) {
    if (module.lookupSymbol(name)) return;
    b.create<FuncOp>(name, type).setPrivate();
  };

  MLIRContext *ctx = module.getContext();
  addDecl(kGetResultStorage, RuntimeAPI::GetResultStorageFunctionType(ctx));
  addDecl(kSetError, RuntimeAPI::SetErrorFunctionType(ctx));
  addDecl(kCustomCall, RuntimeAPI::CustomCallFunctionType(ctx));
}

// -------------------------------------------------------------------------- //

class RuntimeTypeConverter : public TypeConverter {
 public:
  RuntimeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(ConvertKernelContextType);
    addConversion(ConvertStatusType);
  }

  static llvm::Optional<Type> ConvertKernelContextType(KernelContextType type) {
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  }

  static llvm::Optional<Type> ConvertStatusType(StatusType type) {
    return IntegerType::get(type.getContext(), 1);
  }
};

// -------------------------------------------------------------------------- //

// Creates a global string constant in the given module, and returns a value
// corresponding to a pointer to the null terminated string.
static Value CreateGlobalStrCst(ModuleOp module, ImplicitLocOpBuilder &builder,
                                std::string str, StringRef prefix) {
  // Helper to create unique names for string constants.
  int unique_counter = 0;

  mlir::SymbolTable sym_table(module);
  auto sym_name = [&]() -> std::string {
    std::string str = prefix.str();
    while (sym_table.lookup(str))
      str = llvm::formatv("{0}_{1}", prefix, unique_counter++);
    return str;
  };

  // Create a string reference that captures the null terminator.
  StringRef ref(str.data(), str.size() + 1);
  auto str_type = LLVM::LLVMArrayType::get(builder.getI8Type(), ref.size());

  // Create string constant at the start of the module.
  auto str_constant = [&]() {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    return builder.create<LLVM::GlobalOp>(
        str_type, /*isConstant=*/true, LLVM::Linkage::Internal, sym_name(),
        StringAttr::get(module->getContext(), ref));
  }();

  // Get the pointer to the string constant that we'll pass to the runtime.
  auto str_addr = builder.create<LLVM::AddressOfOp>(
      LLVM::LLVMPointerType::get(str_type), str_constant.getSymName());
  auto str_ptr = builder.create<LLVM::BitcastOp>(
      LLVM::LLVMPointerType::get(builder.getI8Type()), str_addr);

  return str_ptr;
}

//===----------------------------------------------------------------------===//
// Convert rt.set_output to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetOutputOpLowering : public OpConversionPattern<SetOutputOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetOutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto kernel_context = adaptor.ctx();
    auto index = rewriter.create<ConstantOp>(loc, adaptor.indexAttr());

    // Get a pointer to the result value storage from the runtime.
    auto result_ptr_ty = RuntimeAPI::OpaquePointerType(rewriter.getContext());
    auto result_ptr = rewriter.create<CallOp>(
        loc, kGetResultStorage, TypeRange(result_ptr_ty),
        ValueRange({kernel_context, index}));

    // Cast from i8* to the LLVM pointer type to store the result.
    auto stored_type = getTypeConverter()->convertType(op.value().getType());
    if (!stored_type)
      return rewriter.notifyMatchFailure(
          op, "failed to convert output type to LLVM type");

    auto casted_result_ptr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(stored_type), result_ptr.getResult(0));

    // Store the output value into the result value storage.
    auto value = adaptor.value();
    rewriter.create<LLVM::StoreOp>(loc, value, casted_result_ptr.getResult());

    // Erase the original runtime operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.set_error to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class SetErrorOpLowering : public OpConversionPattern<SetErrorOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetErrorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Get the error message (pointer to a null terminated string).
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto err_ptr = CreateGlobalStrCst(module, builder, op.error().str(),
                                      "__assert_failed");

    // Call runtime API to report the error.
    auto kernel_context = adaptor.ctx();
    rewriter.replaceOpWithNewOp<CallOp>(op, kSetError, TypeRange(),
                                        ValueRange({kernel_context, err_ptr}));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.is_ok to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

class IsOkOpLowering : public OpConversionPattern<IsOkOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IsOkOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Just pass through the converted operand.
    rewriter.replaceOp(op, adaptor.status());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert rt.custom_call to the corresponding runtime API call.
//===----------------------------------------------------------------------===//

// Arguments to the custom call API intrinsic encoded as an array of opaque
// pointers and at the runtime side available as `void**`. Runtime decodes
// opaque pointers to the C++ data structures (see jitrt/custom_call.h), and
// passes them to the registered callback. Argument encoding/decoding must be
// compatible, otherwise it's very easy to get a segfault because of an illegal
// memory access.

// -------------------------------------------------------------------------- //
// Helper functions for packing attributes and values on the stack.

static TypeID RuntimeTypeId(Type type) {
  if (type.isF32()) return TypeID::get<float>();
  assert(false && "unsupported type");
  return {};
}

// Packs TypeID on the stack. Returns `!llvm.ptr<i64>`.
static Value PackTypeId(ImplicitLocOpBuilder &b, TypeID type_id) {
  Type ptr = LLVM::LLVMPointerType::get(b.getI64Type());
  Value one = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(ptr, one, 0);

  Value encoded_type_id = b.create<ConstantOp>(b.getI64IntegerAttr(
      reinterpret_cast<std::uintptr_t>(type_id.getAsOpaquePointer())));
  b.create<LLVM::StoreOp>(encoded_type_id, mem);

  return mem;
}

// Packs attribute on the stack. Returns `!llvm.ptr<AttrType>`.
static Value PackAttribute(ImplicitLocOpBuilder &b, Attribute value) {
  Type ptr = LLVM::LLVMPointerType::get(value.getType());
  Value one = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(ptr, one, 0);
  b.create<LLVM::StoreOp>(b.create<ConstantOp>(value), mem);

  return mem;
}

// Packs value on the stack. Returns `!llvm.ptr<ValueType>`.
static Value PackValue(ImplicitLocOpBuilder &b, Value value) {
  Type ptr = LLVM::LLVMPointerType::get(value.getType());
  Value one = b.create<ConstantOp>(b.getI32IntegerAttr(1));
  Value mem = b.create<LLVM::AllocaOp>(ptr, one, 0);
  b.create<LLVM::StoreOp>(value, mem);

  return mem;
}

// Stores pointer to packed argument or attribute into the allocated storage.
static void StoreOpaquePtr(ImplicitLocOpBuilder &b, Value ptr, Value alloca,
                           int64_t offset) {
  auto args_ptr_ty = RuntimeAPI::OpaquePointerType(b.getContext());
  Value bitcasted = b.createOrFold<LLVM::BitcastOp>(args_ptr_ty, ptr);
  Value idx = b.create<ConstantOp>(b.getI32IntegerAttr(offset));
  Value gep = b.create<LLVM::GEPOp>(alloca.getType(), alloca, ValueRange(idx));
  b.create<LLVM::StoreOp>(bitcasted, gep);
}

// -------------------------------------------------------------------------- //

// Encodes argument into stack allocated storage according to the ABI.
class CustomCallArgEncoding {
 public:
  struct Encoded {
    Value type_id;  // !llvm.ptr<i64>
    Value value;    // !llvm.ptr<ArgType>
  };

  virtual ~CustomCallArgEncoding() = default;

  virtual FailureOr<Encoded> Encode(ImplicitLocOpBuilder &b, Value value,
                                    Value converted) const = 0;
};

// Encodes MemRef operands according to the MemrefDesc ABI.
class MemrefArgEncoding : public CustomCallArgEncoding {
 public:
  FailureOr<Encoded> Encode(ImplicitLocOpBuilder &b, Value value,
                            Value converted) const override {
    auto memref_type = value.getType().cast<MemRefType>();

    Encoded encoded;
    encoded.type_id = PackTypeId(b, TypeID::get<MemrefDesc>());
    encoded.value = PackValue(b, EncodeMemRef(b, memref_type, converted));

    return encoded;
  }

 private:
  // Encodes memref as the LLVM structure value: type id (i64), rank (i64) and a
  // pointer to the strided memref descriptor (ptr<i8>).
  Value EncodeMemRef(ImplicitLocOpBuilder &b, MemRefType memref_ty,
                     Value descriptor) const {
    MLIRContext *ctx = b.getContext();

    // Encoded memref type: !llvm.struct<(i64, i64, ptr<i8>)>.
    Type i64 = b.getI64Type();
    Type ptr = LLVM::LLVMPointerType::get(b.getI8Type());
    Type type = LLVM::LLVMStructType::getLiteral(ctx, {i64, i64, ptr});

    // Create values for filling encoded memref struct.
    Value type_id = b.create<ConstantOp>(
        b.getI64IntegerAttr(reinterpret_cast<std::uintptr_t>(
            RuntimeTypeId(memref_ty.getElementType()).getAsOpaquePointer())));
    Value rank = b.create<ConstantOp>(b.getI64IntegerAttr(memref_ty.getRank()));
    Value desc = b.create<LLVM::BitcastOp>(ptr, PackValue(b, descriptor));

    auto offset = [&](int64_t i) { return b.getI64ArrayAttr(i); };

    // Create undef value for encoded memref.
    Value memref = b.create<LLVM::UndefOp>(type);
    memref = b.create<LLVM::InsertValueOp>(type, memref, type_id, offset(0));
    memref = b.create<LLVM::InsertValueOp>(type, memref, rank, offset(1));
    memref = b.create<LLVM::InsertValueOp>(type, memref, desc, offset(2));

    return memref;
  }
};

// TODO(ezhulenev): Support dynamic encoding registration for arguments.
static FailureOr<CustomCallArgEncoding::Encoded> EncodeArgument(
    ImplicitLocOpBuilder &b, std::tuple<Value, Value> value_and_converted) {
  Value value = std::get<0>(value_and_converted);
  Value converted = std::get<1>(value_and_converted);

  if (value.getType().isa<MemRefType>())
    return MemrefArgEncoding().Encode(b, value, converted);

  return failure();
}

static FailureOr<Value> EncodeArguments(ImplicitLocOpBuilder &b,
                                        ValueRange operands,
                                        ValueRange converted) {
  llvm::SmallVector<CustomCallArgEncoding::Encoded> encoded;

  // Encode all arguments as a set of pointers.
  for (auto tuple : llvm::zip(operands, converted)) {
    auto encoded_arg = EncodeArgument(b, tuple);
    if (failed(encoded_arg)) return failure();
    encoded.push_back(*encoded_arg);
  }

  // In addition to encoded arguments we store the number of arguments.
  int32_t args_size = 1 + encoded.size() * 2;

  // Allocate storage for passing endoded attributes.
  auto args_type = RuntimeAPI::CustomCallArgumentsType(b.getContext());
  auto alloca_size = b.create<ConstantOp>(b.getI32IntegerAttr(args_size));
  Value alloca = b.create<LLVM::AllocaOp>(args_type, alloca_size, 0);

  // Store the number of encoded arguments.
  Value n_attr = PackAttribute(b, b.getI64IntegerAttr(encoded.size()));
  StoreOpaquePtr(b, n_attr, alloca, 0);

  // Store encoded arguments into the allocated storage.
  for (auto &pair : llvm::enumerate(encoded)) {
    CustomCallArgEncoding::Encoded encoded = pair.value();
    int64_t offset = 1 + pair.index() * 2;

    StoreOpaquePtr(b, encoded.type_id, alloca, offset + 0);
    StoreOpaquePtr(b, encoded.value, alloca, offset + 1);
  }

  return alloca;
}

class CustomCallOpLowering : public OpConversionPattern<CustomCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Get the custom call target (pointer to a null terminated string).
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto callee = CreateGlobalStrCst(module, b, op.callee().str(),
                                     "__rt_custom_call_callee");

    // Encode operation arguments as a runtime API arguments.
    auto args = EncodeArguments(b, op->getOperands(), adaptor.getOperands());
    if (failed(args)) return op.emitOpError() << "failed to encode arguments";

    // Call runtime API to call the custom call target.
    auto i1 = rewriter.getI1Type();
    rewriter.replaceOpWithNewOp<CallOp>(op, kCustomCall, TypeRange(i1),
                                        ValueRange({callee, *args}));

    return success();
  }
};

// -------------------------------------------------------------------------- //

class ConvertRuntimeToLLVMPass
    : public ConvertRuntimeToLLVMPassBase<ConvertRuntimeToLLVMPass> {
  void runOnOperation() override;
};

void ConvertRuntimeToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  // Add declarations for the runtime API functions.
  AddRuntimeApiDeclarations(module);

  RuntimeTypeConverter converter;
  RewritePatternSet patterns(ctx);

  // We use conversion to LLVM type to lower all runtime operands to LLVM types.
  LLVMTypeConverter llvm_converter(ctx);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertKernelContextType);
  llvm_converter.addConversion(RuntimeTypeConverter::ConvertStatusType);

  // Lower from the runtime operations to the runtime API function calls.
  patterns.add<SetOutputOpLowering, SetErrorOpLowering, IsOkOpLowering,
               CustomCallOpLowering>(llvm_converter, ctx);

  // Convert function signatures and call sites.
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);

  // Set up conversion target to rewrite all runtime operations.
  ConversionTarget target(*ctx);
  target.addIllegalDialect<RuntimeDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ConstantOp, UnrealizedConversionCastOp, CallOp>();

  // Add dynamic legality constraints to apply conversions defined above.
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertRuntimeToLLVMPass() {
  return std::make_unique<ConvertRuntimeToLLVMPass>();
}

}  // namespace jitrt
}  // namespace tfrt
