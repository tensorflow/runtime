// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements the main entrypoints for the MLIRToBEF library.
// The converter is implemented in three phases.  The first phase identifies all
// of the strings and attributes that need to be emitted to the string/attribute
// pool.  The second phase optimizes and emits the strings and attributes to
// the file and remembers their offsets.  The third phase emits all of the
// regions in the MLIR program.
//
// MLIR ops are converted to kernel info and stored in BEF. So the term "op" is
// used in MLIR related code, and "kernel" used in BEF related code.
//===----------------------------------------------------------------------===//

#include "tfrt/bef_converter/mlir_to_bef.h"

#include <cstring>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/compiler/stream_analysis.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/traits.h"
#include "tfrt/core_runtime/opdefs/types.h"
#include "tfrt/host_context/debug_info.h"
#include "tfrt/support/aligned_buffer.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

#ifdef DEBUG_MLIR_TO_BEF
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

namespace tfrt {

namespace {
// This is a simple enum used to indicate success or failure in a more
// structured way than a simple bool.
enum class LogicalResult { Success, Failure };

// kInvalidIndex is used when a type or string cannot be found and a dummy
// index is required.
constexpr unsigned kInvalidIndex = 0xFFFF;

}  // namespace

/// Classify this attribute, so the rest of the code can know if it gets
/// special treatment.
static SpecialAttribute ClassifyAttribute(string_view attr_name) {
  return llvm::StringSwitch<SpecialAttribute>(attr_name)
      .Case("bef.nonstrict", SpecialAttribute::kNonStrict)
      .Default(SpecialAttribute::kUnknown);
}

static DType::Kind EncodeIntegerTypeAttribute(mlir::IntegerType integer_type) {
  if (integer_type.isUnsigned()) {
    switch (integer_type.getWidth()) {
      case 8:
        return DType::UI8;
      case 16:
        return DType::UI16;
      case 32:
        return DType::UI32;
      case 64:
        return DType::UI64;
    }
  } else {
    switch (integer_type.getWidth()) {
      case 1:
        return DType::I1;
      case 8:
        return DType::I8;
      case 16:
        return DType::I16;
      case 32:
        return DType::I32;
      case 64:
        return DType::I64;
    }
  }

  llvm_unreachable("unknown integer type width.");
}

static DType::Kind EncodeFloatTypeAttribute(mlir::FloatType float_type) {
  if (float_type.isBF16()) return DType::BF16;
  if (float_type.isF16()) return DType::F16;
  if (float_type.isF32()) return DType::F32;
  if (float_type.isF64()) return DType::F64;

  llvm_unreachable("unknown float type width.");
}

static DType::Kind EncodeComplexTypeAttribute(mlir::ComplexType complex_type) {
  auto element_type = complex_type.getElementType();

  if (element_type.isF32()) return DType::Complex64;
  if (element_type.isF64()) return DType::Complex128;

  llvm_unreachable("unknown complex type width.");
}

static DType::Kind ConvertMLIRDataTypeToTFRTDType(mlir::Type type) {
  if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    return EncodeIntegerTypeAttribute(integer_type);
  }

  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return EncodeFloatTypeAttribute(float_type);
  }

  if (auto string_type = type.dyn_cast<corert::StringType>()) {
    return DType::String;
  }

  if (auto resource_type = type.dyn_cast<corert::ResourceType>()) {
    return DType::Resource;
  }

  if (auto variant_type = type.dyn_cast<corert::VariantType>()) {
    return DType::Variant;
  }

  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    return EncodeComplexTypeAttribute(complex_type);
  }

  llvm_unreachable("unknown type attribute");
}

// Return the kind of this attribute. If it is an array attribute, elements of
// it are checked recursively, and if any element is unsupported,
// BEFAttributeType::Unsupported will be returned.
static BEFAttributeType GetBEFAttributeType(mlir::Attribute attr) {
  // We support 1-bit (stored as 1 byte in BEF), 32-bit, and 64-bit
  // integers.
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
    auto int_type = int_attr.getType().cast<mlir::IntegerType>();
    if (int_type.isUnsigned()) {
      switch (int_type.getWidth()) {
        case 8:
          return static_cast<BEFAttributeType>(DType::UI8);
        case 16:
          return static_cast<BEFAttributeType>(DType::UI16);
        case 32:
          return static_cast<BEFAttributeType>(DType::UI32);
        case 64:
          return static_cast<BEFAttributeType>(DType::UI64);
      }
    } else {
      switch (int_type.getWidth()) {
        case 1:
          return static_cast<BEFAttributeType>(DType::I1);
        case 8:
          return static_cast<BEFAttributeType>(DType::I8);
        case 16:
          return static_cast<BEFAttributeType>(DType::I16);
        case 32:
          return static_cast<BEFAttributeType>(DType::I32);
        case 64:
          return static_cast<BEFAttributeType>(DType::I64);
      }
    }
  }

  // We support BF16, F16, F32 and F64 floats.
  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>()) {
    if (float_attr.getType().isBF16())
      return static_cast<BEFAttributeType>(DType::BF16);
    if (float_attr.getType().isF16())
      return static_cast<BEFAttributeType>(DType::F16);
    if (float_attr.getType().isF32())
      return static_cast<BEFAttributeType>(DType::F32);
    if (float_attr.getType().isF64())
      return static_cast<BEFAttributeType>(DType::F64);
  }

  // We support string attributes.
  if (attr.isa<mlir::StringAttr>())
    return static_cast<BEFAttributeType>(DType::String);

  // We support i1, i8, i16, i32, i64, ui8, ui16, ui32, ui64, bf16, f16, f32,
  //  f64, complex64, complex128, string, resource and variant type attributes.
  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    auto type = type_attr.getValue();
    if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
        type.isInteger(32) || type.isInteger(64) || type.isBF16() ||
        type.isF16() || type.isF32() || type.isF64() ||
        type.isa<corert::StringType>() || type.isa<corert::ResourceType>() ||
        type.isa<corert::VariantType>())
      return BEFAttributeType::kType;

    if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
      auto element_type = complex_type.getElementType();
      if (element_type.isF32() || element_type.isF64())
        return BEFAttributeType::kType;
    }
  }

  // We support corert.shape attributes
  if (attr.isa<tfrt::corert::ShapeAttr>()) {
    return BEFAttributeType::kShape;
  }

  // We support dense attributes.
  if (auto dense_elements_attr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    auto element_type = ConvertMLIRDataTypeToTFRTDType(
        dense_elements_attr.getType().getElementType());
    if (element_type == DType::Unsupported)
      return BEFAttributeType::kUnsupported;

    return GetDenseAttributeType(element_type);
  }

  // We support arrays of supported attribute values.
  if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    if (array_attr.empty()) {
      return BEFAttributeType::kEmptyArray;
    }

    auto first_attr_type = GetBEFAttributeType(*array_attr.begin());

    // Only fixed attributes can be included in an array.
    bool is_array = IsFixedAttribute(first_attr_type);

    for (auto elt : array_attr) {
      auto attr_type = GetBEFAttributeType(elt);
      if (attr_type == BEFAttributeType::kUnsupported)
        return BEFAttributeType::kUnsupported;

      // Arrays requires all elements have the same type and the size.
      if (attr_type != first_attr_type) {
        is_array = false;
        break;
      }
    }

    if (is_array) return GetArrayAttributeType(first_attr_type);

    return BEFAttributeType::kAggregate;
  }

  // We support symbol references to compiled functions.
  if (auto symbol_ref_attr = attr.dyn_cast<mlir::SymbolRefAttr>()) {
    return BEFAttributeType::kSymbolRef;
  }

  return BEFAttributeType::kUnsupported;
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a kernel, even in recursive positions.
static bool IsSupportedAttributeValue(mlir::Attribute attr) {
  return GetBEFAttributeType(attr) != BEFAttributeType::kUnsupported;
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a kernel.
static bool IsSupportedAttribute(mlir::Attribute attr) {
  // We support references to functions.
  if (attr.isa<mlir::SymbolRefAttr>()) return true;

  return IsSupportedAttributeValue(attr);
}

// The "tfrt.return" kernel gets special case handling in BEF files.
static bool IsReturn(mlir::Operation* op) {
  // TODO(tfrt-dev): Use C++ op type here instead of relying on string
  // comparing.
  return op->getName().getStringRef() == "tfrt.return";
}

static bool IsNativeFunc(mlir::FuncOp op) {
  return !!op->getAttr("tfrt.native");
}

static bool IsSyncFunc(mlir::FuncOp op) { return !!op->getAttr("tfrt.sync"); }

static bool IsCompiledModule(mlir::ModuleOp op) {
  return !!op->getAttr("tfrt.compiled");
}

// Returs true if the operation is inside the compiled module or the compiled
// module itself. Compiled modules passed to BEF kernels as serialized MLIR
// blobs, and we do not need to encode any operations in them as BEF.
static bool IsInCompiledModule(mlir::Operation* op) {
  mlir::ModuleOp parent_module = dyn_cast<mlir::ModuleOp>(op);
  if (!parent_module) parent_module = op->getParentOfType<mlir::ModuleOp>();

  while (parent_module) {
    if (IsCompiledModule(parent_module)) return true;
    parent_module = parent_module->getParentOfType<mlir::ModuleOp>();
  }

  return false;
}

static mlir::FunctionType GetRegionFunctionType(mlir::Region* region) {
  // Emit information about the type of the function.
  auto& block = region->front();

  // Arguments.
  SmallVector<mlir::Type, 4> inputs;
  for (auto arg : block.getArguments()) inputs.push_back(arg.getType());

  // Results.
  // MLIR Regions don't have an easy way to identify results in regions, so
  // we just hard code the "tfrt.return" instruction.
  auto& last_op = block.back();
  assert(IsReturn(&last_op));

  SmallVector<mlir::Type, 4> results;
  for (auto op : last_op.getOperands()) results.push_back(op.getType());

  return mlir::FunctionType::get(region->getContext(), inputs, results);
}

static bool IsOpAttrsTyped(mlir::Operation* op) {
  // TODO(tf-runtime-team): Define corert.execute_crt_op in ODS with
  // TypedAttributeTrait.
  return op->hasTrait<mlir::OpTrait::tfrt::corert::TypedAttributeTrait>() ||
         op->getName().getStringRef() == "corert.execute_crt_op";
}

//===----------------------------------------------------------------------===//
// EntityTable
//===----------------------------------------------------------------------===//

namespace {

// This table keeps track of the interesting entities (attributes, types, other
// strings) that we care about.  This is built in the first pass.
struct EntityTable {
  // Uniquing set of attributes we need to emit, kept in order so we always
  // produce a determinstic output file.
  llvm::SetVector<mlir::Attribute> attributes;
  llvm::SetVector<mlir::Attribute> typed_attributes;

  // Uniquing set of the kernels that we need to emit.
  std::vector<string_view> kernels;
  llvm::StringMap<unsigned> kernel_ids;

  struct FunctionEntry {
    FunctionEntry(string_view name, mlir::FunctionType type, FunctionKind kind,
                  mlir::Region* region = nullptr)
        : name(name), type(type), kind(kind), region(region) {}

    string_view name;
    mlir::FunctionType type;
    FunctionKind kind;

    // If region is nullptr, then it is an external function (eg. a native
    // function).
    mlir::Region* region = nullptr;

    bool IsNative() const { return kind == FunctionKind::kNativeFunction; }

    bool IsSync() const { return kind == FunctionKind::kSyncBEFFunction; }
  };

  // List of functions that we need to emit, along with a name if they came
  // from a top level function.
  std::vector<FunctionEntry> functions;
  llvm::DenseMap<mlir::Region*, unsigned> region_function_ids;
  llvm::StringMap<unsigned> named_function_ids;

  // Types we've seen so far.
  std::vector<mlir::Type> types;
  llvm::DenseMap<mlir::Type, unsigned> type_ids;

  // All of the strings we need to emit the BEF file, an unordered
  // collection that we sort before emitting.  We use StringMap here instead of
  // StringSet because StringSet rejects empty strings for no apparent reason.
  llvm::StringMap<uint8_t> strings;

  // This is all of the filenames referred to by locations in the file.
  llvm::SmallVector<string_view, 4> location_filenames;
  llvm::StringMap<unsigned> location_filenames_index;

  // These are the locations for all operations within the file, the first
  // element of the tuple is a index into location_filenames, the second and
  // third are line/col information.
  typedef std::tuple<unsigned, unsigned, unsigned> LocationTuple;
  llvm::DenseMap<mlir::Operation*, LocationTuple> location_positions;

  llvm::DenseMap<mlir::Operation*, DebugInfoEntry> debug_info;

 public:
  LogicalResult Collect(mlir::ModuleOp module,
                        bool collect_attribute_types_and_names);
  ssize_t GetFunctionNamed(string_view name) const;

  void AddString(string_view string);
  void AddType(mlir::Type type);
  unsigned GetOptionalTypeIndex(mlir::Type type) const;
  unsigned GetTypeIndex(mlir::Type type) const;

  void AddNativeFunction(mlir::FuncOp op);
  LogicalResult AddFunction(mlir::Region* region, string_view name,
                            FunctionKind func_kind);
  unsigned GetFunctionID(const mlir::Region& region) const;

  void AddKernel(mlir::Operation* kernel);
  unsigned GetKernelID(mlir::Operation* kernel) const;

  void AddLocation(mlir::Operation* op);

  void AddDebugInfo(mlir::Operation* op);

  void AddAttributeType(mlir::Attribute attr);
};

}  // namespace

void EntityTable::AddString(string_view string) { strings[string] = 0; }

// Add a type to our table, checking it by pointer to reduce string
// conversions.
void EntityTable::AddType(mlir::Type type) {
  // Ignore the type if we've seen it before.
  assert(types.size() != kInvalidIndex);
  if (!type_ids.insert({type, types.size()}).second) return;
  types.push_back(type);

  // If it is new, remember the type name as a string.
  llvm::SmallVector<char, 64> result_str;
  llvm::raw_svector_ostream os(result_str);
  type.print(os);
  AddString(os.str());
}

unsigned EntityTable::GetOptionalTypeIndex(mlir::Type type) const {
  auto it = type_ids.find(type);
  if (it == type_ids.end()) return kInvalidIndex;
  return it->second;
}

unsigned EntityTable::GetTypeIndex(mlir::Type type) const {
  auto it = type_ids.find(type);
  assert(it != type_ids.end() && "unregistered type");
  return it->second;
}

void EntityTable::AddNativeFunction(mlir::FuncOp op) {
  auto function_type = op.getType();

  for (auto type : function_type.getInputs()) AddType(type);
  for (auto type : function_type.getResults()) AddType(type);

  auto name = op.getName();

  AddString(name);
  named_function_ids[name] = functions.size();
  functions.push_back(
      FunctionEntry(name, function_type, FunctionKind::kNativeFunction));
}

LogicalResult EntityTable::AddFunction(mlir::Region* region, string_view name,
                                       FunctionKind func_kind) {
  // Check to see if we support this region kind.
  if (!llvm::hasSingleElement(*region)) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to BEF files";
    return LogicalResult::Failure;
  }

  for (auto type : region->getArgumentTypes()) AddType(type);

  // Remember this function.
  AddString(name);
  region_function_ids[region] = functions.size();
  named_function_ids[name] = functions.size();
  functions.push_back(
      FunctionEntry(name, GetRegionFunctionType(region), func_kind, region));
  return LogicalResult::Success;
}

unsigned EntityTable::GetFunctionID(const mlir::Region& region) const {
  auto it = region_function_ids.find(&region);
  assert(it != region_function_ids.end() && "region not added to entity table");
  return it->second;
}

// Return the index of the specified function name, returning -1 if the
// function name cannot be found.
ssize_t EntityTable::GetFunctionNamed(string_view name) const {
  auto iter = named_function_ids.find(name);
  if (iter == named_function_ids.end()) return -1;
  return iter->second;
}

void EntityTable::AddKernel(mlir::Operation* kernel) {
  // Remember the kernel.
  if (!kernel_ids.insert({kernel->getName().getStringRef(), kernels.size()})
           .second)
    return;
  kernels.push_back(kernel->getName().getStringRef());

  // If we haven't seen it already, add it to the string table.
  AddString(kernel->getName().getStringRef());
}

unsigned EntityTable::GetKernelID(mlir::Operation* kernel) const {
  auto it = kernel_ids.find(kernel->getName().getStringRef());
  assert(it != kernel_ids.end() && "Unknown kernel");
  return it->second;
}

void EntityTable::AddDebugInfo(mlir::Operation* op) {
  auto debug_info_location = op->getLoc();

  // If the location is a FusedLoc, look for a NameLoc among its children.
  // TODO(b/180438663): Handle cases where there are multiple NameLoc.
  if (auto fused_loc = debug_info_location.dyn_cast<mlir::FusedLoc>()) {
    for (auto& location : fused_loc.getLocations()) {
      if (auto named_loc = location.dyn_cast<mlir::NameLoc>()) {
        debug_info_location = location;
        break;
      }
    }
  }

  // If the location is a CallSiteLoc, look whether the callee is a NameLoc.
  if (auto call_site = debug_info_location.dyn_cast<mlir::CallSiteLoc>()) {
    const auto& location = call_site.getCallee();
    if (auto named_loc = location.dyn_cast<mlir::NameLoc>()) {
      debug_info_location = location;
    }
  }

  if (auto named_loc = debug_info_location.dyn_cast<mlir::NameLoc>()) {
    DebugInfoEntry debug_info_entry = named_loc.getName().c_str();
    auto r = debug_info.try_emplace(op, debug_info_entry);
    assert(r.second);
    (void)r;
  }
}

void EntityTable::AddLocation(mlir::Operation* op) {
  auto file_line_col_location = op->getLoc();
  string_view filename = "";
  unsigned line = 0, col = 0;

  // If the location is a FusedLoc, look for a FileLineColLoc among its
  // children.
  // TODO(b/180438663): Handle cases where there are multiple FileLineColLoc.
  if (auto fused_loc = file_line_col_location.dyn_cast<mlir::FusedLoc>()) {
    for (auto& location : fused_loc.getLocations()) {
      if (auto loc = location.dyn_cast<mlir::FileLineColLoc>()) {
        file_line_col_location = loc;
        break;
      }
    }
  }

  if (auto loc = file_line_col_location.dyn_cast<mlir::FileLineColLoc>()) {
    filename = loc.getFilename();
    line = loc.getLine();
    col = loc.getColumn();
  }

  auto next_filename_index = location_filenames.size();
  auto it =
      location_filenames_index.insert({filename, next_filename_index}).first;
  if (it->second == next_filename_index) location_filenames.push_back(filename);

  auto r =
      location_positions.try_emplace(op, LocationTuple{it->second, line, col});
  assert(r.second);
  (void)r;
}

void EntityTable::AddAttributeType(mlir::Attribute attr) {
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
    AddType(int_attr.getType());
  }

  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>()) {
    AddType(float_attr.getType());
  }

  if (auto arr_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    for (auto attr : arr_attr.getValue()) {
      AddAttributeType(attr);
    }
  }
}

LogicalResult EntityTable::Collect(mlir::ModuleOp module,
                                   bool collect_attribute_types_and_names) {
  auto result = LogicalResult::Success;

  std::vector<std::pair<mlir::SymbolRefAttr, mlir::Location>> fn_attrs;

  module.walk(
      [&](mlir::Operation* op) {
        // Ignore the module itself, and a few specific other ops.
        if (op == module.getOperation()) return;

        // Ignore operations inside compiled modules. Symbol references into the
        // compiled modules passes to kernels as a compilation unit attribute.
        if (IsInCompiledModule(op)) return;

        // The return op gets special handling, ensure it is at the end of its
        // enclosing block.
        if (IsReturn(op)) {
          if (&op->getBlock()->back() != op) {
            op->emitError() << "return op must be at the end of its block";
            result = LogicalResult::Failure;
            return;
          }
          // Ignore it, return gets special handling.
          return;
        }

        AddLocation(op);
        AddDebugInfo(op);

        auto* cur_region = op->getParentRegion();

        // Notice the result and argument types of the ops.
        for (auto result : op->getResults()) AddType(result.getType());

        for (auto operand : op->getOperands()) {
          // Verify that the operand is defined inside the current region.  We
          // don't support references to outer regions.
          if (operand.getParentRegion() != cur_region) {
            op->emitError()
                << "BEF executor only supports references to kernels within"
                << " the current region";
            result = LogicalResult::Failure;
            return;
          }
        }

        // We treat functions specially, putting them into the symbol table and
        // ignoring their attributes.
        if (auto fn = llvm::dyn_cast<mlir::FuncOp>(op)) {
          if (IsNativeFunc(fn)) {
            AddNativeFunction(fn);
          } else {
            if (fn.isExternal()) {
              fn.emitError() << "external functions are not allowed";
              result = LogicalResult::Failure;
              return;
            }

            // Verify that all functions end with a return to catch a common
            // error.
            auto& last_op = fn.front().back();
            if (!IsReturn(&last_op)) {
              last_op.emitError() << "all functions need to have a tfrt.return";
              result = LogicalResult::Failure;
              return;
            }

            if (IsSyncFunc(fn)) {
              llvm::SmallSetVector<mlir::Value, 4> return_operands;
              for (auto iter : llvm::enumerate(last_op.getOperands())) {
                auto index = iter.index();
                const auto& operand = iter.value();
                if (operand.isa<mlir::BlockArgument>()) {
                  last_op.emitError() << "return value " << index
                                      << " is an argument in a sync function";
                  result = LogicalResult::Failure;
                  return;
                }

                if (!return_operands.insert(operand)) {
                  last_op.emitError() << "return value " << index
                                      << " is duplicated in a sync function";
                  result = LogicalResult::Failure;
                  return;
                }
              }
            }

            auto func_kind = IsSyncFunc(fn) ? FunctionKind::kSyncBEFFunction
                                            : FunctionKind::kBEFFunction;
            if (AddFunction(&fn.getBody(), fn.getName(), func_kind) ==
                LogicalResult::Failure) {
              result = LogicalResult::Failure;
              return;
            }
          }
        } else {
          AddKernel(op);

          bool is_op_attrs_typed = IsOpAttrsTyped(op);

          // Keep track of any attributes used by this op.
          for (auto attr : op->getAttrs()) {
            // Skip cost attribute which is not used in runtime execution.
            //
            // TODO(tfrt-devs): Use attribute interface instead of hardcoding
            // here.
            if (attr.first == "_tfrt_cost") continue;

            // If this is a special attribute, ignore it.
            if (ClassifyAttribute(attr.first.strref()) !=
                SpecialAttribute::kUnknown)
              continue;

            // Check to make sure that this is a supported attribute, if not,
            // reject it.
            if (!IsSupportedAttribute(attr.second) &&
                result == LogicalResult::Success) {
              op->emitError() << "BEF files cannot encode the '" << attr.first
                              << "' attribute";
              result = LogicalResult::Failure;
              return;
            }

            // Returns a symbol ref to an executable operation (function that
            // needs to be converted to BEF). If the referenced symbol is inside
            // the compiled module returns None. All compiled operations will be
            // added to the attributes section as compilation units.
            auto bef_function_ref = [&]() -> Optional<mlir::SymbolRefAttr> {
              auto sym_attr = attr.second.dyn_cast<mlir::SymbolRefAttr>();
              if (!sym_attr) return llvm::None;

              // Check if the referenced symbol is in the compiled module.
              auto* module_op = module.getOperation();
              auto* sym_op =
                  mlir::SymbolTable::lookupSymbolIn(module_op, sym_attr);
              if (sym_op && IsInCompiledModule(sym_op)) return llvm::None;

              return sym_attr;
            };

            if (auto fn_attr = bef_function_ref()) {
              // Keep track of function attributes specially so we can diagnose
              // them.
              fn_attrs.push_back({*fn_attr, op->getLoc()});

            } else {
              if (collect_attribute_types_and_names) {
                // Add attribute names and types for attribute types section and
                // attribute names section. These will be ignored by executor.
                AddString(attr.first);
                AddAttributeType(attr.second);
              }

              // Skip collecting array of function attributes.
              auto array_attr = attr.second.dyn_cast<mlir::ArrayAttr>();
              if (array_attr) {
                if (!array_attr.empty() &&
                    array_attr.begin()->dyn_cast<mlir::SymbolRefAttr>()) {
                  continue;
                }
              }

              // We ignore the name of attributes, they just get passed as
              // arguments.
              if (is_op_attrs_typed)
                typed_attributes.insert(attr.second);
              else
                attributes.insert(attr.second);
            }
          }

          // Keep add any regions used by this op as BEF functions.
          for (auto& region : op->getRegions()) {
            if (AddFunction(&region, "", FunctionKind::kBEFFunction) ==
                LogicalResult::Failure) {
              result = LogicalResult::Failure;
              return;
            }
          }
        }
      });

  // If we're successful, check to make sure that all functions that should be
  // translated to BEF can be resolved.
  if (result == LogicalResult::Success) {
    for (auto attr_and_loc : fn_attrs) {
      if (GetFunctionNamed(attr_and_loc.first.getRootReference()) == -1) {
        mlir::emitError(attr_and_loc.second)
            << "function " << attr_and_loc.first << " not defined";
        return LogicalResult::Failure;
      }
    }
  }

  return result;
}

namespace {

// This table is built in the second pass, keeping track of the indices that
// each entity is assigned.
class EntityIndex {
 public:
  unsigned GetOptionalStringOffset(string_view str) const {
    auto it = strings_.find(str);
    if (it == strings_.end()) return kInvalidIndex;
    return it->second;
  }

  unsigned GetStringOffset(string_view str) const {
    auto it = strings_.find(str);
    assert(it != strings_.end() &&
           "String didn't get added to the entity collection");
    return it->second;
  }

  void AddString(string_view str, unsigned index) {
    assert(!strings_.count(str) && "string already in index");
    assert(index != kInvalidIndex);
    strings_.insert({str, index});
  }

  unsigned GetTypedAttributeOffset(mlir::Attribute attribute) const {
    auto it = typed_attribute_offsets_.find(attribute);
    assert(it != typed_attribute_offsets_.end() &&
           "typed attribute didn't get added to the entity collection");
    return it->second;
  }

  unsigned GetAttributeOffset(mlir::Attribute attribute) const {
    auto it = attribute_offsets_.find(attribute);
    assert(it != attribute_offsets_.end() &&
           "attribute didn't get added to the entity collection");
    return it->second;
  }

  void AddAttributeOffset(mlir::Attribute attribute, unsigned offset) {
    assert(!attribute_offsets_.count(attribute) &&
           "attribute already in index");
    attribute_offsets_.insert({attribute, offset});
  }

  void AddTypedAttributeOffset(mlir::Attribute attribute, unsigned offset) {
    assert(!typed_attribute_offsets_.count(attribute) &&
           "attribute already in index");
    typed_attribute_offsets_.insert({attribute, offset});
  }

  struct FunctionIndexEntry {
    size_t name_offset;
    size_t function_offset;
    mlir::FunctionType type;
    FunctionKind kind;
  };

  void AddFunction(string_view name, unsigned offset, mlir::FunctionType type,
                   FunctionKind kind) {
    function_index_.push_back({GetStringOffset(name), offset, type, kind});
  }

  llvm::ArrayRef<FunctionIndexEntry> GetFunctionIndex() const {
    return function_index_;
  }

  typedef EntityTable::LocationTuple LocationTuple;

  void AddLocationPosition(mlir::Operation* op, size_t offset) {
    location_position_offsets_[op] = offset;
  }

  size_t GetLocationPositionOffset(mlir::Operation* op) const {
    auto loc_it = location_position_offsets_.find(op);
    assert(loc_it != location_position_offsets_.end() && "unknown location");
    return loc_it->second;
  }

  void AddDebugInfoOffset(mlir::Operation* op, size_t offset) {
    debug_info_offset_[op] = offset;
  }

  llvm::Optional<size_t> GetDebugInfoOffset(mlir::Operation* op) const {
    auto loc_it = debug_info_offset_.find(op);
    if (loc_it != debug_info_offset_.end()) {
      return loc_it->second;
    } else {
      return llvm::None;
    }
  }

 private:
  llvm::StringMap<unsigned> strings_;
  llvm::DenseMap<mlir::Attribute, unsigned> attribute_offsets_;
  llvm::DenseMap<mlir::Attribute, unsigned> typed_attribute_offsets_;

  // This follows the format of the FunctionIndex section, where the first
  // element is the offset of the name in the string section, the second is the
  // offset into the function table.
  std::vector<FunctionIndexEntry> function_index_;

  // This is the location of the offsets into the section.
  llvm::DenseMap<mlir::Operation*, size_t> location_position_offsets_;

  // This is the offset of associated entry in the debug info section (if any)
  // kNoDebugInfoEntryOffset represents no entry.
  llvm::DenseMap<mlir::Operation*, DebugInfoOffset> debug_info_offset_;
};

}  // namespace

namespace {

// This is the emitter that builds a BEF into an std::vector.  This class
// contains the primitive routines used by the various specific emitters.  In
// addition to collecting the bytes contained in this piece of the BEF file,
// this tracks the alignment requirement of the contents.  If this is a
// subsection of the file, then the enclosing container is required to provide
// at least this alignment.
class BEFFileEmitter : public BEFEmitter {
 public:
  static constexpr uint32_t kDummyPseudoKernelCode = 0xABABABAB;
  static constexpr uint32_t kDummyPseudoKernelLocation = 0xCDCDCDCD;

  BEFFileEmitter() {}
  BEFFileEmitter(const BEFFileEmitter&) = delete;
  BEFFileEmitter& operator=(const BEFFileEmitter&) = delete;

  void EmitSection(BEFSectionID section_id,
                   llvm::ArrayRef<uint8_t> section_data,
                   unsigned alignment = 1) {
    // Section start with an identifier.
    result_.push_back(static_cast<uint8_t>(section_id));

    // LENGTH_AND_ALIGNMENT ::= (SECTION_LENGTH << 1) | (SECTION_ALIGNMENT_FLAG)
    const auto shifted_section_length = (section_data.size() << 1);
    bool length_emitted = false;
    if (alignment > 1) {
      auto offset = size() + GetSizeOfVbrInt(shifted_section_length);
      if (offset % alignment != 0) {
        // Emit section length with alignment constraint.
        EmitInt(shifted_section_length | 1);
        EmitByte(alignment);

        // Move up to the right alignment for the section data.
        EmitAlignment(alignment);

        // Mark that the section length has been emitted.
        length_emitted = true;
      }
    }

    if (!length_emitted) {
      // Emit section length without alignment constraint.
      EmitInt(shifted_section_length);
    }

    // Then have the payload data.
    EmitBytes(section_data);
  }

  void EmitSection(BEFSectionID section_id, const BEFFileEmitter& emitter) {
    EmitSection(section_id, emitter.result_, emitter.GetRequiredAlignment());
  }
};

constexpr uint32_t BEFFileEmitter::kDummyPseudoKernelCode;
constexpr uint32_t BEFFileEmitter::kDummyPseudoKernelLocation;

}  // namespace

// Resolve the properties of compilation units in the top level Module
// operation. Compilation units are functions in modules with `tfrt.compiled`
// attribute, which will be compiled at runtime by the BEF program, and passed
// as compilation unit attributes (serialized MLIR).
class BEFCompilationUnits {
 public:
  explicit BEFCompilationUnits(mlir::ModuleOp module) : module_(module) {}

  size_t SerializedSymbolSize(mlir::SymbolRefAttr symbol);
  size_t SerializedOperationSize(mlir::SymbolRefAttr symbol);

  ArrayRef<uint8_t> SerializedSymbolData(mlir::SymbolRefAttr symbol);
  ArrayRef<uint8_t> SerializedOperationData(mlir::SymbolRefAttr symbol);

 private:
  struct Serialized {
    size_t symbol_size;     // size of the serialized symbol name
    size_t operation_size;  // size of the serialized operation
    std::string data;       // symbol_ref + operation
  };

  const Serialized& Serialize(mlir::SymbolRefAttr symbol);

  mlir::ModuleOp module_;
  llvm::DenseMap<mlir::Attribute, Serialized> serialized_;
};

size_t BEFCompilationUnits::SerializedSymbolSize(mlir::SymbolRefAttr symbol) {
  return Serialize(symbol).symbol_size;
}

size_t BEFCompilationUnits::SerializedOperationSize(
    mlir::SymbolRefAttr symbol) {
  return Serialize(symbol).operation_size;
}

ArrayRef<uint8_t> BEFCompilationUnits::SerializedSymbolData(
    mlir::SymbolRefAttr symbol) {
  auto& serialized = Serialize(symbol);
  string_view str = serialized.data;
  return {reinterpret_cast<const uint8_t*>(str.data()), serialized.symbol_size};
}

ArrayRef<uint8_t> BEFCompilationUnits::SerializedOperationData(
    mlir::SymbolRefAttr symbol) {
  auto& serialized = Serialize(symbol);
  string_view str = serialized.data;
  return {reinterpret_cast<const uint8_t*>(str.data()) + serialized.symbol_size,
          serialized.operation_size};
}

const BEFCompilationUnits::Serialized& BEFCompilationUnits::Serialize(
    mlir::SymbolRefAttr symbol) {
  auto* op = mlir::SymbolTable::lookupSymbolIn(module_.getOperation(), symbol);
  assert(IsInCompiledModule(op));

  // Check if the referenced symbol already serialized.
  auto it = serialized_.find(symbol);
  if (it != serialized_.end()) return it->getSecond();

  // Serialize and keep compiled module that defines the symbol.
  auto parent_module = op->getParentOfType<mlir::ModuleOp>();
  assert(IsCompiledModule(parent_module));

  std::string str;
  llvm::raw_string_ostream os(str);

  // Print symbol names.
  os << symbol.getRootReference();
  for (auto nested_ref : symbol.getNestedReferences())
    os << nested_ref.getValue();

  size_t symbol_size = str.size();

  // Use generic form to print the module and improve BEF portability. The
  // pretty print is less stable from a syntax point of view.
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm();
  parent_module.print(os, flags);

  size_t operation_size = str.size() - symbol_size;

  Serialized serialized{symbol_size, operation_size, std::move(str)};
  auto inserted = serialized_.insert({symbol, std::move(serialized)});
  return inserted.first->getSecond();
}

// This is the emitter that builds a BEF into an std::vector.
class BEFModuleEmitter : public BEFFileEmitter {
 public:
  explicit BEFModuleEmitter(mlir::ModuleOp module) : module_(module) {}

  LogicalResult CollectEntities(bool collect_attribute_types_and_names) {
    return entities_.Collect(module_, collect_attribute_types_and_names);
  }

  void EmitLocationInfo();
  void EmitDebugInfo();
  void EmitStrings();
  void EmitAttributes(BEFFileEmitter* attribute_types);
  void EmitKernels();
  void EmitTypes();
  void EmitFunctions(BEFFileEmitter* attribute_names,
                     BEFFileEmitter* register_types);
  void EmitAttributeTypes(const BEFFileEmitter& attribute_types);
  void EmitAttributeNames(const BEFFileEmitter& attribute_names);
  void EmitRegisterTypes(const BEFFileEmitter& register_types);

 private:
  mlir::ModuleOp module_;
  EntityTable entities_;
  EntityIndex entity_index_;
};

void BEFModuleEmitter::EmitLocationInfo() {
  BEFFileEmitter filenames_section;
  for (auto filename : entities_.location_filenames) {
    filenames_section.EmitBytes(
        {reinterpret_cast<const uint8_t*>(filename.data()), filename.size()});
    // Emit a NUL terminator for the filename.
    filenames_section.EmitByte(0);
  }

  EmitSection(BEFSectionID::kLocationFilenames, filenames_section);

  // Emit each of the positions and remember the offsets within the section.
  BEFFileEmitter positions_section;
  for (auto iter : entities_.location_positions) {
    mlir::Operation* op = iter.first;
    auto position = iter.second;
    entity_index_.AddLocationPosition(op, positions_section.size());
    positions_section.EmitInt(std::get<0>(position));
    positions_section.EmitInt(std::get<1>(position));
    positions_section.EmitInt(std::get<2>(position));
  }

  EmitSection(BEFSectionID::kLocationPositions, positions_section);
}

void BEFModuleEmitter::EmitDebugInfo() {
  BEFFileEmitter debug_info_section;

  for (const auto& entry : entities_.debug_info) {
    auto& op = entry.getFirst();
    auto& debug_info = entry.getSecond();

    entity_index_.AddDebugInfoOffset(op, debug_info_section.size());
    debug_info_section.EmitBytes(
        {reinterpret_cast<const uint8_t*>(debug_info.data()),
         debug_info.size()});
    debug_info_section.EmitByte(0);
  }

  EmitSection(BEFSectionID::kDebugInfo, debug_info_section);
}

void BEFModuleEmitter::EmitStrings() {
  // We have an ordered collection of strings: sort them alphabetically to make
  // them stable.
  std::vector<string_view> strs_in_order;
  strs_in_order.reserve(entities_.strings.size());
  for (const auto& str : entities_.strings)
    strs_in_order.push_back(str.getKey());

  std::sort(strs_in_order.begin(), strs_in_order.end());

  // Now that we have all the strings in order, emit them and remember their
  // offsets in the string section.
  BEFFileEmitter string_section;
  for (const auto& entry : strs_in_order) {
    entity_index_.AddString(entry, string_section.size());
    string_section.EmitBytes(
        {reinterpret_cast<const uint8_t*>(entry.data()), entry.size()});
    // Emit a NUL terminator for the string.
    string_section.EmitByte(0);
  }

  EmitSection(BEFSectionID::kStrings, string_section);
}

// This emits attributes without any type or size information.
class BEFAttributeEmitter : public BEFEmitter {
 public:
  explicit BEFAttributeEmitter(BEFCompilationUnits& compilation_units)
      : compilation_units_(compilation_units) {}

  void EmitAttribute(mlir::Attribute attr);

  void EmitBoolAttribute(bool value);
  void EmitStandardAttribute(mlir::Attribute attr);
  void EmitStringAttribute(string_view value);
  void EmitTypeAttribute(mlir::TypeAttr type_attr);
  void EmitArrayAttribute(mlir::ArrayAttr array_attr);
  void EmitSymbolRefAttribute(mlir::SymbolRefAttr symbol_ref_attr);

  void EmitIntegerAttribute(const llvm::APInt& value);
  void EmitFloatAttribute(mlir::FloatAttr attr);

 private:
  BEFCompilationUnits& compilation_units_;
};

void BEFAttributeEmitter::EmitAttribute(mlir::Attribute attr) {
  assert(IsSupportedAttribute(attr) &&
         "EmitAttribute get an unsupported attribute");

  if (auto bool_attr = attr.dyn_cast<mlir::BoolAttr>()) {
    EmitBoolAttribute(bool_attr.getValue());
    return;
  }

  if (attr.isa<mlir::IntegerAttr, mlir::FloatAttr>()) {
    EmitStandardAttribute(attr);
    return;
  }

  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    EmitTypeAttribute(type_attr);
    return;
  }

  if (auto string_attr = attr.dyn_cast<mlir::StringAttr>()) {
    EmitStringAttribute(string_attr.getValue());
    return;
  }

  // We support arrays of attributes.
  if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    EmitArrayAttribute(array_attr);
    return;
  }

  if (auto symbol_ref_attr = attr.dyn_cast<mlir::SymbolRefAttr>()) {
    EmitSymbolRefAttribute(symbol_ref_attr);
    return;
  }

  llvm_unreachable("Unknown attribute");
}

void BEFAttributeEmitter::EmitBoolAttribute(bool value) {
  EmitByte(static_cast<uint8_t>(value));
}

void BEFAttributeEmitter::EmitStandardAttribute(mlir::Attribute attr) {
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
    EmitIntegerAttribute(int_attr.getValue());
    return;
  }

  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>()) {
    EmitFloatAttribute(float_attr);
    return;
  }

  llvm_unreachable("Unknown standard attribute");
}

void BEFAttributeEmitter::EmitIntegerAttribute(const llvm::APInt& value) {
  if (value.getBitWidth() == 1) {
    EmitByte(static_cast<uint8_t>(value.getLimitedValue()));
    return;
  }

  assert(value.getBitWidth() == 8 || value.getBitWidth() == 16 ||
         value.getBitWidth() == 32 || value.getBitWidth() == 64);

  int bytes = value.getBitWidth() / 8;

  EmitAlignment(bytes);

  uint64_t v = value.getLimitedValue();
  for (unsigned i = 0; i != bytes; ++i) {
    EmitByte(static_cast<uint8_t>(v & 255));
    v >>= 8;
  }
}

void BEFAttributeEmitter::EmitFloatAttribute(mlir::FloatAttr float_attr) {
  assert(float_attr.getType().isBF16() || float_attr.getType().isF16() ||
         float_attr.getType().isF32() || float_attr.getType().isF64());

  EmitIntegerAttribute(float_attr.getValue().bitcastToAPInt());
}

void BEFAttributeEmitter::EmitStringAttribute(string_view value) {
  EmitBytes(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(value.data()), value.size()));
}

void BEFAttributeEmitter::EmitTypeAttribute(mlir::TypeAttr type_attr) {
  auto attribute_type = ConvertMLIRDataTypeToTFRTDType(type_attr.getValue());

  EmitByte(static_cast<uint8_t>(attribute_type));
}

void BEFAttributeEmitter::EmitArrayAttribute(mlir::ArrayAttr array_attr) {
  assert(IsArrayAttribute(GetBEFAttributeType(array_attr)));
  for (auto elt : array_attr) EmitAttribute(elt);
}

void BEFAttributeEmitter::EmitSymbolRefAttribute(
    mlir::SymbolRefAttr symbol_ref_attr) {
  EmitBytes(compilation_units_.SerializedSymbolData(symbol_ref_attr));
  EmitBytes(compilation_units_.SerializedOperationData(symbol_ref_attr));
  EmitByte(0);
}

// This emits typed attributes that have BEFAttrBase as head.
//
// TODO(chky): Factor out this class to a standalone library as this should be
// higher level than BEF.
class BEFTypedAttributeEmitter : public BEFEmitter {
 public:
  explicit BEFTypedAttributeEmitter(BEFCompilationUnits& compilation_units)
      : compilation_units_(compilation_units) {
    // Typed attributes should be at least aligned to alignof(BEFAttrBase).
    EmitAlignment(alignof(BEFAttrBase));
  }

  void EmitAttribute(mlir::Attribute attr);

 private:
  void EmitAggregateAttribute(mlir::ArrayAttr aggregate_attr);
  void EmitArrayAttribute(mlir::ArrayAttr array_attr);
  void EmitDenseElementsAttribute(mlir::DenseElementsAttr dense_elements_attr);
  void EmitShapeAttribute(tfrt::corert::ShapeAttr shape_attr);
  void EmitRankedShapeAttribute(tfrt::corert::ShapeAttr shape_attr);

  BEFCompilationUnits& compilation_units_;
};

void BEFTypedAttributeEmitter::EmitAttribute(mlir::Attribute attr) {
  auto attribute_type = GetBEFAttributeType(attr);

  if (attribute_type == BEFAttributeType::kAggregate) {
    EmitAggregateAttribute(attr.cast<mlir::ArrayAttr>());
    return;
  }

  if (attribute_type == BEFAttributeType::kShape) {
    EmitShapeAttribute(attr.cast<tfrt::corert::ShapeAttr>());
    return;
  }

  if (IsArrayAttribute(attribute_type)) {
    EmitArrayAttribute(attr.cast<mlir::ArrayAttr>());
    return;
  }

  if (IsDenseAttribute(attribute_type)) {
    EmitDenseElementsAttribute(attr.cast<mlir::DenseElementsAttr>());
    return;
  }

  // Below logic handle the cases where untyped data is immediately following
  // BEFAttrBase.
  BEFAttributeEmitter untyped_emitter(compilation_units_);
  untyped_emitter.EmitAttribute(attr);

  size_t prev_start = size();

  BEFAttrBase base;
  base.type = attribute_type;

  for (int i = 0; i < sizeof(base); ++i) EmitByte(kDummyByte);

  EmitAlignment(untyped_emitter.GetRequiredAlignment());

  size_t payload_start = size();
  size_t byte_count = payload_start - prev_start + untyped_emitter.size();

  SetBEFAttrByteCount(byte_count, &base);

  OverwriteBytes(prev_start, &base, sizeof(base));

  EmitEmitter(untyped_emitter);
}

void BEFTypedAttributeEmitter::EmitAggregateAttribute(
    mlir::ArrayAttr aggregate_attr) {
  size_t start_offset = size();

  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = AssertAttrFieldSize32(aggregate_attr.size());

  size_t header_size =
      header.num_elements > 0
          ? sizeof(BEFAggregateAttr) +
                sizeof(BEFAggregateAttrOffset32_t) * (header.num_elements - 1)
          : sizeof(BEFAggregateAttr);

  for (int i = 0; i < header_size; ++i) EmitBytes(kDummyByte);

  SmallVector<BEFAggregateAttrOffset32_t, 8> offsets;
  for (auto element : aggregate_attr) {
    BEFTypedAttributeEmitter element_emitter(compilation_units_);
    element_emitter.EmitAttribute(element);
    EmitAlignment(element_emitter.GetRequiredAlignment());
    offsets.push_back(AssertAttrFieldSize32(size()));
    EmitEmitter(element_emitter);
  }

  SetBEFAttrByteCount(size() - start_offset, &header.base);

  size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(start_offset, &header, element_offset);
  OverwriteBytes(start_offset + element_offset, offsets.data(),
                 sizeof(BEFAggregateAttrOffset32_t) * offsets.size());
}

void BEFTypedAttributeEmitter::EmitArrayAttribute(mlir::ArrayAttr array_attr) {
  size_t start_offset = size();

  BEFAttributeType element_type = static_cast<BEFAttributeType>(DType::I32);
  if (!array_attr.empty()) element_type = GetBEFAttributeType(array_attr[0]);

  BEFArrayAttr header;
  header.base.type = GetArrayAttributeType(element_type);
  header.num_elements = AssertAttrFieldSize32(array_attr.size());

  // Reserve space for header.
  for (int i = 0; i < sizeof(header); ++i) EmitBytes(kDummyByte);

  BEFAttributeEmitter elements(compilation_units_);
  elements.EmitArrayAttribute(array_attr);

  EmitAlignment(elements.GetRequiredAlignment());
  header.element_offset = AssertAttrFieldSize32(size() - start_offset);
  EmitEmitter(elements);
  SetBEFAttrByteCount(size() - start_offset, &header.base);

  OverwriteBytes(start_offset, &header, sizeof(header));
}

void BEFTypedAttributeEmitter::EmitDenseElementsAttribute(
    mlir::DenseElementsAttr dense_elements_attr) {
  size_t start_offset = size();

  auto shaped_type = dense_elements_attr.getType();
  assert(shaped_type.hasRank());
  auto shape = shaped_type.getShape();

  BEFDenseAttr header;

  DType::Kind element_type =
      ConvertMLIRDataTypeToTFRTDType(shaped_type.getElementType());
  header.base.type = GetDenseAttributeType(element_type);
  header.rank = AssertAttrFieldSize16(shape.size());
  header.num_elements = AssertAttrFieldSize32(shaped_type.getNumElements());

  // Reserve space for header.
  for (int i = 0; i < sizeof(header); ++i) EmitBytes(kDummyByte);

  EmitAlignment(alignof(int64_t));
  header.shape_offset = AssertAttrFieldSize16(size() - start_offset);
  for (auto dim : shape) EmitInt8(dim);

  BEFAttributeEmitter elements(compilation_units_);

  if (element_type == DType::Complex64 || element_type == DType::Complex128) {
    if (element_type == DType::Complex64) {
      elements.EmitAlignment(alignof(std::complex<float>));
    } else {
      elements.EmitAlignment(alignof(std::complex<double>));
    }
    ArrayRef<char> raw_data = dense_elements_attr.getRawData();
    elements.EmitBytes(llvm::makeArrayRef(
        reinterpret_cast<const uint8_t*>(raw_data.data()), raw_data.size()));
  } else {
    // TODO(tfrt-dev): Use raw data directly for dense elements.
    for (auto attr : dense_elements_attr.getAttributeValues()) {
      elements.EmitAttribute(attr);
    }
  }

  EmitAlignment(elements.GetRequiredAlignment());
  header.element_offset = AssertAttrFieldSize32(size() - start_offset);
  EmitEmitter(elements);
  SetBEFAttrByteCount(size() - start_offset, &header.base);

  OverwriteBytes(start_offset, &header, sizeof(header));
}

void BEFTypedAttributeEmitter::EmitShapeAttribute(
    tfrt::corert::ShapeAttr shape_attr) {
  if (shape_attr.hasRank()) {
    EmitRankedShapeAttribute(shape_attr);
    return;
  }

  BefAttrEncoder encoder;
  auto error = encoder.EncodeUnrankedShapeAttr();
  assert(!error);
  (void)error;

  EmitEmitter(encoder);
}

void BEFTypedAttributeEmitter::EmitRankedShapeAttribute(
    tfrt::corert::ShapeAttr shape_attr) {
  assert(shape_attr.hasRank());

  ArrayRef<int64_t> shape = shape_attr.getShape();

  BefAttrEncoder encoder;
  auto error = encoder.EncodeRankedShapeAttr(shape);
  assert(!error);
  (void)error;

  EmitEmitter(encoder);
}

// This is the emitter that builds the attributes section of a BEF.
class BEFAttributesEmitter : public BEFFileEmitter {
 public:
  BEFAttributesEmitter(BEFCompilationUnits* compilation_units,
                       EntityIndex* entity_index,
                       BEFFileEmitter* attribute_type_emitter)
      : compilation_units_(*compilation_units),
        entity_index_(*entity_index),
        attribute_type_emitter_(*attribute_type_emitter),
        num_attributes_(0) {}

  void EmitAttribute(mlir::Attribute attr, bool typed);

  int GetNumAttributes() const { return num_attributes_; }

 private:
  void EmitAttributeType(size_t offset, BEFAttributeType attribute_type,
                         bool typed) {
    AttributeTag attr_tag(attribute_type, typed);

    attribute_type_emitter_.EmitInt(offset);
    attribute_type_emitter_.EmitInt(attr_tag.data);
  }

  BEFCompilationUnits& compilation_units_;
  EntityIndex& entity_index_;
  BEFFileEmitter& attribute_type_emitter_;

  int num_attributes_;
};

void BEFAttributesEmitter::EmitAttribute(mlir::Attribute attr, bool typed) {
  // Now we are about to emit an attribute.
  num_attributes_++;

  size_t offset;
  auto attribute_type = GetBEFAttributeType(attr);

  if (typed || IsDenseAttribute(attribute_type) ||
      attribute_type == BEFAttributeType::kAggregate ||
      attribute_type == BEFAttributeType::kShape) {
    // Currently DenseAttr and AggregateAttr are always typed.
    //
    // TODO(chky): clean up usage DenseAttr and AggregateAttr in native kernels
    // and remove the special handling here.
    BEFTypedAttributeEmitter attribute_emitter(compilation_units_);
    attribute_emitter.EmitAttribute(attr);

    EmitAlignment(attribute_emitter.GetRequiredAlignment());
    offset = size();
    EmitEmitter(attribute_emitter);

    if (typed)
      entity_index_.AddTypedAttributeOffset(attr, offset);
    else
      entity_index_.AddAttributeOffset(attr, offset);
  } else {
    // Untyped attributes go here.

    BEFAttributeEmitter attribute_emitter(compilation_units_);
    attribute_emitter.EmitAttribute(attr);

    // Emit size information in VBR form for untyped array and string
    // attributes.
    if (IsArrayAttribute(attribute_type) ||
        (IsDataTypeAttribute(attribute_type) &&
         GetDataType(attribute_type) == DType::String)) {
      const size_t len = (IsArrayAttribute(attribute_type))
                             ? attr.cast<mlir::ArrayAttr>().size()
                             : attr.cast<mlir::StringAttr>().getValue().size();

      const unsigned array_alignment = attribute_emitter.GetRequiredAlignment();
      EmitAlignment(array_alignment,
                    CalculateAlignmentPaddingSize(size(), GetSizeOfVbrInt(len),
                                                  array_alignment));
      offset = size();
      EmitInt(len);
      assert(size() % array_alignment == 0);
      EmitEmitter(attribute_emitter);
    } else if (IsSymbolRefAttribute(attribute_type)) {
      offset = size();

      // Emit size information in VBR form for the SymbolRef and
      // serialized compilation unit.
      auto symbol = attr.cast<mlir::SymbolRefAttr>();

      // Length of the root symbol name.
      EmitInt(symbol.getRootReference().size());

      // Lengths of the nested symbols names.
      size_t num_nested_refs = symbol.getNestedReferences().size();
      EmitInt(num_nested_refs);
      llvm::SmallVector<size_t, 4> nested_ref_len(num_nested_refs);
      for (size_t i = 0; i < num_nested_refs; ++i)
        EmitInt(symbol.getNestedReferences()[i].getValue().size());

      // Length of the serialized compilation unit.
      EmitInt(compilation_units_.SerializedOperationSize(symbol));

      EmitAlignment(attribute_emitter.GetRequiredAlignment());
      EmitEmitter(attribute_emitter);
    } else {
      EmitAlignment(attribute_emitter.GetRequiredAlignment());
      offset = size();
      EmitEmitter(attribute_emitter);
    }
    entity_index_.AddAttributeOffset(attr, offset);
  }

  EmitAttributeType(offset, attribute_type, typed);
}

void BEFModuleEmitter::EmitAttributes(BEFFileEmitter* attribute_types) {
  // The attributes are already in a stable order, so just emit them in the
  // order they were found.

  // Keep track of all compilation units in the module.
  BEFCompilationUnits compilation_units(module_);

  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  BEFFileEmitter attribute_type_emitter;
  BEFAttributesEmitter attributes_section(&compilation_units, &entity_index_,
                                          &attribute_type_emitter);
  for (auto attr : entities_.attributes) {
    attributes_section.EmitAttribute(attr, /* typed = */ false);
  }
  for (auto attr : entities_.typed_attributes) {
    attributes_section.EmitAttribute(attr, /* typed = */ true);
  }

  attribute_types->EmitInt(attributes_section.GetNumAttributes());
  attribute_types->EmitEmitter(attribute_type_emitter);

  EmitSection(BEFSectionID::kAttributes, attributes_section);
}

void BEFModuleEmitter::EmitKernels() {
  // The kernels are already in a stable order, so just emit them in the
  // order they were found.
  BEFFileEmitter ops_section;
  // Count of the number of kernels that exist.
  ops_section.EmitInt(entities_.kernels.size());

  for (auto op : entities_.kernels) {
    auto index = entity_index_.GetStringOffset(op);
    ops_section.EmitInt(index);
  }

  EmitSection(BEFSectionID::kKernels, ops_section);
}

void BEFModuleEmitter::EmitTypes() {
  // The types are already in a stable order, so just emit them in the
  // order they were found.
  BEFFileEmitter types_section;

  // Count of the number of types that exist.
  types_section.EmitInt(entities_.types.size());

  // Emit the index of the name of the types.
  for (auto type : entities_.types) {
    llvm::SmallVector<char, 64> result_str;
    llvm::raw_svector_ostream os(result_str);
    type.print(os);
    auto index = entity_index_.GetStringOffset(os.str());
    types_section.EmitInt(index);
  }

  EmitSection(BEFSectionID::kTypes, types_section);
}

// This is the emitter that builds the function entry of a BEF.
class BEFFunctionEmitter : public BEFFileEmitter {
 public:
  BEFFunctionEmitter(const EntityTable& entities,
                     const EntityIndex& entity_index)
      : entities_(entities), entity_index_(entity_index) {}

  void EmitFunction(mlir::Region* region, BEFFileEmitter* attribute_names,
                    BEFFileEmitter* register_types);

 private:
  void EmitRegisterTable(mlir::Block* block, BEFFileEmitter* register_types);
  template <typename UserRange>
  void EmitKernelResultUsers(UserRange users, BEFFileEmitter* kernel_list,
                             BEFFileEmitter* kernel_body) const;
  void EmitArgumentsPseudoKernel(mlir::Block* block,
                                 BEFFileEmitter* kernel_list) const;
  void EmitKernel(mlir::Operation* op, BEFFileEmitter* kernel_list,
                  BEFFileEmitter* attribute_names) const;

  unsigned GetRegisterNumber(mlir::Value reg) const {
    auto it = register_number_.find(reg);
    assert(it != register_number_.end() && "Unknown register");
    return it->second;
  }

  unsigned GetPseudoResultRegisterNumber() const {
    return register_number_.size();
  }

  void Reset() {
    register_number_.clear();
    kernel_index_.clear();
  }

  llvm::DenseMap<mlir::Value, unsigned> register_number_;
  llvm::DenseMap<mlir::Operation*, unsigned> kernel_index_;

  const EntityTable& entities_;
  const EntityIndex& entity_index_;
};

void BEFFunctionEmitter::EmitFunction(mlir::Region* region,
                                      BEFFileEmitter* attribute_names,
                                      BEFFileEmitter* register_types) {
  Reset();

  assert(llvm::hasSingleElement(*region) && "should have a single block");
  auto& block = region->front();

  auto location_offset =
      entity_index_.GetLocationPositionOffset(region->getParentOp());
  EmitInt(location_offset);

  // Emit the register table.
  EmitRegisterTable(&block, register_types);

  // Get a dense numbering of kernels, including the pseudo kernel.
  unsigned num_kernels = 1;

  for (auto& op : block.getOperations()) {
    if (!IsReturn(&op)) kernel_index_[&op] = num_kernels++;
  }

  // Emit a count of kernels, then the offset of each kernel (from the
  // start of the kernel list) then each kernel is emitted in turn.
  EmitInt(num_kernels);

  mlir::Operation* return_op = nullptr;

  BEFFileEmitter kernel_list;

  attribute_names->EmitInt(num_kernels);

  // Perform stream analysis to get stream information for this function.
  //
  // TODO(chky): This analysis is better performed at compiler side. However,
  // due to the limitation that asynchrony is implicit at compile-time the only
  // choice to integrate with BEF executor is to perform analysis in MLIRToBEF.
  // Once we make asynchrony explicit at compile-time, we should be able to move
  // this analysis out.
  compiler::StreamAnalysis stream_analysis(block);

  // Before we emit all the kernels, we always emit a pseudo kernel (with no
  // kernel_code) that is the entry to the other kernels. Specifically, its
  // users are:
  //  1) kernels that are using function arguments, and
  //  2) kernels that take no kernel arguments.

  // Offset of the kernel in the list.
  EmitInt(kernel_list.size());
  // Pseudo has zero operands that need to be available.
  EmitInt(0);
  // The pseudo kernel is always in the root stream.
  EmitInt(stream_analysis.GetRootStream().id());

  EmitArgumentsPseudoKernel(&block, &kernel_list);

  for (auto& op : block) {
    // Return kernels get special processing.
    if (IsReturn(&op)) {
      return_op = &op;
      continue;
    }

    bool is_non_strict = false;
    for (auto attr_and_name : op.getAttrs())
      if (ClassifyAttribute(attr_and_name.first) ==
          SpecialAttribute::kNonStrict) {
        DEBUG_PRINT("This is a non-strict kernel.\n");
        is_non_strict = true;
      }

    // Offset of the kernel in the list.
    EmitInt(kernel_list.size());
    // Number of operands that need to be available before it is ready to go.
    auto num_operands_before_running = op.getNumOperands();

    // We set the number to 1 for non-strict kernels so they get kicked off
    // as soon as any argument is avaiable.  We use 1 instead of zero because we
    // kernels with no operands ready are likely to just wait anyway.
    if (is_non_strict && num_operands_before_running)
      num_operands_before_running = 1;

    EmitInt(num_operands_before_running);

    // Emit stream id from stream analysis.
    const auto& stream = stream_analysis.GetStream(&op);
    EmitInt(stream.id());

    EmitKernel(&op, &kernel_list, attribute_names);
  }

  // Emit the result registers list at the end of the KERNEL_TABLE if present.
  if (return_op) {
    for (auto operand : return_op->getOperands()) {
      EmitInt(GetRegisterNumber(operand));
    }
  }

  // Once we're done, we can emit the kernel data after the kernel index
  // list. Note that kernel entries are fixed32 integers with 4-byte alignment.
  EmitAlignment(4);
  EmitEmitter(kernel_list);

  kernel_index_.clear();
}

void BEFFunctionEmitter::EmitRegisterTable(mlir::Block* block,
                                           BEFFileEmitter* register_types) {
  BEFFileEmitter reg_table;
  BEFFileEmitter reg_type_table;
  unsigned num_registers = 0;

  auto emit_register = [&](mlir::Value reg) {
    // Then the use-count.
    reg_table.EmitInt(std::distance(reg.use_begin(), reg.use_end()));

    // Emit the type index into register types section.
    reg_type_table.EmitInt(entities_.GetTypeIndex(reg.getType()));

    register_number_[reg] = num_registers++;
  };

  for (auto arg : block->getArguments()) emit_register(arg);

  for (auto& op : *block)
    for (auto result : op.getResults()) emit_register(result);

  // Emit the number of registers, then the register table.
  EmitInt(num_registers);
  EmitEmitter(reg_table);

  // Emit the number of registers, then the register type table in register
  // types section.
  register_types->EmitInt(num_registers);
  register_types->EmitEmitter(reg_type_table);
}

template <typename UserRange>
void BEFFunctionEmitter::EmitKernelResultUsers(
    UserRange users, BEFFileEmitter* kernel_list,
    BEFFileEmitter* kernel_body) const {
  int num_users = 0;
  for (auto* user : users) {
    // Ignore the 'return' op, it gets special handling.
    if (IsReturn(user)) continue;

    num_users++;
    auto it = kernel_index_.find(user);
    assert(it != kernel_index_.end() && "Invalid user");
    kernel_body->EmitInt4(it->second);
  }
  kernel_list->EmitInt4(num_users);
}

void BEFFunctionEmitter::EmitArgumentsPseudoKernel(
    mlir::Block* block, BEFFileEmitter* kernel_list) const {
  // This kernel starts with a dummy code and a dummy location. And this kernel
  // only has results and used_bys in its body.

  // code
  kernel_list->EmitInt4(kDummyPseudoKernelCode);
  // location
  kernel_list->EmitInt4(kDummyPseudoKernelLocation);
  // arguments
  kernel_list->EmitInt4(0);
  // attributes
  kernel_list->EmitInt4(0);
  // functions
  kernel_list->EmitInt4(0);
  // results, including the special result for ops with no operands.
  kernel_list->EmitInt4(block->getNumArguments() + 1);
  // special_metadata
  kernel_list->EmitInt4(0);

  BEFFileEmitter kernel_body;
  // The first result is the pseudo result used to trigger execution of kernels
  // with no operands.
  kernel_body.EmitInt4(GetPseudoResultRegisterNumber());
  for (auto arg : block->getArguments())
    kernel_body.EmitInt4(GetRegisterNumber(arg));

  // We also emit all operations with no operands as users for the special
  // result.
  llvm::SmallVector<mlir::Operation*, 4> ready_kernels;
  for (auto& op : *block) {
    if (op.getNumOperands() == 0) ready_kernels.push_back(&op);
  }
  EmitKernelResultUsers(ready_kernels, kernel_list, &kernel_body);

  for (auto arg : block->getArguments())
    EmitKernelResultUsers(arg.getUsers(), kernel_list, &kernel_body);

  assert(kernel_list->size() % kKernelEntryAlignment == 0);
  assert(kernel_body.GetRequiredAlignment() == kKernelEntryAlignment);
  kernel_list->EmitEmitter(kernel_body);
}

void BEFFunctionEmitter::EmitKernel(mlir::Operation* op,
                                    BEFFileEmitter* kernel_list,
                                    BEFFileEmitter* attribute_names) const {
  // Each kernel starts out with an opcode record.
  kernel_list->EmitInt4(entities_.GetKernelID(op));

  // Include a location.
  auto location_offset = entity_index_.GetLocationPositionOffset(op);
  kernel_list->EmitInt4(location_offset);

  // Because the numbers of each types of entries are emitted first, we use
  // another emitter to keep all entries and append them to kernel_list later.
  BEFFileEmitter kernel_body;

  // Then we have the arguments.
  kernel_list->EmitInt4(op->getNumOperands());
  for (auto operand : op->getOperands())
    kernel_body.EmitInt4(GetRegisterNumber(operand));

  // Then attributes.
  int num_input_functions = 0;
  int num_input_attributes = 0;
  BEFFileEmitter input_function_emitter;
  BEFFileEmitter input_attribute_emitter;
  uint32_t special_attribute = 0;
  bool is_op_attrs_typed = IsOpAttrsTyped(op);
  for (auto attr_name_pair : op->getAttrs()) {
    // Skip cost attribute which is not used in runtime execution.
    //
    // TODO(tfrt-devs): Use attribute interface instead of hardcoding here.
    if (attr_name_pair.first == "_tfrt_cost") continue;

    // Emit a flag in kernel header to indicate that the kernel is non-strict.
    if (ClassifyAttribute(attr_name_pair.first.strref()) ==
        SpecialAttribute::kNonStrict) {
      special_attribute |= static_cast<uint32_t>(SpecialAttribute::kNonStrict);
      continue;
    }

    // Emit array of function attributes.
    if (auto array_fn_attr =
            attr_name_pair.second.dyn_cast<mlir::ArrayAttr>()) {
      if (!array_fn_attr.empty() &&
          array_fn_attr.begin()->dyn_cast<mlir::FlatSymbolRefAttr>()) {
        for (auto fn : array_fn_attr) {
          num_input_functions++;
          input_function_emitter.EmitInt4(entities_.GetFunctionNamed(
              fn.dyn_cast<mlir::FlatSymbolRefAttr>().getValue()));
        }
        continue;
      }
    }

    if (auto fn_attr =
            attr_name_pair.second.dyn_cast<mlir::FlatSymbolRefAttr>()) {
      // Function references are output as regions.
      num_input_functions++;
      input_function_emitter.EmitInt4(
          entities_.GetFunctionNamed(fn_attr.getValue()));
    } else {
      attribute_names->EmitInt(
          entity_index_.GetOptionalStringOffset(attr_name_pair.first));
      num_input_attributes++;

      unsigned offset;
      if (is_op_attrs_typed)
        offset = entity_index_.GetTypedAttributeOffset(attr_name_pair.second);
      else
        offset = entity_index_.GetAttributeOffset(attr_name_pair.second);

      input_attribute_emitter.EmitInt4(offset);
    }
  }

  kernel_list->EmitInt4(num_input_attributes);
  kernel_body.EmitEmitter(input_attribute_emitter);

  // Then regions.
  num_input_functions += op->getNumRegions();
  for (auto& region : op->getRegions())
    input_function_emitter.EmitInt4(entities_.GetFunctionID(region));

  kernel_list->EmitInt4(num_input_functions);
  kernel_body.EmitEmitter(input_function_emitter);

  kernel_list->EmitInt4(op->getNumResults());
  for (auto result : op->getResults())
    kernel_body.EmitInt4(GetRegisterNumber(result));

  auto debug_info_offset = entity_index_.GetDebugInfoOffset(op);
  if (debug_info_offset.hasValue()) {
    special_attribute |= static_cast<uint32_t>(SpecialAttribute::kHasDebugInfo);
  }

  // Emit non-strict flag to special_metadata field of kernel header.
  kernel_list->EmitInt4(special_attribute);

  // Then results with the kernels that use them.
  for (auto result : op->getResults())
    EmitKernelResultUsers(result.getUsers(), kernel_list, &kernel_body);

  if (debug_info_offset.hasValue()) {
    kernel_body.EmitInt4(debug_info_offset.getValue());
  }

  assert(kernel_list->size() % kKernelEntryAlignment == 0);
  assert(kernel_body.size() == 0 ||
         kernel_body.GetRequiredAlignment() == kKernelEntryAlignment);
  kernel_list->EmitAlignment(4);
  kernel_list->EmitEmitter(kernel_body);
}

void BEFModuleEmitter::EmitFunctions(BEFFileEmitter* attribute_names,
                                     BEFFileEmitter* register_types) {
  BEFFunctionEmitter functions_section(entities_, entity_index_);

  attribute_names->EmitInt(entities_.functions.size());
  register_types->EmitInt(entities_.functions.size());
  for (auto function_entry : entities_.functions) {
    // Remember that we emitted this region to this offset.
    entity_index_.AddFunction(function_entry.name, functions_section.size(),
                              function_entry.type, function_entry.kind);
    if (!function_entry.IsNative()) {
      functions_section.EmitFunction(function_entry.region, attribute_names,
                                     register_types);
    }
  }

  // TODO(hyojun): Reduce the increased peak memory usage for keeping
  // function_index_section and functions_section to write the FunctionIndex
  // section before the Functions section.
  // We could improve it by changing the format of FunctionIndex section to
  // use FIXED32 (or FIXED64) instead of VBR integer for function offsets.
  // Or, we could introduce FunctionOffsetTable section, which could be placed
  // after Functions section.
  auto function_index = entity_index_.GetFunctionIndex();
  BEFFileEmitter function_index_section;

  // Count of the number of functions that exist.
  function_index_section.EmitInt(function_index.size());

  for (const auto& entry : function_index) {
    function_index_section.EmitByte(static_cast<uint8_t>(entry.kind));
    function_index_section.EmitInt(entry.function_offset);
    function_index_section.EmitInt(entry.name_offset);

    // Arguments.
    function_index_section.EmitInt(entry.type.getInputs().size());
    for (auto type : entry.type.getInputs())
      function_index_section.EmitInt(entities_.GetTypeIndex(type));

    // Results.
    function_index_section.EmitInt(entry.type.getResults().size());
    for (auto type : entry.type.getResults())
      function_index_section.EmitInt(entities_.GetTypeIndex(type));
  }

  EmitSection(BEFSectionID::kFunctionIndex, function_index_section);
  EmitSection(BEFSectionID::kFunctions, functions_section);
}

void BEFModuleEmitter::EmitAttributeTypes(
    const BEFFileEmitter& attribute_types) {
  EmitSection(BEFSectionID::kAttributeTypes, attribute_types);
}

void BEFModuleEmitter::EmitAttributeNames(
    const BEFFileEmitter& attribute_names) {
  EmitSection(BEFSectionID::kAttributeNames, attribute_names);
}

void BEFModuleEmitter::EmitRegisterTypes(const BEFFileEmitter& register_types) {
  EmitSection(BEFSectionID::kRegisterTypes, register_types);
}

// This function converts the specified MLIR module containing a host executor
// compatible program to the BinaryExecutableFormat (BEF) format, which is the
// low level format that the executor takes.
//
// On error, this emits the error message through the MLIR error handler, and
// returns an empty std:vector.
AlignedBuffer<8> ConvertMLIRToBEF(mlir::ModuleOp module,
                                  bool disable_optional_sections) {
  BEFModuleEmitter emitter(module);

  // Build the entities table.
  if (emitter.CollectEntities(!disable_optional_sections) ==
      LogicalResult::Failure)
    return {};

  // Emit magic numbers and format version.
  emitter.EmitBytes({kBEFMagic1, kBEFMagic2, kBEFVersion0});

  BEFFileEmitter attribute_types;
  BEFFileEmitter attribute_names;
  BEFFileEmitter register_types;

  // Emit each section of the file.
  emitter.EmitLocationInfo();
  emitter.EmitDebugInfo();
  emitter.EmitStrings();
  emitter.EmitAttributes(&attribute_types);
  emitter.EmitKernels();
  emitter.EmitTypes();
  emitter.EmitFunctions(&attribute_names, &register_types);

  if (!disable_optional_sections) {
    emitter.EmitAttributeTypes(attribute_types);
    emitter.EmitAttributeNames(attribute_names);
    emitter.EmitRegisterTypes(register_types);
  }

  // Return the result.
  return emitter.TakeResult();
}

}  // namespace tfrt
