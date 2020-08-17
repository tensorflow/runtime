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

//===- mlir_to_bef.cc -----------------------------------------------------===//
//
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
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "tfrt/bef_converter/bef_emitter.h"
#include "tfrt/core_runtime/opdefs/attributes.h"
#include "tfrt/core_runtime/opdefs/traits.h"
#include "tfrt/core_runtime/opdefs/types.h"
#include "tfrt/support/aligned_buffer.h"
#include "tfrt/support/bef_encoding.h"
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

static BEFDataType EncodeIntegerTypeAttribute(mlir::IntegerType integer_type) {
  if (integer_type.isUnsigned()) {
    switch (integer_type.getWidth()) {
      case 8:
        return BEFDataType::kUI8;
      case 16:
        return BEFDataType::kUI16;
      case 32:
        return BEFDataType::kUI32;
      case 64:
        return BEFDataType::kUI64;
    }
  } else {
    switch (integer_type.getWidth()) {
      case 1:
        return BEFDataType::kI1;
      case 8:
        return BEFDataType::kI8;
      case 16:
        return BEFDataType::kI16;
      case 32:
        return BEFDataType::kI32;
      case 64:
        return BEFDataType::kI64;
    }
  }

  llvm_unreachable("unknown integer type width.");
}

static BEFDataType EncodeFloatTypeAttribute(mlir::FloatType float_type) {
  if (float_type.isBF16()) return BEFDataType::kBF16;
  if (float_type.isF16()) return BEFDataType::kF16;
  if (float_type.isF32()) return BEFDataType::kF32;
  if (float_type.isF64()) return BEFDataType::kF64;

  llvm_unreachable("unknown float type width.");
}

static BEFDataType EncodeComplexTypeAttribute(mlir::ComplexType complex_type) {
  auto element_type = complex_type.getElementType();

  if (element_type.isF32()) return BEFDataType::kComplex64;
  if (element_type.isF64()) return BEFDataType::kComplex128;

  llvm_unreachable("unknown complex type width.");
}

static BEFDataType ConvertMLIRDataTypeToBEFDataType(mlir::Type type) {
  if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    return EncodeIntegerTypeAttribute(integer_type);
  }

  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return EncodeFloatTypeAttribute(float_type);
  }

  if (auto string_type = type.dyn_cast<corert::StringType>()) {
    return BEFDataType::kString;
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
  // We support bool attributes (stored as 1 byte in BEF).
  if (attr.isa<mlir::BoolAttr>())
    return static_cast<BEFAttributeType>(BEFDataType::kBool);

  // We support 1-bit (stored as 1 byte in BEF), 32-bit, and 64-bit
  // integers.
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
    auto int_type = int_attr.getType().cast<mlir::IntegerType>();
    if (int_type.isUnsigned()) {
      switch (int_type.getWidth()) {
        case 8:
          return static_cast<BEFAttributeType>(BEFDataType::kUI8);
        case 16:
          return static_cast<BEFAttributeType>(BEFDataType::kUI16);
        case 32:
          return static_cast<BEFAttributeType>(BEFDataType::kUI32);
        case 64:
          return static_cast<BEFAttributeType>(BEFDataType::kUI64);
      }
    } else {
      switch (int_type.getWidth()) {
        case 1:
          return static_cast<BEFAttributeType>(BEFDataType::kI1);
        case 8:
          return static_cast<BEFAttributeType>(BEFDataType::kI8);
        case 16:
          return static_cast<BEFAttributeType>(BEFDataType::kI16);
        case 32:
          return static_cast<BEFAttributeType>(BEFDataType::kI32);
        case 64:
          return static_cast<BEFAttributeType>(BEFDataType::kI64);
      }
    }
  }

  // We support BF16, F16, F32 and F64 floats.
  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>()) {
    if (float_attr.getType().isBF16())
      return static_cast<BEFAttributeType>(BEFDataType::kBF16);
    if (float_attr.getType().isF16())
      return static_cast<BEFAttributeType>(BEFDataType::kF16);
    if (float_attr.getType().isF32())
      return static_cast<BEFAttributeType>(BEFDataType::kF32);
    if (float_attr.getType().isF64())
      return static_cast<BEFAttributeType>(BEFDataType::kF64);
  }

  // We support string attributes.
  if (attr.isa<mlir::StringAttr>())
    return static_cast<BEFAttributeType>(BEFDataType::kString);

  // We support i1, i8, i16, i32, i64, ui8, ui16, ui32, ui64, bf16, f16, f32,
  //  f64, complex64, complex128 and string type attributes.
  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    auto type = type_attr.getValue();
    if (type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
        type.isInteger(32) || type.isInteger(64) || type.isBF16() ||
        type.isF16() || type.isF32() || type.isF64() ||
        type.isa<corert::StringType>())
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
    auto element_type = ConvertMLIRDataTypeToBEFDataType(
        dense_elements_attr.getType().getElementType());
    if (element_type == BEFDataType::kUnsupported)
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
  return !!op.getAttr("tfrt.native");
}

static bool IsSyncFunc(mlir::FuncOp op) { return !!op.getAttr("tfrt.sync"); }

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

  return mlir::FunctionType::get(inputs, results, region->getContext());
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

  // This is a list of locations within the file, the first element of the
  // tuple is a index into location_filenames, the second and third are line/col
  // information.
  typedef std::tuple<unsigned, unsigned, unsigned> LocationTuple;
  llvm::SetVector<LocationTuple> location_positions;

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

  void AddLocation(mlir::Location loc);

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

void EntityTable::AddLocation(mlir::Location loc) {
  string_view filename = "";
  unsigned line = 0, col = 0;
  if (auto file_line_col = loc.dyn_cast<mlir::FileLineColLoc>()) {
    filename = file_line_col.getFilename();
    line = file_line_col.getLine();
    col = file_line_col.getColumn();
  }

  auto next_filename_index = location_filenames.size();
  auto it =
      location_filenames_index.insert({filename, next_filename_index}).first;
  if (it->second == next_filename_index) location_filenames.push_back(filename);

  location_positions.insert(LocationTuple{it->second, line, col});
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

  module.walk([&](mlir::Operation* op) {
    // Ignore the module itself, and a few specific other ops.
    if (op == module.getOperation() || llvm::isa<mlir::ModuleTerminatorOp>(op))
      return;

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

    AddLocation(op->getLoc());

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

        // Verify that all functions end with a return to catch a common error.
        auto& last_op = fn.front().back();
        if (!IsReturn(&last_op)) {
          last_op.emitError() << "all functions need to have a tfrt.return";
          result = LogicalResult::Failure;
          return;
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
        // If this is a special attribute, ignore it.
        if (ClassifyAttribute(attr.first.strref()) !=
            SpecialAttribute::kUnknown)
          continue;

        // Check to make sure that this is a supported attribute, if not, reject
        // it.
        if (!IsSupportedAttribute(attr.second) &&
            result == LogicalResult::Success) {
          op->emitError() << "BEF files cannot encode the '" << attr.first
                          << "' attribute";
          result = LogicalResult::Failure;
          return;
        }

        if (auto fn_attr = attr.second.dyn_cast<mlir::SymbolRefAttr>()) {
          // Keep track of function attributes specially so we can diagnose
          // them.
          fn_attrs.push_back({fn_attr, op->getLoc()});

        } else {
          if (collect_attribute_types_and_names) {
            // Add attribute names and types for attribute types section and
            // attribute names section. These will be ignored by executor.
            AddString(attr.first);
            AddAttributeType(attr.second);
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

  // If we're successful, check to make sure that all functions can be resolved.
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

  void AddLocationPosition(LocationTuple position, size_t offset) {
    location_position_offsets_[position] = offset;
  }

  size_t GetLocationPositionOffset(mlir::Location loc,
                                   const EntityTable& entities) const {
    string_view filename = "";
    unsigned line = 0, col = 0;
    if (auto file_line_col = loc.dyn_cast<mlir::FileLineColLoc>()) {
      filename = file_line_col.getFilename();
      line = file_line_col.getLine();
      col = file_line_col.getColumn();
    }

    auto fn_it = entities.location_filenames_index.find(filename);
    assert(fn_it != entities.location_filenames_index.end() &&
           "unknown location");
    auto fn_idx = fn_it->second;

    auto loc_it =
        location_position_offsets_.find(LocationTuple{fn_idx, line, col});
    assert(loc_it != location_position_offsets_.end() && "unknown location");
    return loc_it->second;
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
  llvm::DenseMap<LocationTuple, size_t> location_position_offsets_;
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

  // Emit a vbr encoded integer with low byte first
  void EmitIntLowByteFirst(size_t value) {
    EmitBEFArrayLength(value, &result_);
  }

  void EmitSection(BEFSectionID section_id,
                   llvm::ArrayRef<uint8_t> section_data,
                   unsigned alignment = 1) {
    // Section start with an identifier.
    result_.push_back(static_cast<uint8_t>(section_id));

    bool has_alignment = alignment > 1;

    // Then have a length along with a bit indicating whether and alignment is
    // present.
    EmitInt((section_data.size() << 1) | (has_alignment ? 1 : 0));

    // TODO(tfrt-devs): In the case where we already happen to be aligned,
    // we could save N bytes of output by noticing that we're already aligned,
    // propagating the alignment to our container, but not emitting the
    // alignment marker or the fill bytes.
    if (has_alignment) {
      EmitByte(alignment);

      // Move up to the right alignment for the section data.
      EmitAlignment(alignment);
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

// This is the emitter that builds a BEF into an std::vector.
class BEFModuleEmitter : public BEFFileEmitter {
 public:
  explicit BEFModuleEmitter(mlir::ModuleOp module) : module_(module) {}

  LogicalResult CollectEntities(bool collect_attribute_types_and_names) {
    return entities_.Collect(module_, collect_attribute_types_and_names);
  }

  void EmitFormatVersion();
  void EmitLocationInfo();
  void EmitStrings();
  void EmitAttributes(BEFFileEmitter* attribute_types);
  void EmitKernels();
  void EmitTypes();
  void EmitFunctions(BEFFileEmitter* attribute_names,
                     BEFFileEmitter* register_types);
  void EmitFunctionIndex();
  void EmitAttributeTypes(const BEFFileEmitter& attribute_types);
  void EmitAttributeNames(const BEFFileEmitter& attribute_names);
  void EmitRegisterTypes(const BEFFileEmitter& register_types);

 private:
  mlir::ModuleOp module_;
  EntityTable entities_;
  EntityIndex entity_index_;
};

void BEFModuleEmitter::EmitFormatVersion() {
  uint8_t version = kBEFVersion0;
  EmitSection(BEFSectionID::kFormatVersion, version);
}

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
  for (auto position : entities_.location_positions) {
    entity_index_.AddLocationPosition(position, positions_section.size());
    positions_section.EmitInt(std::get<0>(position));
    positions_section.EmitInt(std::get<1>(position));
    positions_section.EmitInt(std::get<2>(position));
  }

  EmitSection(BEFSectionID::kLocationPositions, positions_section);
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
  void EmitAttribute(mlir::Attribute attr);

  void EmitBoolAttribute(bool value);
  void EmitStandardAttribute(mlir::Attribute attr);
  void EmitStringAttribute(string_view value);
  void EmitTypeAttribute(mlir::TypeAttr type_attr);
  void EmitArrayAttribute(mlir::ArrayAttr array_attr);

  void EmitIntegerAttribute(const llvm::APInt& value);
  void EmitFloatAttribute(mlir::FloatAttr attr);
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
  auto attribute_type = ConvertMLIRDataTypeToBEFDataType(type_attr.getValue());

  EmitByte(static_cast<uint8_t>(attribute_type));
}

void BEFAttributeEmitter::EmitArrayAttribute(mlir::ArrayAttr array_attr) {
  assert(IsArrayAttribute(GetBEFAttributeType(array_attr)));
  for (auto elt : array_attr) EmitAttribute(elt);
}

// This emits typed attributes that have BEFAttrBase as head.
//
// TODO(chky): Factor out this class to a standalone library as this should be
// higher level than BEF.
class BEFTypedAttributeEmitter : public BEFEmitter {
 public:
  BEFTypedAttributeEmitter() {
    // Typed attributes should be at least aligned to alignof(BEFAttrBase).
    EmitAlignment(alignof(BEFAttrBase));
  }

  void EmitAttribute(mlir::Attribute attr);

 private:
  void EmitAggregateAttribute(mlir::ArrayAttr aggregate_attr);
  void EmitArrayAttribute(mlir::ArrayAttr array_attr);
  void EmitDenseElementsAttribute(mlir::DenseElementsAttr dense_elements_attr);
  void EmitShapeAttribute(tfrt::corert::ShapeAttr shape_attr);
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
  BEFAttributeEmitter untyped_emitter;
  untyped_emitter.EmitAttribute(attr);

  size_t prev_start = size();

  BEFAttrBase base;
  base.type = attribute_type;

  for (int i = 0; i < sizeof(base); ++i) EmitByte(kDummyByte);

  EmitAlignment(untyped_emitter.GetRequiredAlignment());

  size_t payload_start = size();

  base.byte_count =
      AssertAttrFieldSize(payload_start - prev_start + untyped_emitter.size());

  OverwriteBytes(prev_start, &base, sizeof(base));

  EmitEmitter(untyped_emitter);
}

void BEFTypedAttributeEmitter::EmitAggregateAttribute(
    mlir::ArrayAttr aggregate_attr) {
  size_t start_offset = size();

  // TODO(chky): Consider directly allocate header in result buffer.
  BEFAggregateAttr header;
  header.base.type = BEFAttributeType::kAggregate;
  header.num_elements = AssertAttrFieldSize(aggregate_attr.size());

  size_t header_size = header.num_elements > 0
                           ? sizeof(BEFAggregateAttr) +
                                 sizeof(uint16_t) * (header.num_elements - 1)
                           : sizeof(BEFAggregateAttr);

  for (int i = 0; i < header_size; ++i) EmitBytes(kDummyByte);

  SmallVector<uint16_t, 8> offsets;
  for (auto element : aggregate_attr) {
    BEFTypedAttributeEmitter element_emitter;
    element_emitter.EmitAttribute(element);
    EmitAlignment(element_emitter.GetRequiredAlignment());
    offsets.push_back(AssertAttrFieldSize(size()));
    EmitEmitter(element_emitter);
  }

  header.base.byte_count = AssertAttrFieldSize(size() - start_offset);

  size_t element_offset = offsetof(BEFAggregateAttr, offsets);
  OverwriteBytes(start_offset, &header, element_offset);
  OverwriteBytes(start_offset + element_offset, offsets.data(),
                 sizeof(uint16_t) * offsets.size());
}

void BEFTypedAttributeEmitter::EmitArrayAttribute(mlir::ArrayAttr array_attr) {
  size_t start_offset = size();

  BEFAttributeType element_type =
      static_cast<BEFAttributeType>(BEFDataType::kI32);
  if (!array_attr.empty()) element_type = GetBEFAttributeType(array_attr[0]);

  BEFArrayAttr header;
  header.base.type = GetArrayAttributeType(element_type);
  header.num_elements = AssertAttrFieldSize(array_attr.size());

  // Reserve space for header.
  for (int i = 0; i < sizeof(header); ++i) EmitBytes(kDummyByte);

  BEFAttributeEmitter elements;
  elements.EmitArrayAttribute(array_attr);

  EmitAlignment(elements.GetRequiredAlignment());
  header.element_offset = AssertAttrFieldSize(size() - start_offset);
  EmitEmitter(elements);
  header.base.byte_count = AssertAttrFieldSize(size() - start_offset);

  OverwriteBytes(start_offset, &header, sizeof(header));
}

void BEFTypedAttributeEmitter::EmitDenseElementsAttribute(
    mlir::DenseElementsAttr dense_elements_attr) {
  size_t start_offset = size();

  auto shaped_type = dense_elements_attr.getType();
  assert(shaped_type.hasRank());
  auto shape = shaped_type.getShape();

  BEFDenseAttr header;
  header.base.type = GetDenseAttributeType(
      ConvertMLIRDataTypeToBEFDataType(shaped_type.getElementType()));
  header.rank = shape.size();
  header.num_elements = AssertAttrFieldSize(shaped_type.getNumElements());

  // Reserve space for header.
  for (int i = 0; i < sizeof(header); ++i) EmitBytes(kDummyByte);

  EmitAlignment(alignof(int64_t));
  header.shape_offset = AssertAttrFieldSize(size() - start_offset);
  for (auto dim : shape) EmitInt8(dim);

  BEFAttributeEmitter elements;
  for (auto attr : dense_elements_attr.getAttributeValues()) {
    elements.EmitAttribute(attr);
  }

  EmitAlignment(elements.GetRequiredAlignment());
  header.element_offset = AssertAttrFieldSize(size() - start_offset);
  EmitEmitter(elements);
  header.base.byte_count = AssertAttrFieldSize(size() - start_offset);

  OverwriteBytes(start_offset, &header, sizeof(header));
}

void BEFTypedAttributeEmitter::EmitShapeAttribute(
    tfrt::corert::ShapeAttr shape_attr) {
  assert(shape_attr.hasRank());

  ArrayRef<int64_t> shape = shape_attr.getShape();

  uint16_t rank = AssertAttrFieldSize(shape.size());
  uint16_t byte_count = AssertAttrFieldSize(sizeof(BEFShapeAttr));
  if (rank > 0) {
    byte_count = AssertAttrFieldSize(sizeof(int64_t) * (rank - 1) + byte_count);
  }

  EmitAlignment(alignof(BEFShapeAttr));

  EmitInt2(static_cast<uint16_t>(BEFAttributeType::kShape));
  EmitInt2(byte_count);
  EmitInt2(rank);
  EmitByte(kDummyByte);
  EmitByte(kDummyByte);

  for (int64_t dim : shape) {
    EmitInt8(dim);
  }
}

// This is the emitter that builds the attributes section of a BEF.
class BEFAttributesEmitter : public BEFFileEmitter {
 public:
  explicit BEFAttributesEmitter(EntityIndex* entity_index,
                                BEFFileEmitter* attribute_type_emitter)
      : entity_index_(*entity_index),
        attribute_type_emitter_(*attribute_type_emitter) {}

  void EmitAttribute(mlir::Attribute attr, bool typed);

  int GetNumAttributes() const { return num_attributes_; }

 private:
  void EmitAttributeType(size_t offset, BEFAttributeType attribute_type,
                         bool typed) {
    AttributeTag attr_tag(attribute_type, typed);

    attribute_type_emitter_.EmitInt(offset);
    attribute_type_emitter_.EmitInt(attr_tag.data);
  }

  EntityIndex& entity_index_;
  BEFFileEmitter& attribute_type_emitter_;

  int num_attributes_ = 0;
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
    BEFTypedAttributeEmitter attribute_emitter;
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

    BEFAttributeEmitter attribute_emitter;
    attribute_emitter.EmitAttribute(attr);

    // Emit size information in reversed VBR form for untyped array and string
    // attributes.
    if (IsArrayAttribute(attribute_type) ||
        (IsDataTypeAttribute(attribute_type) &&
         GetDataType(attribute_type) == BEFDataType::kString)) {
      size_t len = 0;

      if (IsArrayAttribute(attribute_type)) {
        len = attr.cast<mlir::ArrayAttr>().size();
      } else {
        len = attr.cast<mlir::StringAttr>().getValue().size();
      }

      size_t size_start = size();
      // Emit the number of elements first and then emit all of the attributes
      // consecutively, returning the offset of the first element in the
      // attribute array.
      EmitIntLowByteFirst(len);
      size_t size_end = size();

      EmitAlignment(attribute_emitter.GetRequiredAlignment());
      offset = size();
      EmitEmitter(attribute_emitter);

      // If there is a gap between the size and the first attribute value, we
      // move the size to be immediately before the first attribute value to
      // remove the gap.
      if (size_end < offset) {
        MoveResult(offset - (size_end - size_start), size_start,
                   size_end - size_start);
        // Set the unused bytes to the dummy byte
        SetResult(size_start, kDummyByte, offset - size_end);
      }
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

  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  BEFFileEmitter attribute_type_emitter;
  BEFAttributesEmitter attributes_section(&entity_index_,
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
  void EmitKernelResultUsers(mlir::Value result, BEFFileEmitter* kernel_list,
                             BEFFileEmitter* kernel_body) const;
  void EmitArgumentsPseudoOp(mlir::Block* block,
                             BEFFileEmitter* kernel_list) const;
  void EmitKernel(mlir::Operation* op, BEFFileEmitter* kernel_list,
                  BEFFileEmitter* attribute_names) const;

  unsigned GetRegisterNumber(mlir::Value reg) const {
    auto it = register_number_.find(reg);
    assert(it != register_number_.end() && "Unknown register");
    return it->second;
  }

  llvm::DenseMap<mlir::Value, unsigned> register_number_;
  llvm::DenseMap<mlir::Operation*, unsigned> kernel_index_;

  const EntityTable& entities_;
  const EntityIndex& entity_index_;
};

void BEFFunctionEmitter::EmitFunction(mlir::Region* region,
                                      BEFFileEmitter* attribute_names,
                                      BEFFileEmitter* register_types) {
  assert(llvm::hasSingleElement(*region) && "should have a single block");
  auto& block = region->front();

  auto location_offset =
      entity_index_.GetLocationPositionOffset(region->getLoc(), entities_);
  EmitInt(location_offset);

  // Emit the register table.
  EmitRegisterTable(&block, register_types);

  // Get a dense numbering of kernels.
  unsigned num_kernels = 0;

  // If the function has arguments, we emit a pseudo-op that provides the
  // argument values.
  if (block.getNumArguments() != 0) ++num_kernels;

  for (auto& op : block.getOperations()) {
    if (!IsReturn(&op)) kernel_index_[&op] = num_kernels++;
  }

  // Emit a count of kernels, then the offset of each kernel (from the
  // start of the kernel list) then each kernel is emitted in turn.
  EmitInt(num_kernels);

  mlir::Operation* return_op = nullptr;

  BEFFileEmitter kernel_list;

  attribute_names->EmitInt(num_kernels);

  // If we have arguments, emit a pseudo op (with no opcode) that produces the
  // registers on entry to the function.
  if (block.getNumArguments() != 0) {
    // Offset of the kernel in the list.
    EmitInt(kernel_list.size());
    // Pseudo has zero operands that need to be available.
    EmitInt(0);

    EmitArgumentsPseudoOp(&block, &kernel_list);

    // Pseudo is not non-strict. And pseudo op has no attributes.
    attribute_names->EmitByte(static_cast<uint8_t>(SpecialAttribute::kUnknown));
  }

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

    if (is_non_strict) {
      attribute_names->EmitByte(
          static_cast<uint8_t>(SpecialAttribute::kNonStrict));
    } else {
      attribute_names->EmitByte(
          static_cast<uint8_t>(SpecialAttribute::kUnknown));
    }

    EmitKernel(&op, &kernel_list, attribute_names);
  }

  // Emit the result registers list at the end of the KERNEL_TABLE if present.
  if (return_op) {
    for (auto operand : return_op->getOperands())
      EmitInt(GetRegisterNumber(operand));
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

void BEFFunctionEmitter::EmitKernelResultUsers(
    mlir::Value result, BEFFileEmitter* kernel_list,
    BEFFileEmitter* kernel_body) const {
  int num_users = 0;
  for (auto* user : result.getUsers()) {
    // Ignore the 'return' op, it gets special handling.
    if (IsReturn(user)) continue;

    num_users++;
    auto it = kernel_index_.find(user);
    assert(it != kernel_index_.end() && "Invalid user");
    kernel_body->EmitInt4(it->second);
  }
  kernel_list->EmitInt4(num_users);
}

void BEFFunctionEmitter::EmitArgumentsPseudoOp(
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
  // results
  kernel_list->EmitInt4(block->getNumArguments());
  // special_metadata
  kernel_list->EmitInt4(0);

  BEFFileEmitter kernel_body;
  for (auto arg : block->getArguments())
    kernel_body.EmitInt4(GetRegisterNumber(arg));
  for (auto arg : block->getArguments())
    EmitKernelResultUsers(arg, kernel_list, &kernel_body);

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
  auto location_offset =
      entity_index_.GetLocationPositionOffset(op->getLoc(), entities_);
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
    // Emit a flag in kernel header to indicate that the kernel is non-strict.
    if (ClassifyAttribute(attr_name_pair.first.strref()) ==
        SpecialAttribute::kNonStrict) {
      special_attribute |= static_cast<uint32_t>(SpecialAttribute::kNonStrict);
      continue;
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

  // Emit non-strict flag to special_metadata field of kernel header.
  kernel_list->EmitInt4(special_attribute);

  // Then results with the kernels that use them.
  for (auto result : op->getResults())
    EmitKernelResultUsers(result, kernel_list, &kernel_body);

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

  EmitSection(BEFSectionID::kFunctions, functions_section);
}

void BEFModuleEmitter::EmitFunctionIndex() {
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

  // Magic number at the start of the file.
  emitter.EmitBytes({kBEFMagic1, kBEFMagic2});

  BEFFileEmitter attribute_types;
  BEFFileEmitter attribute_names;
  BEFFileEmitter register_types;

  // Emit each section of the file.
  emitter.EmitFormatVersion();
  emitter.EmitLocationInfo();
  emitter.EmitStrings();
  emitter.EmitAttributes(&attribute_types);
  emitter.EmitKernels();
  emitter.EmitTypes();
  emitter.EmitFunctions(&attribute_names, &register_types);
  emitter.EmitFunctionIndex();

  if (!disable_optional_sections) {
    emitter.EmitAttributeTypes(attribute_types);
    emitter.EmitAttributeNames(attribute_names);
    emitter.EmitRegisterTypes(register_types);
  }

  // Return the result.
  return emitter.TakeResult();
}

}  // namespace tfrt
