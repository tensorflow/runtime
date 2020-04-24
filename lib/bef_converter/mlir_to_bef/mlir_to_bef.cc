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

static AttributeKind EncodeIntegerTypeAttribute(
    mlir::IntegerType integer_type) {
  switch (integer_type.getWidth()) {
    case 1:
      return AttributeKind::kI1;
    case 32:
      return AttributeKind::kI32;
    case 64:
      return AttributeKind::kI64;
  }

  llvm_unreachable("unknown integer type width.");
}

static AttributeKind EncodeFloatTypeAttribute(mlir::FloatType float_type) {
  switch (float_type.getWidth()) {
    case 16:
      return AttributeKind::kF16;
    case 32:
      return AttributeKind::kF32;
    case 64:
      return AttributeKind::kF64;
  }

  llvm_unreachable("unknown float type width.");
}

static AttributeKind EncodeTypeAttribute(mlir::Type type) {
  if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    return EncodeIntegerTypeAttribute(integer_type);
  }

  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return EncodeFloatTypeAttribute(float_type);
  }

  llvm_unreachable("unknown type attribute");
}

class AttrInfo {
 public:
  explicit AttrInfo(AttributeKind kind, size_t size = 0)
      : kind_descriptor_{kind}, size_(size) {
    assert(IsScalar() || size_ == 0);
  }
  explicit AttrInfo(AttributeKind kind, AttributeKind element_kind)
      : kind_descriptor_{kind, element_kind} {
    assert(IsScalar() || size_ == 0);
  }

  const AttributeKindDescriptor& kind_descriptor() const {
    return kind_descriptor_;
  }

  // Return the byte size of this attribute. If it is not a scalar attribute,
  // return 0.
  size_t size() const { return size_; }

  bool IsSupported() const {
    return kind_descriptor().kind != AttributeKind::kUnsupported;
  }

  bool IsScalar() const { return IsScalarAttribute(kind_descriptor().kind); }

 private:
  AttributeKindDescriptor kind_descriptor_;
  size_t size_ = 0;
};

// Return the kind of this attribute. If it is an array attribute, elements of
// it are checked recursively, and if any element is unsupported,
// AttributeKind::Unsupported will be returned.
static AttrInfo GetAttrInfo(mlir::Attribute attr) {
  // We support bool attributes (stored as 1 byte in BEF).
  if (attr.isa<mlir::BoolAttr>()) return AttrInfo{AttributeKind::kBool, 1};

  // We support 1-bit (stored as 1 byte in BEF), 32-bit, and 64-bit
  // integers.
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>()) {
    switch (int_attr.getValue().getBitWidth()) {
      case 1:
        return AttrInfo{AttributeKind::kI1, 1};
      case 32:
        return AttrInfo{AttributeKind::kI32, 4};
      case 64:
        return AttrInfo{AttributeKind::kI64, 8};
    }
  }

  // We support F16, F32 and F64 floats.
  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>()) {
    if (float_attr.getType().isF16()) return AttrInfo{AttributeKind::kF16, 2};
    if (float_attr.getType().isF32()) return AttrInfo{AttributeKind::kF32, 4};
    if (float_attr.getType().isF64()) return AttrInfo{AttributeKind::kF64, 8};
  }

  // We support string attributes.
  if (attr.isa<mlir::StringAttr>()) return AttrInfo{AttributeKind::kString};

  // We support i1, i32, i64, f16, f32 and f64 type attributes.
  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    auto type = type_attr.getValue();
    if (type.isInteger(1) || type.isInteger(32) || type.isInteger(64) ||
        type.isF16() || type.isF32() || type.isF64())
      return AttrInfo{AttributeKind::kType, 1};
  }

  // We support dense attributes.
  if (auto dense_elements_attr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    size_t size = 16;  // The dense attr header contains two uint64_t.

    auto num_elements = dense_elements_attr.getNumElements();
    if (num_elements == 0) return AttrInfo{AttributeKind::kDenseElements, size};

    auto element_size = GetBEFAttributeSize(
        EncodeTypeAttribute(dense_elements_attr.getType().getElementType()));
    assert(element_size != 0);
    size += num_elements * element_size;

    auto first_attr_info = GetAttrInfo(*dense_elements_attr.attr_value_begin());
    // If dense element type is unsupported, then this attribute is unsupported.
    if (!first_attr_info.IsSupported())
      return AttrInfo{AttributeKind::kUnsupported};

    // MLIR guarantees that all dense elements have the same type.
    return AttrInfo{AttributeKind::kDenseElements, size};
  }

  // We support arrays of supported attribute values.
  if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    if (array_attr.size() == 0) {
      // `array_element_kind` kI32 and is used as a dummy.
      return AttrInfo{AttributeKind::kFlatArray, AttributeKind::kI32};
    }

    auto first_attr_info = GetAttrInfo(*array_attr.begin());

    // Only scalar attributes can be included in a flat array.
    bool is_flat = first_attr_info.IsScalar();

    for (auto elt : array_attr) {
      auto attr_info = GetAttrInfo(elt);
      if (!attr_info.IsSupported())
        return AttrInfo{AttributeKind::kUnsupported};

      // Flat arrays requires all elements have the same type and the size.
      if (attr_info.kind_descriptor().kind !=
              first_attr_info.kind_descriptor().kind ||
          attr_info.size() != first_attr_info.size()) {
        is_flat = false;
        break;
      }
    }
    if (is_flat)
      return AttrInfo{AttributeKind::kFlatArray,
                      first_attr_info.kind_descriptor().kind};

    return AttrInfo{AttributeKind::kOffsetArray, 0};
  }

  return AttrInfo{AttributeKind::kUnsupported};
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a kernel, even in recursive positions.
static bool IsSupportedAttributeValue(mlir::Attribute attr) {
  return GetAttrInfo(attr).IsSupported();
}

// Return true if this is a supported attribute that can be emitted as a
// attribute reference in a kernel.
static bool IsSupportedAttribute(mlir::Attribute attr) {
  // We support references to functions.
  if (attr.isa<mlir::SymbolRefAttr>()) return true;

  return IsSupportedAttributeValue(attr);
}

// The "hex.return" kernel gets special case handling in BEF files.
static bool IsReturn(mlir::Operation* op) {
  return op->getName().getStringRef() == "hex.return";
}

static bool IsNativeFunc(mlir::FuncOp op) { return !!op.getAttr("hex.native"); }

static mlir::FunctionType GetRegionFunctionType(mlir::Region* region) {
  // Emit information about the type of the function.
  auto* block = &region->getBlocks().front();

  // Arguments.
  SmallVector<mlir::Type, 4> inputs;
  for (auto arg : block->getArguments()) inputs.push_back(arg.getType());

  // Results.
  // MLIR Regions don't have an easy way to identify results in regions, so
  // we just hard code the "hex.return" instruction.
  auto& last_op = block->back();
  assert(IsReturn(&last_op));

  SmallVector<mlir::Type, 4> results;
  for (auto op : last_op.getOperands()) results.push_back(op.getType());

  return mlir::FunctionType::get(inputs, results, region->getContext());
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
  LogicalResult AddFunction(mlir::Region* region, string_view name);
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

LogicalResult EntityTable::AddFunction(mlir::Region* region, string_view name) {
  // Check to see if we support this region kind.
  if (region->getBlocks().size() != 1) {
    mlir::emitError(region->getLoc())
        << "multi-block regions cannot be emitted to BEF files";
    return LogicalResult::Failure;
  }

  for (auto arg : region->front().getArguments()) AddType(arg.getType());

  // Remember this function.
  AddString(name);
  region_function_ids[region] = functions.size();
  named_function_ids[name] = functions.size();
  functions.push_back(FunctionEntry(name, GetRegionFunctionType(region),
                                    FunctionKind::kBEFFunction, region));
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
        auto& last_op = fn.getBlocks().front().back();
        if (!IsReturn(&last_op)) {
          last_op.emitError() << "all functions need to have a hex.return";
          result = LogicalResult::Failure;
          return;
        }

        if (AddFunction(&fn.getBody(), fn.getName()) ==
            LogicalResult::Failure) {
          result = LogicalResult::Failure;
          return;
        }
      }
    } else {
      AddKernel(op);

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
          attributes.insert(attr.second);
        }
      }

      // Keep add any regions used by this op as BEF functions.
      for (auto& region : op->getRegions()) {
        if (AddFunction(&region, "") == LogicalResult::Failure) {
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

  llvm::Optional<unsigned> GetOptionalAttributeOffset(
      mlir::Attribute attribute) const {
    auto it = attribute_offsets_.find(attribute);
    if (it == attribute_offsets_.end()) return llvm::None;
    return it->second;
  }

  unsigned GetAttributeOffset(mlir::Attribute attribute) const {
    auto offset = GetOptionalAttributeOffset(attribute);
    assert(offset && "attribute didn't get added to the entity collection");
    return *offset;
  }

  void AddAttributeOffset(mlir::Attribute attribute, unsigned offset) {
    assert(!attribute_offsets_.count(attribute) &&
           "attribute already in index");
    attribute_offsets_.insert({attribute, offset});
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
class BEFEmitter {
 public:
  static constexpr uint8_t kDummyByte = 0xCC;
  static constexpr uint32_t kDummyPseudoKernelCode = 0xABABABAB;
  static constexpr uint32_t kDummyPseudoKernelLocation = 0xCDCDCDCD;

  BEFEmitter() {}
  BEFEmitter(const BEFEmitter&) = delete;
  BEFEmitter& operator=(const BEFEmitter&) = delete;

  // Return the alignment required by this chunk of a BEF file.
  unsigned GetRequiredAlignment() const { return required_alignment_; }

  size_t size() const { return result_.size(); }

  void EmitByte(uint8_t byte) { result_.push_back(byte); }

  void EmitBytes(llvm::ArrayRef<uint8_t> bytes) {
    result_.insert(result_.end(), bytes.begin(), bytes.end());
  }

  void EmitAlignment(unsigned alignment) {
    // Alignment of 0 and 1 is a noop.
    if (alignment < 2) return;

    assert(llvm::isPowerOf2_32(alignment));

    // We need attributes to have proper alignment in the file, so figure out
    // whether we need padding before this to make sure it ends up at the right
    // address.
    size_t cur_offset = size();
    size_t needed_padding = llvm::alignTo(cur_offset, alignment) - cur_offset;

    // Emit dummy padding bytes to get up to the right offset.
    while (needed_padding--) EmitByte(kDummyByte);

    // Keep track of the maximum required alignment.
    required_alignment_ = std::max(required_alignment_, alignment);
  }

  // Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt4(uint32_t value) {
    EmitAlignment(4);
    uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF),
                      uint8_t((value >> 16) & 0xFF),
                      uint8_t((value >> 24) & 0xFF)};
    EmitBytes(data);
  }

  // Emit a guaranteed 8-byte integer aligned to 8 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt8(uint64_t value) {
    EmitAlignment(8);
    uint8_t data[] = {
        uint8_t(value & 0xFF),         uint8_t((value >> 8) & 0xFF),
        uint8_t((value >> 16) & 0xFF), uint8_t((value >> 24) & 0xFF),
        uint8_t((value >> 32) & 0xFF), uint8_t((value >> 40) & 0xFF),
        uint8_t((value >> 48) & 0xFF), uint8_t((value >> 56) & 0xFF)};
    EmitBytes(data);
  }

  // Emit a vbr encoded integer of arbitrary width.
  void EmitInt(size_t value) { EmitIntImpl(value, false); }
  // Emit a vbr encoded integer with low byte first
  void EmitIntLowByteFirst(size_t value) {
    EmitBEFArrayLength(value, &result_);
  }

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper function emits the subpieces once they are
  // constructed, ensuring that alignment requirements of the nested emitter
  // are maintained correctly.
  void EmitEmitter(const BEFEmitter& emitter) {
    EmitAlignment(emitter.GetRequiredAlignment());
    EmitBytes(emitter.result_);
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

    // TODO(tf-runtime-team): In the case where we already happen to be aligned,
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

  void EmitSection(BEFSectionID section_id, const BEFEmitter& emitter) {
    EmitSection(section_id, emitter.result_, emitter.GetRequiredAlignment());
  }

  std::vector<uint8_t> TakeResult() { return std::move(result_); }

  // Move size bytes in the result from src_offset to dst_offset.
  void MoveResult(size_t dst_offset, size_t src_offset, size_t size);

  // Set size bytes in the result from offset to value
  void SetResult(size_t offset, uint8_t value, size_t size);

 private:
  void EmitIntImpl(size_t value, bool is_high_part);
  // Keep track of the alignment required for the start of this object.
  unsigned required_alignment_ = 1;
  std::vector<uint8_t> result_;
};

}  // namespace

void BEFEmitter::MoveResult(size_t dst_offset, size_t src_offset, size_t size) {
  memmove(result_.data() + dst_offset, result_.data() + src_offset, size);
}

void BEFEmitter::SetResult(size_t offset, uint8_t value, size_t size) {
  memset(result_.data() + offset, value, size);
}

// Our fundamental unit is a bytestream, but we want to be able to emit large
// values as well.  We use a VBR encoding, where the high bit set indicates
// that this is only a portion of the value.
void BEFEmitter::EmitIntImpl(size_t value, bool is_high_part) {
  if ((value >> 7) != 0) EmitIntImpl(value >> 7, /*is_high_part=*/true);

  result_.push_back(
      static_cast<uint8_t>((value & 127) | (is_high_part ? 128 : 0)));
}

// This is the emitter that builds a BEF into an std::vector.
class BEFModuleEmitter : public BEFEmitter {
 public:
  explicit BEFModuleEmitter(mlir::ModuleOp module) : module_(module) {}

  LogicalResult CollectEntities(bool collect_attribute_types_and_names) {
    return entities_.Collect(module_, collect_attribute_types_and_names);
  }

  void EmitFormatVersion();
  void EmitLocationInfo();
  void EmitStrings();
  void EmitAttributes(BEFEmitter* attribute_types);
  void EmitKernels();
  void EmitTypes();
  void EmitFunctions(BEFEmitter* attribute_names, BEFEmitter* register_types);
  void EmitFunctionIndex();
  void EmitAttributeTypes(const BEFEmitter& attribute_types);
  void EmitAttributeNames(const BEFEmitter& attribute_names);
  void EmitRegisterTypes(const BEFEmitter& register_types);

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
  BEFEmitter filenames_section;
  for (auto filename : entities_.location_filenames) {
    filenames_section.EmitBytes(
        {reinterpret_cast<const uint8_t*>(filename.data()), filename.size()});
    // Emit a NUL terminator for the filename.
    filenames_section.EmitByte(0);
  }

  EmitSection(BEFSectionID::kLocationFilenames, filenames_section);

  // Emit each of the positions and remember the offsets within the section.
  BEFEmitter positions_section;
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
  BEFEmitter string_section;
  for (const auto& entry : strs_in_order) {
    entity_index_.AddString(entry, string_section.size());
    string_section.EmitBytes(
        {reinterpret_cast<const uint8_t*>(entry.data()), entry.size()});
    // Emit a NUL terminator for the string.
    string_section.EmitByte(0);
  }

  EmitSection(BEFSectionID::kStrings, string_section);
}

struct OffsetAndType {
  size_t offset;
  size_t attribute_type;
};

// This is the emitter that builds the attributes section of a BEF.
class BEFAttributesEmitter : public BEFEmitter {
 public:
  explicit BEFAttributesEmitter(const EntityTable& entities,
                                EntityIndex* entity_index,
                                BEFEmitter* attribute_type_emitter)
      : entities_(entities),
        entity_index_(*entity_index),
        attribute_type_emitter_(*attribute_type_emitter) {}

  size_t EmitAttribute(mlir::Attribute attr);

  int GetNumAttributes() const { return num_attributes_; }

 private:
  OffsetAndType EmitAttributeInternal(mlir::Attribute attr);

  size_t EncodeAttributeType(AttributeTypeID attribute_type_id,
                             size_t payload) {
    assert(static_cast<size_t>(attribute_type_id) <= kMaxAttributeTypeID &&
           "AttributeTypeID overflow");
    return static_cast<size_t>(attribute_type_id) |
           (payload << kAttributeTypeIDShift);
  }

  void EmitAttributeType(size_t offset, size_t attribute_type) {
    attribute_type_emitter_.EmitInt(offset);
    attribute_type_emitter_.EmitInt(attribute_type);
  }

  // Return {offset, attribute_type} for each type of attribute.
  OffsetAndType EmitBoolAttribute(bool value);
  OffsetAndType EmitStandardAttribute(mlir::Attribute attr);
  OffsetAndType EmitDenseElementsAttribute(
      mlir::DenseElementsAttr dense_elements_attr);
  OffsetAndType EmitStringAttribute(string_view value);
  OffsetAndType EmitTypeAttribute(mlir::TypeAttr type_attr);
  OffsetAndType EmitArrayAttribute(mlir::ArrayAttr array_attr);

  size_t EmitIntegerAttribute(const llvm::APInt& value);
  size_t EmitFloatAttribute(const llvm::APFloat& value);

  const EntityTable& entities_;
  EntityIndex& entity_index_;
  BEFEmitter& attribute_type_emitter_;

  int num_attributes_ = 0;
};
size_t BEFAttributesEmitter::EmitAttribute(mlir::Attribute attr) {
  // Since there might be nested array attributes, they might have been already
  // emitted.
  if (auto offset = entity_index_.GetOptionalAttributeOffset(attr))
    return *offset;

  // Now we are about to emit an attribute.
  num_attributes_++;

  auto offset_and_type = EmitAttributeInternal(attr);

  auto offset = offset_and_type.offset;
  auto attribute_type = offset_and_type.attribute_type;
  entity_index_.AddAttributeOffset(attr, offset);
  EmitAttributeType(offset, attribute_type);

  return offset;
}

OffsetAndType BEFAttributesEmitter::EmitAttributeInternal(
    mlir::Attribute attr) {
  assert(IsSupportedAttribute(attr) &&
         "EmitAttribute get an unsupported attribute");

  if (auto bool_attr = attr.dyn_cast<mlir::BoolAttr>()) {
    return EmitBoolAttribute(bool_attr.getValue());
  }

  if (attr.isa<mlir::IntegerAttr>() || attr.isa<mlir::FloatAttr>()) {
    return EmitStandardAttribute(attr);
  }

  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    return EmitTypeAttribute(type_attr);
  }

  if (auto dense_elements_attr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
    return EmitDenseElementsAttribute(dense_elements_attr);
  }

  // We support arrays of attributes.
  if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    return EmitArrayAttribute(array_attr);
  }

  if (auto string_attr = attr.dyn_cast<mlir::StringAttr>()) {
    return EmitStringAttribute(string_attr.getValue());
  }

  llvm_unreachable("Unknown attribute");
}

OffsetAndType BEFAttributesEmitter::EmitStandardAttribute(
    mlir::Attribute attr) {
  auto attribute_type =
      EncodeAttributeType(AttributeTypeID::kStandardAttribute,
                          entities_.GetOptionalTypeIndex(attr.getType()));
  if (auto int_attr = attr.dyn_cast<mlir::IntegerAttr>())
    return {EmitIntegerAttribute(int_attr.getValue()), attribute_type};

  if (auto float_attr = attr.dyn_cast<mlir::FloatAttr>())
    return {EmitFloatAttribute(float_attr.getValue()), attribute_type};

  llvm_unreachable("Unknown standard attribute");
}

OffsetAndType BEFAttributesEmitter::EmitBoolAttribute(bool value) {
  auto offset = size();
  auto attribute_type = EncodeAttributeType(AttributeTypeID::kBoolAttribute, 0);
  EmitByte(static_cast<uint8_t>(value));
  return {offset, attribute_type};
}

size_t BEFAttributesEmitter::EmitIntegerAttribute(const llvm::APInt& value) {
  if (value.getBitWidth() == 1) {
    auto offset = size();
    EmitByte(static_cast<uint8_t>(value.getLimitedValue()));
    return offset;
  }

  int bytes;
  if (value.getBitWidth() == 32) {
    bytes = 4;
  } else {
    assert(value.getBitWidth() == 64);
    bytes = 8;
  }

  EmitAlignment(bytes);

  auto offset = size();

  uint64_t v = value.getLimitedValue();
  for (unsigned i = 0; i != bytes; ++i) {
    EmitByte(static_cast<uint8_t>(v & 255));
    v >>= 8;
  }
  return offset;
}

size_t BEFAttributesEmitter::EmitFloatAttribute(const llvm::APFloat& value) {
  llvm::APInt int_val = value.bitcastToAPInt();
  return EmitIntegerAttribute(int_val);
}

OffsetAndType BEFAttributesEmitter::EmitStringAttribute(string_view value) {
  // In order to support strings of data with NUL characters in them, we emit
  // strings as modified Pascal-style strings, that start with a modified-VBR
  // length.  The string passed to the function points to the start of the data,
  // but the bytes before it specify how long the string is.  To enforce this
  // notion, we don't emit a zero byte at the end of the string data.
  EmitIntLowByteFirst(value.size());

  size_t offset = size();
  EmitBytes(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(value.data()), value.size()));

  auto attribute_type =
      EncodeAttributeType(AttributeTypeID::kStringAttribute, 0);

  return {offset, attribute_type};
}

OffsetAndType BEFAttributesEmitter::EmitTypeAttribute(
    mlir::TypeAttr type_attr) {
  size_t offset = size();
  auto attribute_type = EncodeAttributeType(AttributeTypeID::kTypeAttribute, 0);

  EmitByte(static_cast<uint8_t>(EncodeTypeAttribute(type_attr.getValue())));
  return {offset, attribute_type};
}

OffsetAndType BEFAttributesEmitter::EmitArrayAttribute(
    mlir::ArrayAttr array_attr) {
  bool is_flat = GetAttrInfo(array_attr).kind_descriptor().kind ==
                 AttributeKind::kFlatArray;

  auto elts = array_attr.getValue();

  BEFEmitter offset_array_element_emitter;
  if (!is_flat) {
    // If it is not a flat array, we first recursively emit its elements.
    for (auto elt : elts) {
      size_t element_offset = EmitAttribute(elt);
      offset_array_element_emitter.EmitInt4(element_offset);
      auto kind_descriptor = GetAttrInfo(elt).kind_descriptor();
      offset_array_element_emitter.EmitByte(
          static_cast<uint8_t>(kind_descriptor.kind));
      offset_array_element_emitter.EmitByte(
          static_cast<uint8_t>(kind_descriptor.array_element_kind));
      offset_array_element_emitter.EmitAlignment(4);
    }
  }

  size_t size_start = size();

  // Emit the number of elements first and then emit all of the attributes
  // consecutively, returning the offset of the first element in the attribute
  // array.
  EmitIntLowByteFirst(elts.size());

  size_t size_end = size();

  size_t attribute_type = 0;

  // The attribute array is empty. We just return the offset past the size.
  // The payload of attribute_type will be ignored.
  if (elts.empty()) {
    attribute_type =
        EncodeAttributeType(AttributeTypeID::kFlatArrayAttribute, 0);
    return {size_end, attribute_type};
  }

  size_t array_attribute_offset;
  if (is_flat) {
    // For flat array attributes, its offset is the offset of the first
    // element.
    auto offset_and_type = EmitAttributeInternal(elts.front());
    array_attribute_offset = offset_and_type.offset;
    attribute_type = EncodeAttributeType(AttributeTypeID::kFlatArrayAttribute,
                                         offset_and_type.attribute_type);

    for (auto elt : elts.drop_front()) (void)EmitAttributeInternal(elt);
  } else {
    // Emit alignment paddings to the correct offset of this array attribute.
    EmitAlignment(offset_array_element_emitter.GetRequiredAlignment());
    array_attribute_offset = size();
    attribute_type =
        EncodeAttributeType(AttributeTypeID::kOffsetArrayAttribute, 0);

    EmitEmitter(offset_array_element_emitter);
  }

  // If there is a gap between the size and the first attribute value, we
  // move the size to be immediately before the first attribute value to
  // remove the gap.
  if (size_end < array_attribute_offset) {
    MoveResult(array_attribute_offset - (size_end - size_start), size_start,
               size_end - size_start);
    // Set the unused bytes to the dummy byte
    SetResult(size_start, kDummyByte, array_attribute_offset - size_end);
  }
  return {array_attribute_offset, attribute_type};
}

// TODO(tf-runtime-team): Consider using an encoding scheme similar to array.
OffsetAndType BEFAttributesEmitter::EmitDenseElementsAttribute(
    mlir::DenseElementsAttr dense_elements_attr) {
  // Align to uint64_t since the attribute header starts with uint64_t.
  EmitAlignment(8);
  size_t start_offset = size();
  auto shaped_type = dense_elements_attr.getType();

  // Emit dtype and rank of shape as one uint64_t.
  uint64_t dtype =
      static_cast<uint64_t>(EncodeTypeAttribute(shaped_type.getElementType()));
  assert(shaped_type.hasRank());
  auto shape = shaped_type.getShape();
  size_t rank = shape.size();

  uint64_t dtype_and_shape_rank = 0;
  assert(rank >> 56 == 0 && "top byte of shape rank is non-zero");
  dtype_and_shape_rank = (dtype << 56) | rank;
  EmitInt8(dtype_and_shape_rank);

  // Emit the payload element count.
  EmitInt8(shaped_type.getNumElements());

  // Emit shape elements.
  for (auto shape_elt : shape) {
    EmitInt8(shape_elt);
  }

  // Emit payload elements.
  // TODO(zhangqiaorjc): Check element alignment <= 8.
  auto elts_iter = dense_elements_attr.attr_value_begin();
  while (elts_iter != dense_elements_attr.attr_value_end()) {
    EmitStandardAttribute(*elts_iter);
    ++elts_iter;
  }

  size_t attribute_type =
      EncodeAttributeType(AttributeTypeID::kDenseElementsAttribute, 0);

  return {start_offset, attribute_type};
}

void BEFModuleEmitter::EmitAttributes(BEFEmitter* attribute_types) {
  // The attributes are already in a stable order, so just emit them in the
  // order they were found.

  // Emit attributes and record them in EntityIndex. Nested array attributes
  // will be traversed recursively and their elements will be emitted and
  // recorded before the top level offsets array is emitted.
  BEFEmitter attribute_type_emitter;
  BEFAttributesEmitter attributes_section(entities_, &entity_index_,
                                          &attribute_type_emitter);
  for (auto cst : entities_.attributes) {
    auto offset = attributes_section.EmitAttribute(cst);
    (void)offset;
  }

  attribute_types->EmitInt(attributes_section.GetNumAttributes());
  attribute_types->EmitEmitter(attribute_type_emitter);

  EmitSection(BEFSectionID::kAttributes, attributes_section);
}

void BEFModuleEmitter::EmitKernels() {
  // The kernels are already in a stable order, so just emit them in the
  // order they were found.
  BEFEmitter ops_section;
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
  BEFEmitter types_section;

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
class BEFFunctionEmitter : public BEFEmitter {
 public:
  BEFFunctionEmitter(const EntityTable& entities,
                     const EntityIndex& entity_index)
      : entities_(entities), entity_index_(entity_index) {}

  void EmitFunction(mlir::Region* region, BEFEmitter* attribute_names,
                    BEFEmitter* register_types);

 private:
  void EmitRegisterTable(mlir::Block* block, BEFEmitter* register_types);
  void EmitKernelResultUsers(mlir::Value result, BEFEmitter* kernel_list,
                             BEFEmitter* kernel_body) const;
  void EmitArgumentsPseudoOp(mlir::Block* block, BEFEmitter* emitter) const;
  void EmitKernel(mlir::Operation* op, BEFEmitter* kernel_list,
                  BEFEmitter* attribute_names) const;

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
                                      BEFEmitter* attribute_names,
                                      BEFEmitter* register_types) {
  assert(region->getBlocks().size() == 1 && "should have a single block");
  auto& block = region->getBlocks().front();

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

  BEFEmitter kernel_list;

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
                                           BEFEmitter* register_types) {
  BEFEmitter reg_table;
  BEFEmitter reg_type_table;
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

void BEFFunctionEmitter::EmitKernelResultUsers(mlir::Value result,
                                               BEFEmitter* kernel_list,
                                               BEFEmitter* kernel_body) const {
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

void BEFFunctionEmitter::EmitArgumentsPseudoOp(mlir::Block* block,
                                               BEFEmitter* kernel_list) const {
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

  BEFEmitter kernel_body;
  for (auto arg : block->getArguments())
    kernel_body.EmitInt4(GetRegisterNumber(arg));
  for (auto arg : block->getArguments())
    EmitKernelResultUsers(arg, kernel_list, &kernel_body);

  assert(kernel_list->size() % kKernelEntryAlignment == 0);
  assert(kernel_body.GetRequiredAlignment() == kKernelEntryAlignment);
  kernel_list->EmitEmitter(kernel_body);
}

void BEFFunctionEmitter::EmitKernel(mlir::Operation* op,
                                    BEFEmitter* kernel_list,
                                    BEFEmitter* attribute_names) const {
  // Each kernel starts out with an opcode record.
  kernel_list->EmitInt4(entities_.GetKernelID(op));

  // Include a location.
  auto location_offset =
      entity_index_.GetLocationPositionOffset(op->getLoc(), entities_);
  kernel_list->EmitInt4(location_offset);

  // Because the numbers of each types of entries are emitted first, we use
  // another emitter to keep all entries and append them to kernel_list later.
  BEFEmitter kernel_body;

  // Then we have the arguments.
  kernel_list->EmitInt4(op->getNumOperands());
  for (auto operand : op->getOperands())
    kernel_body.EmitInt4(GetRegisterNumber(operand));

  // Then attributes.
  int num_input_functions = 0;
  int num_input_attributes = 0;
  BEFEmitter input_function_emitter;
  BEFEmitter input_attribute_emitter;
  uint32_t special_attribute = 0;
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
      input_attribute_emitter.EmitInt4(
          entity_index_.GetAttributeOffset(attr_name_pair.second));
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

void BEFModuleEmitter::EmitFunctions(BEFEmitter* attribute_names,
                                     BEFEmitter* register_types) {
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

  BEFEmitter function_index_section;

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

void BEFModuleEmitter::EmitAttributeTypes(const BEFEmitter& attribute_types) {
  EmitSection(BEFSectionID::kAttributeTypes, attribute_types);
}

void BEFModuleEmitter::EmitAttributeNames(const BEFEmitter& attribute_names) {
  EmitSection(BEFSectionID::kAttributeNames, attribute_names);
}

void BEFModuleEmitter::EmitRegisterTypes(const BEFEmitter& register_types) {
  EmitSection(BEFSectionID::kRegisterTypes, register_types);
}

// This function converts the specified MLIR module containing a host executor
// compatible program to the BinaryExecutableFormat (BEF) format, which is the
// low level format that the executor takes.
//
// On error, this emits the error message through the MLIR error handler, and
// returns an empty std:vector.
std::vector<uint8_t> ConvertMLIRToBEF(mlir::ModuleOp module,
                                      bool disable_optional_sections) {
  BEFModuleEmitter emitter(module);

  // Build the entities table.
  if (emitter.CollectEntities(!disable_optional_sections) ==
      LogicalResult::Failure)
    return {};

  // Magic number at the start of the file.
  emitter.EmitBytes({kBEFMagic1, kBEFMagic2});

  BEFEmitter attribute_types;
  BEFEmitter attribute_names;
  BEFEmitter register_types;

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
