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

//===- data_ops.td --------------------------------------------------------===//
//
// Operation definitions for data_ops.
//
//===----------------------------------------------------------------------===//

#ifdef DATA_OPS
#else
#define DATA_OPS

include "tfrt/tfrt_op_base.td"

// "data" dialect
def Data_Dialect : Dialect {
  let name = "data";

  let description = [{
    The Data dialect.

    This dialect contains common data operations.
  }];

  let cppNamespace = "data";
}

// Type definition.
def DatasetType: OpaqueType<"hex", "dataset", "!hex.dataset type">;
def IteratorType: OpaqueType<"hex", "iterator", "!hex.iterator type">;

// Base class for Data dialect ops.
class Data_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Data_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the Data namespace needs to provide all of a printer,
  // parser and verifier.
  let printer = [{ return tfrt::data::print(p, *this); }];
  let parser = [{ return tfrt::data::parse$cppClass(parser, result); }];
}

def MakeIteratorOp : Data_Op<"make_iterator"> {
  let summary = "data make_iterator operation";
  let description = [{
    data.make_iterator creates an iterator from a dataset.

    Example:
      %iterator = data.make_iterator (%dataset)
  }];

  let arguments = (ins DatasetType:$dataset);
  let results = (outs IteratorType:$iterator);

  let assemblyFormat = "`(`operands`)` attr-dict";
}

def IteratorGetNextOp : Data_Op<"iterator_get_next"> {
  let summary = "data iterator_get_next operation";
  let description = [{
    data.iterator_get_next yields the next element from an iterator.

    Example:
      %iterator = data.make_iterator (%dataset)
      %image, %label = data.iterator_get_next (%iterator) : !t.tensor, !t.tensor
  }];

  let arguments = (ins IteratorType:$iterator, ChainType:$chain);
  let results = (outs Variadic<AnyType>:$components);

  let assemblyFormat = "`(`operands`)` attr-dict `:` type($components)";
}

def EnumerateIteratorOp : Data_Op<"enumerate.iterator"> {
  let summary = "data enumerate.iterator operation";
  let description = [{
    data.enumerate.iterator repeatedly draws the next element from the iterator
    and executes the function on it to update the arguments until the iterator
    is exhausted.

    The result types are specified after the colon. The results have the same
    types as the operands, excluding the first operand, which is the iterator.

    example:
      data.enumerate.iterator (%iterator, %zero) { function = @add_one } : i64
  }];

  let arguments = (ins
    IteratorType:$iterator,
    Variadic<AnyType>:$arguments,

    FlatSymbolRefAttr:$function
  );

  let results = (outs Variadic<AnyType>:$results);
  let verifier = [{ return tfrt::data::verify(*this); }];
}

class BatchDatasetOp<string suffix>
  : Data_Op<"batch_dataset." # suffix> {
  let summary = "data batch_dataset operation";
  let description = [{
    data.batch_dataset wraps around another dataset instance and batches the
    underlying elements.

    Example:
      %dataset1 = data.range_dataset.i32 (%start, %stop, %step)
      %dataset2 = data.batch_dataset.i32 (%dataset1)
        { batch_size = 3: i32, same_input_metadata = 1 : i1 }
  }];

  let arguments = (ins
     DatasetType:$input_dataset,

     I32Attr:$batch_size,
     I1Attr:$same_input_metadata
  );
  let results = (outs DatasetType:$handle);

  let assemblyFormat = "`(`operands`)` attr-dict";
}

// TODO(b/155892156): In the long term, dataset ops will not be specialized
// for types. Instead, type information will be passed in via attributes.
def BatchDatasetI32Op : BatchDatasetOp<"i32">;
def BatchDatasetI64Op : BatchDatasetOp<"i64">;
def BatchDatasetTensorOp : BatchDatasetOp<"tensor">;
def BatchDatasetTensorAndI64Op : BatchDatasetOp<"tensor_and_i64">;

class RangeDatasetOp<string suffix>
  : Data_Op<"range_dataset." # suffix> {
  let summary = "data range_dataset operation";
  let description = [{
    data.range_dataset yields a step-separated range of values from
    start (inclusive) to stop (exclusive).

    Example:
      %dataset = data.range_dataset.i32 (%start, %stop, %step)
  }];

  let arguments = (ins
     I64:$start,
     I64:$stop,
     I64:$step
  );
  let results = (outs DatasetType:$handle);

  let assemblyFormat = "`(`operands`)` attr-dict";
}

// TODO(b/155892156): In the long term, dataset ops will not be specialized
// for types. Instead, type information will be passed in via attributes.
def RangeDatasetI32Op : RangeDatasetOp<"i32">;
def RangeDatasetI64Op : RangeDatasetOp<"i64">;

// TODO(rachelim): Add verification to map functions.
def MapDatasetOp : Data_Op<"map_dataset"> {
  let summary = "data make_dataset operation";
  let description = [{
    data.map_dataset maps a user-defined function over the elements in its
    input dataset.

    Example:
      %dataset1 = data.range_dataset.i32 (%start, %stop, %step)
      %dataset2 = data.map_dataset (%dataset1) {function = @times_two}
  }];

  let arguments = (ins
    DatasetType:$input_dataset,
    Variadic<AnyType>:$other_arguments,

    FlatSymbolRefAttr:$function
  );

  let results = (outs DatasetType:$handle);

  let assemblyFormat = [{
    `(`$input_dataset (`,` $other_arguments^ `:` type($other_arguments))?`)`
    attr-dict
  }];
}

#endif  // DATA_OPS
