# Binary Executable Format

<!--* freshness: {
  owner: 'chky'
  owner: 'hyojun'
  owner: 'xldrx'
  reviewed: '2021-06-07'
} *-->

<!-- TOC -->

This document describes the goals and rationale for the in-memory representation
the executor uses to run host kernels, known as the Binary Executable Format
"BEF".

We have the following goals:

-   We want this to be simple and relatively stable - though it is versioned
    just in case (tm). We don't want this to contain domain specific
    abstractions that will require changing it frequently over time.

-   This format is designed to be usable as an immutable region in memory,
    allowing it to be built offline and conveniently `mmap`'d in (or stored in
    the `.rodata` segment of an executable, etc).

-   This is aligned with the abstractions used by the host executor and includes
    key abstractions that it can directly use to make it speedy and memory
    efficient. The design puts more burden on the encoder of a BEF file (e.g.
    requiring multiple passes) to make the decoder fast.

-   We expect this to be produced online (e.g. when a `TF_Function` is built on
    demand) e.g. in training or interactive use cases.

-   We need to represent location information in order to reflect runtime errors
    (e.g. shape errors) back to the user source program.

-   We should follow normal best practices for binary encodings, e.g. having
    tools to round trip them to a textual form (MLIR in our case), support
    arbitrary sized files (not limited to 4GB), be efficient to read, make
    reasonable attempts to be dense, etc.

In addition to describing the format, this document describes some of the
rationale for the design decisions as well as TODO items that need to be
completed before this work is done. This file format shouldn't be declared
stable until all the TODOs are resolved.

## Fundamental Concepts

#### Grammar

```none
  BYTE                   ::= `0x00`...`0xFF`
  INTEGER                ::= (`0x80`...`0xFF`)* (`0x00`...`0x7F`)
  FIXED32                ::= BYTE BYTE BYTE BYTE
  NULL_TERMINATED_STRING ::= (`0x01`...`0xFF`)* `0x00`
  OFFSET                 ::= INTEGER
  INDEX                  ::= INTEGER
```

A BEF file is formed as a byte stream whose top-level structure is a list of
"sections" that hold various sorts of data. It uses a few fundamental concepts:

*   *Integers* are encoded in a "Variable Byte Rate" (VBR) encoding, allowing
    small integers (less than 2^7) to be stored in one byte, integers up to 2^14
    to be stored in two bytes, etc. This is done by using the high bit of each
    byte in the stream to indicate "more bytes are coming".

*   *Fixed Integers* are unsigned integers with fixed bit width. The values are
    stored in little-endian byte order.

*   *Offsets* are Integers that demarcate a byte offset in the stream from a
    fixed position (typically the start of a section). These are useful when the
    section contains a bunch of variable length things, e.g. when referring to a
    string in the string table. Because variable sized integers are used
    pervasively in BEF files, offsets are very common.

*   *Indexes* are Integers that demarcate an entry in a table, e.g. a register
    or kernel in a function. These are typically used when indexing into a table
    where the structure is expected to be decoded and held in memory by the
    reader, e.g. as is the case with the kernels and types tables.

BEF files can be created by an arbitrary producer, but the standard ways is to
use the
[MLIRToBEF](https://github.com/tensorflow/runtime/blob/master/lib/bef_converter/mlir_to_bef/mlir_to_bef.cc)
translation library (or standalone tool), which turns an MLIR representation of
a host program into a BEF file.

A
[BEFToMLIR](https://github.com/tensorflow/runtime/blob/master/lib/bef_converter/bef_to_mlir/bef_to_mlir.cc)
translation library performs the reverse translation from BEF file to MLIR
representation of a host program. It is useful for writing BEF encoder testcases
and for dealing with that "random file someone sent you".

## Sections

#### Grammar

```none
  SECTION                ::= SECTION_HEADER SECTION_DATA
  SECTION_HEADER         ::= SECTION_ID INTEGER<LENGTH_AND_ALIGNMENT> SECTION_BODY_ALIGNMENT?
  SECTION_ID             ::= BYTE
  LENGTH_AND_ALIGNMENT   ::= (SECTION_LENGTH << 1) | (SECTION_ALIGNMENT_FLAG)
  SECTION_BODY_ALIGNMENT ::= BYTE<"Alignment"> BYTE<"AlignmentPadding">*
```

*Sections* are the top level entities in the file (after the header). Each
section contains a *Section ID*, a length (allowing the section to be skipped
over entirely by the reader) and an optional alignment for the contents of the
section.

Section IDs and other fundamental constants are defined in
[`bef_encoding.h`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/bef/bef_encoding.h),
and utilities for decoding the basic file structures like VBR integers are
defined in
[`bef_reader.h`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/bef/bef_reader.h).

The LENGTH_AND_ALIGNMENT contains SECTION_LENGTH and SECTION_ALIGNMENT_FLAG. The
SECTION_LENGTH contains one bit shifted value of the section length and the
SECTION_ALIGNMENT_FLAG (0 bit) indicates if SECTION_BODY_ALIGNMENT exists or
not; 0 means the section body starts immediately and 1 means
SECTION_BODY_ALIGNMENT follows.

## Top Level Structure

#### Grammar

```none
  BEF_FILE     ::= `0x0B` `0xEF` FORMAT_VERSION_NUMBER SECTION*

  FORMAT_VERSION_NUMBER ::= `0x00`

  SECTION_DATA ::= STRINGS_SECTION
  SECTION_DATA ::= ATTRIBUTES_SECTION
  SECTION_DATA ::= KERNELS_SECTION
  SECTION_DATA ::= TYPES_SECTION
  SECTION_DATA ::= FUNCTION_INDEX_SECTION
  SECTION_DATA ::= FUNCTIONS_SECTION
  SECTION_DATA ::= LOCATION_STRINGS_SECTION
  SECTION_DATA ::= LOCATIONS_SECTION
  SECTION_DATA ::= ATTRIBUTE_TYPES_SECTION
  SECTION_DATA ::= ATTRIBUTE_NAMES_SECTION
  SECTION_DATA ::= REGISTER_TYPES_SECTION

  // Unknown section.
  SECTION_DATA ::= BYTE*
```

The top level structure of the file is a two-byte "magic number" of `0x0BEF`
followed by one byte sized FORMAT_VERSION_NUMBER and a list of sections.

The current FORMAT_VERSION_NUMBER is 0, and will be increased when BEF format is
changed.

The reader skips over unknown sections, which could be useful for future
evolution of the format, e.g. if we want to store extra metadata in the BEF
format for some purpose.

### Strings Section

#### Grammar

```none
  STRINGS_SECTION ::= NULL_TERMINATED_STRING*
```

The Strings section contains a list of NULL terminated strings used by the
program (e.g. for type and kernel names). Entries in this section are referenced
by an Offset from the start of the section.

Note: This format doesn't support embedded NULL strings, which is sufficient for
the existing use cases. We could switch to modified Pascal strings if embeded
NULL characters become important for something (at a space/complexity cost).

### Attributes Section

#### Grammar

```none
  ATTRIBUTES_SECTION ::= BYTE*
```

The Attributes section contains the value of attributes used by kernels in the
BEF program. They are stored on the natural alignment boundary of the type, and
the address of the attribute is directly passed into the kernel implementation
function as an const pointer.

BEF files only support a subset of MLIR attributes, currently including:

*   booleans, stored as 1-byte integers.
*   i1 integers, stored as 1-byte integers.
*   i32 integers, stored as 4-byte little endian integers.
*   i64 integers, stored as 8-byte little endian integers.
*   f32 floats, stored as IEEE single precision floats.
*   f64 floats, stored as IEEE double precision floats.
*   type enums, stored as 1-byte integers. Currently supported type enums are
    i1, i32, i64, f32 and f64.
*   strings, stored as arrays of bytes, not necessarily NULL terminated.
*   dense elements, stored as shape dtype, shape rank, elements count, followed
    by shape elements and elements themselves. Each element can be any of the
    integer and float format above.
*   arrays, all elements of which are of the same type and fixed in width (eg.
    i32, f32, type).
*   aggregates, stored as array of i32 integers, which are offsets to other
    constants in Attributes Section. These nested constants can be of any
    supported attribute type including aggregates. Unlike arrays, an aggregate
    can contain a mix of different attribute types.

TODO: Support 8/16-bit integers and floating point constants.

#### Rationale

The host executor needs to refer to attribute values, and since the BEF file is
designed to be memory mapped in, we can directly use the encoding in memory.
This means that kernels will need to bswap attributes for big-endian systems,
but they generally have bswap'ing loads anyway so this shouldn't impose a
performance penalty.

### Kernels Section

#### Grammar

```none
  KERNELS_SECTION ::= INTEGER<"KernelCount"> OFFSET<"KernelNameOffset">*
```

The Kernels section defines a table of kernel names, directly corresponding to
the names in the MLIR host executor program. This allows references from the
Functions section to use dense indexes into the table.

The format of this section is an Integer count of the number of kernels in the
table, followed by 'count' Integer values which are offsets into the string
table for the kernel name.

#### Rationale

We want dense indexes from kernels in the Functions section, and the Host
Executor wants to resolve names to kernel implementations at startup time
anyway, as such, it makes sense for the host executor to build its own mutable
array in memory for kernels.

### Types Section

#### Grammar

```none
  TYPES_SECTION ::= INTEGER<"TypeCount"> OFFSET<"TypeNameOffset">*
```

The Types section defines a table of type names used by the host program, and is
laid out exactly the same as the [Kernels section](#kernels-section). The
[Functions section](#functions-section) uses Indexes into this section to
specify types of registers.

### FunctionIndex Section

#### Grammar

```none
  FUNCTION_INDEX_SECTION ::= INTEGER<"NumFunctions"> FUNCTION_ENTRY*
  FUNCTION_ENTRY         ::= BYTE<"FunctionKind"> OFFSET<"Function"> \
                             OFFSET<"Name"> INTEGER<"NumArguments"> \
                             INDEX<"Type">* INTEGER<"NumResults"> INDEX<"Type">*
```

The FunctionIndex section defines a table of functions in the BEF file, one for
each function, including the kind of this function (defined in
[`bef_encoding.h`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/bef/bef_encoding.h)),
an Offset into the [Functions Section](#functions-section), a name (an Offset
into the [Strings section](#strings-section), which may be an empty string), and
a list of argument and result types.

#### Rationale

This defines a symbol table for the BEF file, allowing clients to look up
functions by name. While we could intersperse this information into the
[Functions section](#functions-section) itself, doing so would require the
reader to make a pass over all of the functions ahead of time. We'd prefer to
have a quick index that the reader can scan at load time, deferring processing
of any individual function until it is needed.

### Functions Section

#### Grammar

```none
  FUNCTIONS_SECTION ::= FUNCTION*
```

The Functions section is a list of Function records emitted to the byte stream
and then addressed by an Offset indicated by the
[FunctionIndex section](#functionindex-section). Functions section is 4-byte
aligned.

#### Rationale

Regions in MLIR are a generalization of a unit of computation. The BEF format
and BEF executor support a subset of MLIR region features that are core to the
abstractions we need to model - in particular, while MLIR regions can have
multiple basic blocks in them, BEF Functions only support a single block - this
guarantees that they are always be a DAG of computation.

Functions are used for top-level functions, which BEF files have direct support
for (allowing lookup of functions by name) as well as in nested positions for
control flow and other concepts that occur with MLIR regions.

### Function Definition

#### Grammar

```none
  FUNCTION       ::= OFFSET<"Location"> REGISTER_TABLE KERNEL_TABLE \
                     RESULT_REGS BYTE<"AlignmentPadding">* KERNEL+

  REGISTER_TABLE ::= INTEGER<"NumRegs"> REGISTER_ENTRY*
  REGISTER_ENTRY ::= INTEGER<"NumUses">

  KERNEL_TABLE   ::= INTEGER<"NumKernels"> KERNEL_ENTRY*
  KERNEL_ENTRY   ::= OFFSET<"KernelOffset"> INTEGER<"NumOperands"> \
                     INTEGER<"StreamId">

  RESULT_REGS    ::= INDEX<"Register">*
```

Each function is defined by a location (an offset into the
[LocationPositions section](#locationpositions-section)), a register table, a
kernel table, a list of result registers, a list of kernels with 4-byte
alignment, and ends with a fixed32 integer of value zero.

The Register Table is a count of registers, and an entry for each register -
indicating the number of kernels in this section that use the register.

The Kernel Table for a function is a count of kernels, an offset (from the end
of the Kernel Table) of the start of the kernel, the number of operands that the
kernel has, and a stream id that is used to help runtime scheduling decisions,
e.g. successive kernels with the same stream id can be executed in the same
thread.

The kernel list that is following the Kernel Table contains all the kernels used
in this function. Note that every function has a pseudo kernel that is the
single entry point to the rest of the kernels. Specifially, a pseudo kernel
defines registers for function arguments and a pseudo register that is
conceptually used by kernels that takes no arguments.

The result registers specify the register values to return, and must align with
the function result types from the
[FunctionIndex section](#functionindex-section).

#### Rationale

These two tables are key to allowing the BEF executor to efficiently dispatch
and destroy kernels in a lock-free way. At startup time, the executor inflates
these two tables into arrays, resolving the type descriptors for the types, and
building a table of ready counts for the kernels. Having a table of kernels,
allow the use of kernel indexes.

### Kernel Definition

#### Grammar

```none
  KERNEL             ::= KERNEL_HEADER KERNEL_RESULT_TABLE KERNEL_BODY

  KERNEL_HEADER      ::= FIXED32<"KernelCode"> FIXED32<"KernelLocation"> \
                         FIXED32<"NumArguments"> FIXED32<"NumAttributes"> \
                         FIXED32<"NumFunctions"> FIXED32<"NumResults">

  KERNEL_RESULT_TABLE::= FIXED32<"NumUsedBys">*

  KERNEL_BODY        ::= FIXED32<KernelArgument>* FIXED32<KernelAttribute>* \
                         FIXED32<KernelFunction>* FIXED32<KernelResult>* \
                         FIXED32<KernelUsedBy>*
```

Each instance of a kernel includes a kernel header, a result table and a kernel
body. The kernel header consists of a opcode (an index into the Kernels table,
defined by the Kernels section), a location (an offset into the
[LocationPositions section](#locationpositions-section)) the numbers of
arguments, attributes, functions and results in the kernel body.

The result table contains NumResults fixed32 integers, indicating the number of
users for each corresponding result. The kernel body consists of zero or more
inputs (indexes into the Register Table for this function), zero or more
constants (offsets into the [Attributes section](#attributes-section)), zero or
more functions (indexes into the
[FunctionIndex section](#functionindex-section)), zero or more results (indexes
into the Register table for this function), and zero or more 'used by' records
(which are indexes into the Kernel Table for this function). 'used by' records
for the same result are grouped together and these groups are in the same order
of 'result' records. For example, if there are two results, A and B, and A has
one user (a0) and B has two users (b0, b1), then FIXED32<"NumResults"> will be
`0x00000002`, followed by two FIXED32<"NumUsedBys">, `0x00000001` `0x00000002`;
and in the kernel body there will be three "used by" records, a0, b0 and b1,
consecutively.

#### Rationale

This record allows us to efficiently form the array of arguments and results
that get passed to a kernel implementation function. When the kernel completes,
the UsedBy records allow us to efficiently trigger execution of data dependent
kernels if they are ready, by decrementing the "NumOperands" field for the
kernel.

### LocationStrings Section

#### Grammar

```none
  LOCATION_STRINGS_SECTION ::= NULL_TERMINATED_STRING*
```

This section contains a list of NULL terminated strings used by locations
section. A filename field of FileLineCol location and name field of Name
location are stored in this section.

#### Rationale

We choose to store these separately from the string table, because these
locations are only ever decoded in the case of an error. There is no reason to
dirty data cache lines with them, and we expect no reuse with other general
strings.

### Locations Section

#### Grammar

```none
  LOCATIONS_SECTION ::= LOCATION*

  LOCATION = <UNKNOWN_LOC | FILELINECOL_LOC | NAME_LOC | CALLSITE_LOC
              | FUSED_LOC>

  UNKNOWN_LOC ::= `0x00`

  FILELINECOL_LOC ::= `0x01` OFFSET<"Filename"> INTEGER<"LineNum"> \
                      INTEGER<"ColumnNum">

  NAME_LOC ::= `0x02` OFFSET<"Name"> LOCATION<"Child">

  CALLSITE_LOC ::= `0x03` LOCATION<"Callee"> LOCATION<"Caller">

  FUSED_LOC ::= `0x04` INTEGER<"NumLocations"> LOCATION*
```

This section contains a list of "locations", which can have five different
types: Unknown, FileLineCol, Name, CallSite location, and Fused locations.
Strings used in FileLineCol and Name locations are added to the LocationStrings
section.

### AttributeTypes Section

#### Grammar

```none
  ATTRIBUTE_TYPES_SECTION   ::= INTEGER<"NumAttributes"> ATTRIBUTE_TYPE_ENTRY*
  ATTRIBUTE_TYPE_ENTRY      ::= OFFSET<"Attributes"> BYTE<"AttributeType">
```

The AttributeTypes section is an optional section which is not needed for BEF
execution, but this section is necessary for bef-to-mlir conversion. This
section describes the type information for each attribute (specified by an
Offset into [Attributes section](#attributes-section)). An attribute type is an
one-byte enum described in [Attributes section](#attributes-section) and defined
in bef_encoding.h.

### AttributeNames Section

#### Grammar

```none
  ATTRIBUTE_NAMES_SECTION ::= INTEGER<"NumFunctions"> KERNEL_TABLE*
  KERNEL_TABLE            ::= INTEGER<"NumKernels"> OFFSET<"AttributeName">*
```

The AttributeNames sections is an optional section which is not needed for BEF
execution, but this section is necessary for bef-to-mlir conversion. This
section describes the attribute names used by each kernel. There is a 1:1
mapping between Kernel Table in this section and Function Entries in
[FunctionIndex section](#functionindex-section). And there is a 1:1 mapping
between kernel entries in this section and kernel entries in Functions section's
kernel table. Each kernel entry contains any number of AttributeNames.
AttributeName is an offset to Strings section that specifies the name of the
attribute used by this kernel.

### RegisterTypes Section

#### Grammar

```none
  REGISTER_TYPES_SECTION ::= INTEGER<"NumFunctions"> REGISTER_TYPE_TABLE
  REGISTER_TYPE_TABLE    ::= INTEGER<"NumRegs"> INDEX<"Types">*
```

The RegisterTypes section is an optional section which is not needed for BEF
execution, but this section is necessary for bef-to-mlir conversion. This
section describes the type information for registers in each function. The
RegisterType Table is a count of registers, and an entry for each register
indicating its index into [Types section](#types-section). There is a 1:1
mapping between RegisterType Tables and Function Entries in
[FunctionIndex section](#functionindex-section). And it is a 1:1 mapping between
types in a RegisterType Table and registers in the RegisterTable of the
corresponding Function.
