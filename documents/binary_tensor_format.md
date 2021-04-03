# Binary Tensor Format

<!--* freshness: {
  owner: 'jingdong'
  reviewed: '2021-04-02'
} *-->

<!-- TOC -->

WARNING: This format is experimental.

This document describes the file format for storing tensors, known as the Binary
Tensor Format, "BTF".

A BTF file is a self-describing file format that stores arbitrary number of
Tensors. It is formed as a byte stream whose top-level structure is a "File
Header" followed by a list of "Tensor Record"s.

<!-- TODO(doak): Make a diagram for the format -->

```none
FILE                 ::= FILE_HEADER TENSOR_RECORD*
FILE_HEADER          ::= NUM_TENSORS TENSOR_RECORD_OFFSET*
TENSOR_RECORD        ::= TENSOR_HEADER TENSOR_PAYLOAD TENSOR_PADDING
TENSOR_HEADER        ::= RANK DTYPE LAYOUT RESERVED_SPACE

NUM_TENSORS          ::= uint64_t
TENSOR_RECORD_OFFSET ::= uint64_t
RANK                 ::= uint64_t
DTYPE                ::= uint8_t
RESERVED_SPACE       ::= 0 (6 byte wide)
TENSOR_PADDING       ::= 0 (0-7 bytes)
```

A well-formed BTF file should have `NUM_TENSOR` of `TENSOR_RECORD_OFFSET`s and
`TENSOR_RECORD`s, where each `TENSOR_RECORD_OFFSET` points to the byte offset of
the corresponding `TENSOR_RECORD`. Padding is added by `TENSOR_PADDING` to each
`TENSOR_RECORD` to ensure that the `TENSOR_RECORD` size is a multiple of 64 bits
(and therefore that `TENSOR_RECORD_OFFSET` is a multiple of 8). Padding may be
omitted for the last record.

`DIMS` should consist of `RANK` number of `uint64_t`. The number of bytes in
`TENSOR_DATA` should be the same as specified by `DIMS` and `DTYPE`.

The currently supported `DTYPE` and their encoded numeric values are:

```none
int8: 0
int16: 1
int32: 2
int64: 3
float32: 4
float64: 5
```

Each tensor header contains an 8-bit "magic" number identifying the layout used
to represent the tensor. The magic numbers currently reserved are:

<!-- TODO(doak): It's better to have these magic numbers and their names defined
     centrally and import it here and in the languages that parse it to ensure
     they're kept in sync -->

```none
LAYOUT := RMD (uint8_t)
RMD := 0 (Row-Major Dense tensor)
COO := 2 (COrdinate-Order sparse tensor)
```

Specific tensor payload are implemented for different tensor layouts. The
payloads currently implemented are:

```none
TENSOR_PAYLOAD       ::= DENSE_TENSOR_PAYLOAD | COO_TENSOR_PAYLOAD

DENSE_TENSOR_PAYLOAD ::= DIMS DENSE_TENSOR_DATA
DENSE_TENSOR_DATA    ::= uint8_t*

COO_TENSOR_PAYLOAD   ::= DIMS INDICES VALUES
INDICES              ::= DENSE_TENSOR_PAYLOAD
VALUES               ::= DENSE_TENSOR_PAYLOAD

DIMS                 ::= uint64_t*
```

`INDICES` is a `DENSE_TENSOR_PAYLOAD` with the following attributes:

-   Rank: 2
-   Dims: `(N, RANK)`, where `N` is the number of nonzero indices in the sparse
    tensor.
-   Type: `uint64_t`

`VALUES` is a `DENSE_TENSOR_PAYLOAD` with the following attributes:

-   Rank: 1
-   Dims: `(N)`
-   Type: the type of the COO tensor.
