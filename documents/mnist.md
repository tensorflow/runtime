# MNIST Integration Test

<!--* freshness: {
  owner: 'chuanhao'
  reviewed: '2021-05-13'
} *-->

<!-- TOC -->

The goal of this test is to demonstrate that the TFRT can run a simple MNIST
model in inference mode (forward-pass only). This is a "hello-world" integration
test for TFRT graph-mode. This document describes the setup of the MNIST
integration test and walks through the instructions to run it.

## Test Setup

This test has the following components:

*   [mnist.mlir](https://github.com/tensorflow/runtime/blob/master/integrationtest/mnist/mnist.mlir)
    describes the MNIST model.

*   [test_data/mnist_tensors.btf](https://github.com/tensorflow/runtime/blob/master/integrationtest/mnist/test_data/mnist_tensors.btf)
    Test data, test label, and trained MNIST model weights stored in the custom
    [Binary Tensor Format](binary_tensor_format.md) (BTF).

*   [mnist_tensor_kernels.cc](https://github.com/tensorflow/runtime/blob/master/backends/cpu/lib/ops/test/mnist_tensor_kernels.cc)
    is a set of kernel implementations in C++.

### MNIST model in MLIR

The MNIST model is also written in
[mnist.mlir](https://github.com/tensorflow/runtime/blob/master/integrationtest/mnist/mnist.mlir).
Today this model file has to be written by hand and the tensor shapes hard-coded
in. As the MLIR graph compiler gets built, we can generate this MLIR model file
from a Tensorflow GraphDef or SavedModel, and the Tensor shapes should be done
as a shape inference compiler pass.

### Kernel Implementations

All kernels invoked in the MNIST model file, e.g., `tfrt_test.matmul.f32.2`,
`tfrt_test.broadcast.f32.1` are implemented in
[mnist_tensor_kernels.cc](https://github.com/tensorflow/runtime/blob/master/backends/cpu/lib/ops/test/mnist_tensor_kernels.cc).

## Test Instructions

We trained a MNIST model to convergence and saved the model weights to the BTF
file.

Use TFRT to run the MNIST model in inference mode.

```shell
$ bazel test //integrationtest/mnist:mnist.mlir.test
```

The test runs MNIST inference and checks that we get the expected average
accuracy on the test dataset.
