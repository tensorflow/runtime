# High Level Overview of TFRT

<!--* freshness: {
  owner: 'hongm'
  reviewed: '2020-04-26'
} *-->

<!-- TOC -->

TFRT is a new TensorFlow runtime. Leveraging MLIR, it aims to provide a unified,
extensible infrastructure layer with best-in-class performance across a wide
variety of domain specific hardware. This approach provides efficient use of
multithreaded host CPUs, supports fully asynchronous programming models, and
focuses on low-level efficiency.

For a high level introduction of the project goals and end user benefits, please
watch our [TF Dev Summit 2020 presentation](https://youtu.be/15tiQoPpuZ8).

For an overview of the system components, please refer to
[Library and Subsystem Overview](subsystems.md).

For the project design philosophy, please refer to
[TFRT Design Philosophy](design_philosophy.md).

To contribute to the codebase, please refer to
[TFRT C++ style guide](style_guide.md).
