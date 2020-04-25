# Error Handling

<!--* freshness: {
  owner: 'lauj'
  reviewed: '2020-03-27'
} *-->

<!-- TOC -->

## Reporting Errors From Kernels

Kernels may report error in a few ways, but most use
`KernelErrorHandler::ReportError`, which notifies HostContext of the error, sets
results that are concrete but unavailable to error, and fills in any unset
result values with `AsyncValue`s that contain errors. For example:

```c++
static void HexTestFail(Result<int32_t> error_out, KernelErrorHandler handler) {
  // Sets error_out to an AsyncValue with an error message.
  handler.ReportError("something bad happened");
}
```

Note that `ReportError` only sets the kernel's *unavailable* concrete results to
error state and the kernel's *unset* results to an `AsyncValue` with error. For
example:

```c++
static void HexTestPartialFail(Result<int32_t> one, Result<int32_t> error_out,
                               KernelErrorHandler handler) {
  one.Emplace(1);
  error_out.Allocate<int32_t>();
  // Only sets error_out, which is an unavailable ConcreteAsyncValue, to error
  // state. `one` is untouched.
  handler.ReportError("something bad happened");
}
```

```c++
static void HexTestPartialFail(Result<int32_t> one, Result<int32_t> error_out,
                               KernelErrorHandler handler) {
  one.Emplace(1);
  // Only sets error_out to an error AsyncValue. `one` is untouched.
  handler.ReportError("something bad happened");
}
```

`HexTestPartialFail` returns two values: `1` and an `AsyncValue` with error.
This allows for fine-grained error handling: a kernel may partially succeed by
returning a mix of valid values and error values.

`ReportError` can only report an error when there are unset results or
unavailable concrete results. For example this kernel triggers an assertion
failure:

```c++ {.bad}
static void HexNoErrorReported(Result<int32_t> out, KernelErrorHandler handler) {
  out.Emplace(1);
  // No unset results at this point. ReportError can't report an error.
  handler.ReportError("something bad happened");
}
```

Asynchronous kernels must synchronously allocate their results. Asynchronous
kernels may report errors through unavailable results, like this:

```c++ {.good}
static void HexNoErrorReportedAsync(Result<int32_t> out, HostContext* host,
                                    KernelFrame* frame) {
  host->EnqueueWork(
    [out_ref = FormRef(out.Allocate()), frame = *frame]() mutable {
      if (/*error condition*/) {
          frame.ReportError("something bad happened");
      } else {
          // No error: Set the unavailable value to 1.
          out_ref->emplace(1);
      }
    });
}
```

Note: In this example, it is not possible to asynchronously allocate `out`.
Kernels must synchronously allocate values for all their results.

Note: [Asynchronous kernels](async_value.md#asynchronous-kernels) explains why
we need `FormRef`.

Note: In this example, it is currently necessary to use `KernelFrame` instead of
`KernelErrorHandler` because `KernelErrorHandler` holds a pointer to the
`KernelFrame`, but the `KernelFrame` is destroyed when the kernel returns. So if
`HexNoErrorReportedAsync` tried to use `KernelErrorHandler` asynchronously, it
would dereference an invalid pointer.

## BEFExecutor Error Handling

If any argument values to a kernel are errors, BEFExecutor does not run the
kernel.

Instead, the executor automatically propagates one of the kernel's argument
errors to all of the kernel's results. This lets the executor quickly skip the
parts of the dataflow graph that are invalidated by errors. Example:

```c++
func @test_partial_fail() -> !hex.chain {
  // Sets %x to 1 and %y to an error.
  %x, %y = "tfrt_test.partial_fail"() : () -> (i32, i32)

  %ch0 = hex.new.chain
  // This hex.print will run.
  hex.print.i32 %x, %ch0
  // This hex.print won't run. The executor propagates the error in %y to %ch1.
  %ch1 = hex.print.i32 %y, %ch0

  // Returns an error.
  hex.return %ch1 : !hex.chain
}
```

In this example, we call the `HexTestPartialFail` kernel we defined earlier. The
results are bound to `%x` and `%y`. The first print runs as usual, because all
its arguments are valid. But the executor does not run the second print, because
`%y` is an `AsyncValue` with error. Instead, the executor automatically
propagates the error in `%y` to `%ch1`, and so this function returns an error
instead of a `!hex.chain`.

Errors are just values that prevent execution of downstream kernels. Unused
errors behave like any other unused value: they're completely ignored by the
executor. This function returns 1, even though it internally generated an unused
`AsyncValue` with error:

```c++
func @test_ignore_error() -> i32 {
  %unused = "tfrt_test.fail"() : () -> i32
  %x = hex.constant.i32 1
  hex.return %x : i32
}
```

More generally, a function can only return an error if there is dataflow from
the error to any of the function's return values. This should generally happen
naturally, because all side effecting operations are connected with
[Chains](explicit_dependency.md), and non-side effecting operations generally
produce values that are used, or don't matter for the computation in the first
place.

### Cancellation

To cancel execution, call `HostContext::CancelExecution`. This makes the
executor pass an `AsyncValue` with error as every kernel's argument. This
effectively makes the executor skip all remaining kernel invocations.

`CancelExecution` does not interrupt currently running kernels. It is up to
implementors of potentially long-running kernels to periodically check
`HostContext::GetCancelAsyncValue` and return early when CancelAsyncValue is
not-null. For example,
[`HexRepeatI32`](https://cs.opensource.google/tensorflow/runtime/+/master:lib/basic_kernels/control_flow_kernels.cc)
checks CancelAsyncValue once per loop iteration.

## Managing Resource Lifetimes In The Presence Of Errors

It's best to use AsyncValue's reference counts to keep resources alive for as
long as they're needed. The executor extends an AsyncValue's lifetime until
there are no more users for the value. AsyncValue will invoke the resource's
destructor when the resource is no longer needed.

Some users may want more control over resource lifetime. As an example, suppose
we're managing an accelerator's memory. We could define an `AllocateAccelMemory`
kernel that returns `AccelMemory`. `AccelMemory` is wrapped in AsyncValue, and
`AccelMemory`'s destructor deallocates the memory:

```c++
class AccelMemory {
  public:
  AccelMemory() {
    // Allocate accelerator memory.
  }
  ~AccelMemory() {
    // Deallocate accelerator memory.
  }
};

static void AllocateAccelMemory(Result<AccelMemory> accel_memory_out) {
  accel_memory_out.Emplace();
}
```

This works, but the memory will remain allocated as long as there are
outstanding users of the AsyncValue produced by `AllocateAccelMemory`.

If we require manual control over the memory's lifetime, we might try writing
kernels like these:

```c++ {.bad}
static void AllocateAccelMemory(Result<AccelMemory*> accel_memory_out) {
  accel_memory_out.Emplace(new AccelMemory());
}

static void DeallocateAccelMemory(Argument<AccelMemory*> accel_memory_in) {
  delete accel_memory_in.get();
}
```

Note that these kernels operate on `AccelMemory*` (pointer type), instead of
`AccelMemory` in the earlier example.

But recall that cancellation makes the executor skip all remaining kernel
invocations. This means that `DeallocateAccelMemory` *may not run at all* if
execution is cancelled, which would leak accelerator memory.

Note: In this scenario, AsyncValue's reference counting *does* automatically
deallocate the 8 bytes for `AccelMemory*`, because that's what the AsyncValue
actually holds. AsyncValue won't automatically deallocate the actual
`AccelMemory` object because it was separately allocated.

To address this issue, use `std::unique_ptr` instead of a raw pointer:

```c++ {.good}
static void AllocateAccelMemory(
    Result<std::unique_ptr<AccelMemory>> accel_memory_out) {
  accel_memory_out.Emplace(new AccelMemory());
}

static void DeallocateAccelMemory(
    Argument<std::unique_ptr<AccelMemory>> accel_memory_in) {
  accel_memory_in->reset();
}
```

`unique_ptr` provides manual control over `AccelMemory`'s lifetime: we can
invoke the `DeallocateAccelMemory` kernel whenever we want to force early
deallocation, and `unique_ptr` guarantees that the memory will be deallocated
even if `DeallocateAccelMemory` is never called, which can happen if execution
is cancelled or an error occurs.

## Error Message Text Guidelines

These are guidelines, not rules: they can and will conflict with each other.

Remember that error messages are for human consumption and are not generally
parsed by code. Try to write error messages that help whoever reads them.
Resolve guideline conflicts by thinking about what would be more helpful for the
user. And remember that you might not be the user - what's helpful to you might
confuse others.

### Use Complete Sentences

Capitalize the first letter of each sentence and use punctuation.

### Be Specific And Succinct

Be specific about what caused the error. Describe the error briefly and clearly.
If the error can't be described briefly, provide a link to documentation rather
than writing a long error message.

### Provide Higher Level Context

Kernel errors are produced at a low level, and propagate through many layers of
software before reaching the user. Users are often unfamiliar with the specific
kernel that produced the error. Provide higher level error context when
feasible.
