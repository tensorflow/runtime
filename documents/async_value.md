# AsyncValue

<!--* freshness: {
  owner: 'lauj'
  reviewed: '2020-04-27'
} *-->

<!-- TOC -->

This document explains some of the concepts behind `AsyncValue`.

## `AsyncValue`

[`AsyncValue`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/async_value.h)
is an abstract future data type. It's conceptually similar to
[std::future](https://en.cppreference.com/w/cpp/thread/future), except that
`AsyncValue` does not let callers wait until the value becomes available.
Instead, the caller enqueues a closure that uses the value with
`AsyncValue::AndThen`. `AsyncValue::emplace` will run any enqueued closures when
the value becomes available. This approach is similar to
[continuation passing](https://en.wikipedia.org/wiki/Continuation-passing_style).

### Reference counting

`AsyncValue`s are reference counted, or refcounted. A newly constructed
`AsyncValue` has a refcount of 1, the refcount may be manipulated with `AddRef`
and `DropRef`, and any `AsyncValue` with a refcount of 0 is immediately
destroyed.

[`RCReference`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/support/ref_count.h)
is a smart pointer class that calls `DropRef` when it is destroyed. It's
conceptually similar to
[std::shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr). Kernel
implementations typically use
[`AsyncValueRef`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/async_value_ref.h)
described [below](#asyncvalueref) rather than `RCReference`, unless type erasure
is needed.

### Type Erasure

`AsyncValue` is type-erased: users can manipulate an `AsyncValue` without
knowing its contained type. For example, given an `AsyncValue`, a user can check
its availability with `AsyncValue::IsAvailable()` or enqueue a closure with
`AsyncValue::AndThen()`, without knowing the actual type contained in the
`AsyncValue`. Type information is only needed when *accessing* the contained
data, for example with `AsyncValue::get<T>()` or `AsyncValue::emplace<T>()`.

`AsyncValue`'s type erasure simplifies code that does not need to access the
contained data. For example, `BEFExecutor` uses one generic code path based on
`AsyncValue`s to forward results from one kernel to arguments to another kernel.

#### `AsyncValueRef`

`AsyncValueRef` improves type safety when type erasure is not needed.

`AsyncValueRef` is functionally equivalent to `RCReference<AsyncValue>`, except
that it carries type information for the contained data. For example these two
code blocks are equivalent:

```c++
// AsyncValueRef version. Note how int32_t is specified in AsyncValueRef's
// template parameter. Prefer this version when type erasure is not needed.
AsyncValueRef<int32_t> int_value = ...;
int_value.emplace(42);
```

```c++
// RCReference<AsyncValue> version. Note how int32_t is specified in emplace's
// template parameter. Use this pattern when type erasure is needed.
RCReference<AsyncValue> int_value = ...;
int_value->emplace<int32_t>(42);
```

For example,
[`TensorHandle`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/tensor_handle.h)
is constructed from two `AsyncValues`, one which must contain `TensorMetadata`
and another which must contain `Tensor`.

Because these two `AsyncValues` must contain specific types, prefer defining
`TensorHandle`s constructor like this:

```c++ {.good}
// Prefer AsyncValueRef for TensorHandle's constructor because the AsyncValues
// must contain specific types.
TensorHandle(AsyncValueRef<TensorMetadata> async_metadata,
             AsyncValueRef<Tensor> tensor);
```

instead of like this:

```c++ {.bad}
// Type erasure is not useful here because the AsyncValues must contain specific
// types.
TensorHandle(RCReference<AsyncValue> async_metadata,
             RCReference<AsyncValue> tensor);
```

In this example, type erasure is not useful because `TensorHandle`'s constructor
expects `AsyncValue`s that contain specific types. `TensorHandle`'s constructor
does not accept `AsyncValue`s containing arbitrary types.

##### `AsyncValueRef` conversions

`AsyncValueRef` and `RCReference<AsyncValue>` are functionally equivalent so we
often convert between the two as needed. For example:

```c++
// Convert RCReference<AsyncValue> to AsyncValueRef. This is an explicit
// conversion because the contained type must be specified.
RCReference<AsyncValue> generic_value = ...;
AsyncValueRef<int32_t> typed_value(std::move(generic_value));
```

```c++
// AsyncValueRef is implicitly convertible to RCReference<AsyncValue>.
AsyncValueRef<int32_t> typed_value = ...;
RCReference<AsyncValue> generic_value = std::move(typed_value);
```

## Calling convention

The same calling convention is used for both:

Running a kernel
:   `BEFExecutor` is the caller, and the `kernel_fn` is the callee.

Executing a BEF function
:   a caller invokes `BEFExecutor::Execute`, usually via `BEFFunction::Execute`,
    and a newly constructed `BEFExecutor` is the callee.

### Caller's responsibilities

Arguments
:   The caller must keep all arguments alive. This is known as a "+0" calling
    convention.

Return Values
:   The caller takes ownership of the callee's return values - they are returned
    as "+1" values, which means one ref transfers from the callee to the caller.
    The caller becomes responsible for destroying these "+1" values, or
    transferring the caller's ref to someone else as a "+1" value.

Note: Ownership of a return value can transfer many times. For example, a kernel
generates the return value for a BEF function. The kernel initially owns the
return value, then ownership transfers to the `BEFExecutor` that's running the
function, then ownership transfers to whoever invoked `BEFExecutor` on the
function.

### Callee's responsibilities

Arguments
:   Arguments are passed to the caller without transferring ownership, as "+0"
    values. The callee must not destroy any argument.

Return Values
:   The callee returns "+1" values. Return values are created by the callee, and
    ownership is transferred to the caller.

    If the callee wants to extend the lifetime of an argument beyond the
    invocation of the kernel, then the callee must add and remove additional
    references to keep the argument alive.

    For example, if a kernel wants to do some deferred work that needs an
    argument, the kernel should call `Argument::ValueRef()` to create a new
    `AsyncValueRef` for the argument, and transfer ownership of the
    `AsyncValueRef` to the deferred work function:

    ```c++
     static void MyKernel(Argument<int32_t> in, HostContext* host) {
       host->EnqueueWork([in_ref = in.ValueRef()] {
         // Retrieve the int32_t value with in_ref.get().
       });
     }
    ```

## Implementing a kernel

For synchronous kernels, the `TFRT_KERNEL` macro manages all calling convention
responsibilities. For asynchronous kernels, keep the following in mind:

Arguments are "+0" values, so use `AsyncValueRef` to keep argument reference
counts balanced.

Return values are "+1" values, so each return value must have one reference that
transfers to the caller. Newly constructed `AsyncValue`s start with one
reference, and this reference typically transfers to the caller. `AsyncValueRef`
is still used to keep any additional references balanced.

Asynchronous kernels must add an additional reference on any argument or return
value that they will use asynchronously. For arguments, these additional refs
are typically added and maintained with `Argument::ValueRef()`. For example:

```c++
static AsyncValueRef<int32_t> AsyncCopy(Argument<int32_t> in,
                                        HostContext* host) {
  return host->EnqueueWork(
      [in_ref = in.ValueRef()] { return in_ref.get(); });
}
```

For kernels that directly return `AsyncValueRef` like the example above, these
additional refs are automatically added by the `TFRT_KERNEL` macro. For kernels
that use the `Result` template, the additional ref is added by
`Result::Allocate()`. For example:

```c++
static void TestAsyncCopy(Argument<int32_t> in, Result<int32_t> out,
                          HostContext* host) {
  host->EnqueueWork([in_ref = in.ValueRef(), out_ref = out.Allocate()] {
    out_ref.emplace(in_ref.get());
  });
}
```

See [Asynchronous kernels](#asynchronous_kernels) for more details.

When return values are identical to argument values, a kernel may reuse the
argument values to avoid copies. If a kernel returns N references to an
argument, the kernel must increase the refcount of that argument by N. These
refs are usually added by `Result::Set`:

```c++
static void CopyToTwo(Argument<int32_t> in, Result<int32_t> out_1,
                      Result<int32_t> out_2, HostContext* host) {
  out_1.Set(in);
  out_2.Set(in);
}
```

## Implementation details

The following sections describe some `BEFExecutor` internals.

### `ConcreteAsyncValue` and `IndirectAsyncValue`

`internal::ConcreteAsyncValue` (with the `internal::` prefix omitted below for
brevity) and `IndirectAsyncValue` are the two key `AsyncValue` implementations
in the executor. Clients generally use abstract `AsyncValue`s and don't need to
understand the implementations.

Most instances of `AsyncValue` are `ConcreteAsyncValue`s.
`ConcreteAsyncValue<T>` is an `AsyncValue` with inline storage for an instance
of `T`, so we create `ConcreteAsyncValue`s whenever the value's type is known.

`IndirectAsyncValue`s are used when an `AsyncValue` is needed, but the C++ type
is not yet known. As such, these are pretty uncommon, and mostly used for
control flow kernels and in the internals of `BEFExecutor`.

We generally try to avoid creating `IndirectAsyncValue`s for efficiency. The
following sections show some examples of how `ConcreteAsyncValue` and
`IndirectAsyncValue` are used.

#### Returning a new result value from a kernel

Returning a new result value from a kernel is the most common way new
`AsyncValues` are constructed. When a kernel returns a value, the kernel usually
knows the return type, so most kernels construct and return a new
`ConcreteAsyncValue` via `MakeAvailableAsyncValueRef`.

Some kernels do not know their return types when they return. Examples include
non-strict control flow kernels like `tfrt.if` and `tfrt.repeat`. When the
return types are unknown, these kernels return `IndirectAsyncValue` via
`HostContext::MakeIndirectAsyncValue()`.

#### Returning a result from an asynchronous BEF function may require an `IndirectAsyncValue`

When executing an asynchronous BEF function, there are two possibilities:

1.  The asynchronous BEF function completes execution before
    `BEFExecutor::Execute` returns.
2.  The asynchronous BEF function is still running when `BEFExecutor::Execute`
    returns.

Case (1) is simple: the kernel that produced the BEF function's return value has
already run, so we directly return the kernel's result value. The `BEFExecutor`
does not construct a new `AsyncValue` in this case.

Case (2) is more complex. `Execute` must supply return `AsyncValue`s to the
caller, but the kernel that produces the BEF function's return value has not yet
run. The BEF function's return types are unknown when `Execute` returns. In case
(2), `Execute` returns `IndirectAsyncValue`s. When the BEF function's return
values become available, these `IndirectAsyncValue`s are forwarded to the actual
return values. Example:

```c++
func @indirect_async_return(%c1: i32) -> i32 {
  // %c1 is an available ConcreteAsyncValue.
  %c1 = tfrt.constant.i32 1

  // %v2 is an unavailable ConcreteAsyncValue (not an IndirectAsyncValue!).
  %v2 = "tfrt.async_add.i32"(%c1, %c1) : (i32, i32) -> i32

  // %v3 is empty: this async_add can't run because %v2 is not available.
  %v3 = "tfrt.async_add.i32"(%v2, %v2) : (i32, i32) -> i32

  // Executor creates an unavailable IndirectAsyncValue for %v3 because it must
  // return an AsyncValue to indirect_async_return's caller.
  tfrt.return %v3 : i32
}
```

#### Passing an argument to a non-strict kernel may require an `IndirectAsyncValue`

Non-strict kernels can start executing before all their arguments are ready. So
when passing an argument to a non-strict kernel, there are two possibilities:

1.  All arguments to the non-strict kernel are ready.
2.  Some argument to the non-strict kernel is not ready.

Case (1) is simple; we just pass the argument `AsyncValue`s as usual. The
`BEFExecutor` does not construct a new `AsyncValue` in this case.

In case (2), an argument is not available. The argument's type is also not
available, because the kernel that generates the argument has not yet run. So
the executor creates `IndirectAsyncValue`s for each unavailable argument to the
non-strict function. When the kernel's arguments become available, these
`IndirectAsyncValue`s are forwarded to the actual return values. Example:

```c++
func @return_first_arg(%x: i32, %y: i32) -> i32 {
  tfrt.return %x : i32
}

func @indirect_async_arg() {
  // %c1 is an available ConcreteAsyncValue.
  %c1 = tfrt.constant.i32 1

  // %v2 is an unavailable ConcreteAsyncValue (not an IndirectAsyncValue!).
  %v2 = "tfrt.async_add.i32"(%c1, %c1) : (i32, i32) -> i32

  // %v3 is empty: this async_add can't run because %v2 is not available.
  %v3 = "tfrt.async_add.i32"(%v2, %v2) : (i32, i32) -> i32

  // BEFExecutor allocates an IndirectAsyncValue for %v3 because it must provide
  // an AsyncValue for return_first_arg's second argument. Note: all tfrt.call
  // invocations are nonstrict by default.
  %x = tfrt.call @return_first_arg(%c1, %v3) : (i32, i32) -> i32

  tfrt.return
}
```

### `BEFExecutor::Execute`

These basic principles of "+0" arguments and "+1" return values can compose in
complex ways.

For example:

```c++
func @share(%x: i32) -> (i32, i32) {
  tfrt.return %x, %x : i32, i32
}

func @caller() -> (i32, i32) {
  %c1 = tfrt.constant.i32 1
  %c2, %c3 = tfrt.call @share(%c1) : (i32) -> (i32, i32)
  tfrt.return %c2, %c3
}
```

In this example, `BEFExecutor::Execute` first executes `caller`, which contains
a `tfrt.call`, which invokes a BEF function. `tfrt.call` creates a new
`BEFExecutor` and recursively invokes Execute on `share`. `tfrt.call`'s kernel
arguments and results (`%c1`, `%c2`) become the inner `BEFExecutor`'s BEF
function arguments and returns (`%x`).

In this example, only one `AsyncValue` is created. The system does not copy
`AsyncValue`s: `%x`, `%c1`, `%c2`, and `%c3` are all just pointers to the same
`AsyncValue`.

### Asynchronous kernels

The executor `DropRef`s an `AsyncValue` as soon as it believes the `AsyncValue`
is no longer needed. This can be a problem for asynchronous kernels. Consider
this example:

```c++
func @foo() -> () {
  %in = tfrt.constant.i32 1
  %out = "async_kernel"(%in) : (i32) -> i32
  tfrt.return
}
```

`%in` and `%out` are unused after `async_kernel` returns control to
`BEFExecutor`. So `BEFExecutor` will `DropRef` `%in` and `%out` as soon as
`async_kernel` returns, and those `DropRef`s may destroy the values. But
`async_kernel` is asynchronous, and `async_kernel` may still be running in the
background when it returns control to `BEFExecutor`. If `async_kernel` needs to
use its arguments or result values asynchronously, `async_kernel` *must maintain
its own `AsyncValueRef`s on those values* to ensure that they remain live.

### Returning an `IndirectAsyncValue`

When a kernel generates a BEF function's return value, the executor may have
already allocated an `IndirectAsyncValue` for the return value. If so, the
executor forwards the `IndirectAsyncValue` to the underlying value, and
transfers ownership of the +1 return value to `IndirectAsyncValue`.

The underlying `AsyncValue` is typically a `ConcreteAsyncValue` with refcount 1,
and so the underlying return value typically gets destroyed by
`IndirectAsyncValue::DestructPayload`.

### `RegisterInfo::user_count`

The executor tracks `AsyncValues` in `RegisterInfo`s. These registers are used
to forward results from one kernel to arguments to another kernel. Registers are
just pointers to `AsyncValue`s, and multiple `RegisterInfo`s can point at the
same `AsyncValue`.

BEF files store [`NumUses`](binary_executable_format.md#function-definition) for
each register. This is kept in `RegisterInfo::user_count`. `user_count` is
function scoped: it does not track users across BEF function boundaries. When
the register is set, we set the `AsyncValue`'s initial refcount to `user_count`.
In this example, the `user_count` for `%x` is 4:

```c++
// Initial refcount for %x is 4 (1 + 2 + 1).
func @foo() -> (i32, i32) {
  // 1 use of %x here: setting %x counts as a use (see next section).
  %x = tfrt.constant.i32 42

  // 2 uses of %x here.
  %y = tfrt.add.i32 %x, %x

  // 1 more use of %x here.
  tfrt.return %x, %y : i32, i32
}
```

BEF files also store
[`UsedBy` records](binary_executable_format.md#kernel-definition) for each
kernel's result. These records direct the executor to downstream kernels that
depend on the produced value. In the example above, `%x` has one `UsedBy` record
for `tfrt.add.i32`, because `tfrt.add.i32` depends on `%x`. After running a
kernel, the executor drops one ref every time a register is used. So in the
example above, after executing `tfrt.constant.i32`, `%x`'s refcount drops to 3.
And after executing `tfrt.add.i32`, `%x`'s refcount drops to 1.

Note: *`tfrt.return` is not a kernel*. In the example above, `%x` does *not*
have a `UsedBy` record for `tfrt.return`. Registers passed to `tfrt.return` are
counted in `user_count`, but do not have `UsedBy` records. This effectively
gives each return value an extra ref, because `user_count` always adds a ref for
`tfrt.return`, but there is no corresponding `DropRef`, because there is no
kernel to execute for `tfrt.return`. The executor uses this trick to transfer
"+1" return values to the executor's caller: in the example above, note how `%x`
conveniently has a refcount of 1 after `tfrt.add.i32` executes, so the executor
does not need to adjust the returned value's refcount.

#### Setting a register counts as a use

Setting a register must count as a use to properly handle unused
`IndirectAsyncValue`s. Consider this example:

```c++
// BEFExecutor allocates an IndirectAsyncValue for this function's return.
func @make_indirect() -> i32 {
  %c1 = tfrt.constant.i32 1
  %v2 = "tfrt.async_add.i32"(%c1, %c1) : (i32, i32) -> i32
  // Executor can not execute this tfrt.async_add.i32 because %v2 is not
  // available.
  %v3 = "tfrt.async_add.i32"(%v2, %v2) : (i32, i32) -> i32
  tfrt.return %v3 : i32
}

func @caller() {
  // The returned IndirectAsyncValue is unused, so naively we'd immediately
  // destroy the IndirectAsyncValue. But tfrt.async_add.i32 has not yet
  // forwarded the IndirectAsyncValue. Setting this IndirectAsyncValue must
  // count as a use to keep it alive until it is forwarded.
  %unused = tfrt.call @make_indirect() : (i32) -> i32

  tfrt.return
}
```

In this example, the executor must allocate an `IndirectAsyncValue` for
`make_indirect`'s return value, because `%v3` is empty when `make_indirect`
returns control back to `caller`.

But `caller` does not use this return value. If we counted users naively, `%v3`
would be returned to `caller` with only 1 ref (returning `%v3` would be the only
use). After control returns from `tfrt.call`, the executor would drop the only
ref because `%unused` has no users. This would prematurely destroy the
IndirectAsyncValue, before `make_indirect`'s `tfrt.async_add.i32` can
asynchronously forward it.

We fix this by always counting register assignment as an additional use. This
increases every register's `RegisterInfo::user_count` by 1. In this example,
`%v3` has a `user_count` of 2 (setting `%v3` counts as one use and returning
`%v3` counts as another), so `%v3` has an initial refcount of 2. Note that the
second `tfrt.async_add.i32` only drops its ref on `%v3` *after* it
asynchronously assigns `%v3`.

So when control returns to `caller`, the `IndirectAsyncValue` still has its
initial refcount of 2, and `caller` drops the +1 ref it received from
`make_indirect` because `%unused` has no users. At this point the
`IndirectAsyncValue` still has one ref, and so it remains alive until
`tfrt.async_add.i32` finishes forwarding the `IndirectAsyncValue`.

Note: as described in the [Implementing a kernel](#implementing-a-kernel)
section, `tfrt.async_add.i32` will add a ref before it calls `EnqueueWork` to
keep its output value alive. But in this example, the executor *can not execute*
the second `tfrt.async_add.i32` kernel because the kernel's input `%v2` values
are not available, so `tfrt.async_add.i32` does not get an opportunity to add a
ref.
