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

// Test both serial and concurrent workqueues: all tests should be determinstic.

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: bef_executor -work_queue_type=mstd $(bef_name %s) | FileCheck %s --dump-input=fail

// Asynchronously increment %counter once.
func @async_incs(%counter : !test.atomic.i32, %ch : !tfrt.chain) -> !tfrt.chain {
  %ch_ret = tfrt_test.do.async %counter, %ch : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain) {
    %ch2 = "tfrt_test.atomic.inc.i32"(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> !tfrt.chain
    tfrt.return %ch2 : !tfrt.chain
  }
  tfrt.return %ch_ret : !tfrt.chain
}

// Increment %counter once in a nested async call.
func @nested_async_incs(%counter : !test.atomic.i32, %ch : !tfrt.chain) -> !tfrt.chain {
  %ch_ret = tfrt_test.do.async %counter, %ch : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain) {
    %ch1 = tfrt_test.do.async %counter, %ch : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain) {
      %ch2 = "tfrt_test.atomic.inc.i32"(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> !tfrt.chain
      tfrt.return %ch2 : !tfrt.chain
    }
    tfrt.return %ch1 : !tfrt.chain
  }
  tfrt.return %ch_ret : !tfrt.chain
}

// This test that all atomic increments scheduled to run asynchronously complete.
func @async_incs_complete() {
  %loop_count = tfrt.constant.i32 1024
  %counter = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %merged_chain = tfrt.new.chain

  %res:2 = tfrt.repeat.i32 %loop_count, %merged_chain, %counter : !tfrt.chain, !test.atomic.i32 {
    %ch = tfrt.new.chain
    %async_ret_ch = tfrt.call @async_incs(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain)
    %ret_merged_chain = tfrt.merge.chains %merged_chain, %async_ret_ch
    tfrt.return %ret_merged_chain, %counter : !tfrt.chain, !test.atomic.i32
  }

  // Use an input chain to ensure all pending work complete.
  %v:2 = "tfrt_test.atomic.get.i32"(%counter, %res#0)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 1024
  tfrt.print.i32 %v#0, %v#1

  tfrt.return
}

// This test that all atomic increments scheduled to run asynchronously complete
// after HostContext::Quiesce.
func @async_incs_complete_after_quiesce() {
  %loop_count = tfrt.constant.i32 1024
  %counter = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32

  %res = tfrt.repeat.i32 %loop_count, %counter : !test.atomic.i32 {
    %ch = tfrt.new.chain
    %async_ret_ch = tfrt.call @async_incs(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain)
    tfrt.return %counter : !test.atomic.i32
  }

  // Call HostContext::Quiesce to ensure all pending work complete.
  "tfrt_test.quiesce"() : () -> ()

  // Note that the input chain is a new chain.
  %new_ch = tfrt.new.chain
  %v:2 = "tfrt_test.atomic.get.i32"(%counter, %new_ch)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 1024
  tfrt.print.i32 %v#0, %v#1

  tfrt.return
}

// This test that all atomic increments scheduled to run asynchronously complete.
func @nested_async_incs_complete() {
  %loop_count = tfrt.constant.i32 1024
  %counter = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %merged_chain = tfrt.new.chain

  %res:2 = tfrt.repeat.i32 %loop_count, %merged_chain, %counter : !tfrt.chain, !test.atomic.i32 {
    %ch = tfrt.new.chain
    %async_ret_ch = tfrt.call @nested_async_incs(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain)
    %ret_merged_chain = tfrt.merge.chains %merged_chain, %async_ret_ch
    tfrt.return %ret_merged_chain, %counter : !tfrt.chain, !test.atomic.i32
  }

  // Use an input chain to ensure all pending work completes.
  %v:2 = "tfrt_test.atomic.get.i32"(%counter, %res#0)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 1024
  tfrt.print.i32 %v#0, %v#1

  tfrt.return
}

// This test that all atomic increments scheduled to run asynchronously complete
// after HostContext::Quiesce.
func @nested_async_incs_complete_after_quiesce() {
  %loop_count = tfrt.constant.i32 1024
  %counter = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32

  %res = tfrt.repeat.i32 %loop_count, %counter : !test.atomic.i32 {
    %ch = tfrt.new.chain
    %async_ret_ch = tfrt.call @nested_async_incs(%counter, %ch) : (!test.atomic.i32, !tfrt.chain) -> (!tfrt.chain)
    tfrt.return %counter : !test.atomic.i32
  }

  // Call HostContext::Quiesce to ensure all pending work completes.
  "tfrt_test.quiesce"() : () -> ()

  // Note that the input chain is a new chain.
  %new_ch = tfrt.new.chain
  %v:2 = "tfrt_test.atomic.get.i32"(%counter, %new_ch)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 1024
  tfrt.print.i32 %v#0, %v#1

  tfrt.return
}


// These test blocking work queue.
// CHECK-LABEL: --- Running 'simple_blocking_sleep'
func @simple_blocking_sleep() {
  %a = tfrt.constant.i32 1000
  "tfrt_test.usleep"(%a) : (i32) -> ()
  // CHECK: Slept for 1000 microseconds
  tfrt.return
}

// CHECK-LABEL: --- Running 'simple_blocking_sleep_with_chain'
func @simple_blocking_sleep_with_chain() -> !tfrt.chain {
  %a = tfrt.constant.i32 999
  // CHECK: Slept for 999 microseconds
  %ch = "tfrt_test.blocking.usleep"(%a) : (i32) -> (!tfrt.chain)
  tfrt.return %ch : !tfrt.chain
}

// This tests blocking work queue by adding multiple blocking tasks.
// CHECK-LABEL: --- Running 'repeated_blocking_sleep1'
func @repeated_blocking_sleep1() -> i32 {
  %count = tfrt.constant.i32 3
  %a0 = tfrt.constant.i32 100000
  %inc0 = tfrt.constant.i32 1

  // Repeatedly sleep for 3 times with the sleep time increased by 1 us every cycle.
  %a1, %inc1 = tfrt.repeat.i32 %count, %a0, %inc0 : i32, i32 {
    "tfrt_test.usleep"(%a0) : (i32) -> ()
    %a2 = "tfrt.add.i32"(%a0, %inc0) : (i32, i32) -> i32
    tfrt.return %a2, %inc0 : i32, i32
  }

  // The last sleep should be 100002 us.
  // CHECK: Slept for 100002 microseconds

  tfrt.return %a1 : i32
}

// This tests blocking work queue by adding more blocking tasks more than
// max_num_blocking_threads (4) configured for BlockingWorkQueue.
// CHECK-LABEL: --- Running 'repeated_blocking_sleep2'
func @repeated_blocking_sleep2() -> i32 {
  %count = tfrt.constant.i32 300
  %a0 = tfrt.constant.i32 1000
  %inc0 = tfrt.constant.i32 1

  // Repeatedly sleep for 300 times with the sleep time increased by 1 us every cycle.
  %a1, %inc1 = tfrt.repeat.i32 %count, %a0, %inc0 : i32, i32 {
    "tfrt_test.usleep"(%a0) : (i32) -> ()
    %a2 = "tfrt.add.i32"(%a0, %inc0) : (i32, i32) -> i32
    tfrt.return %a2, %inc0 : i32, i32
  }

  // The last sleep should be 1299 us.
  // CHECK: Slept for 1299 microseconds
  tfrt.return %a1 : i32
}
