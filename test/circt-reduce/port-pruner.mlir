// RUN: circt-reduce %s --test %S/test.sh --test-arg cat --test-arg "firrtl.module @Bar" --keep-best=0 --include port-pruner | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<3>) {
    // CHECK: %bar_a = firrtl.wire
    // CHECK: %bar_c = firrtl.wire
    // CHECK: %bar_e = firrtl.wire
    // CHECK: %bar_b, %bar_d = firrtl.instance bar @Bar
    %bar_a, %bar_b, %bar_c, %bar_d, %bar_e = firrtl.instance bar @Bar (in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>, out e: !firrtl.uint<1>)
    firrtl.connect %bar_a, %x : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %bar_b, %x : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = firrtl.add %bar_c, %bar_d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %1 = firrtl.add %0, %bar_e : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    firrtl.connect %y, %1 : !firrtl.uint<3>, !firrtl.uint<3>
  }

  // We're only ever using ports %b and %d -- the rest should be stripped.
  // CHECK-LABEL: firrtl.module @Bar
  // CHECK-NOT: in %a
  // CHECK-SAME: in %b
  // CHECK-NOT: out %c
  // CHECK-SAME: out %d
  // CHECK-NOT: out %e
  firrtl.module @Bar(
    in %a: !firrtl.uint<1>,
    in %b: !firrtl.uint<1>,
    out %c: !firrtl.uint<1>,
    out %d: !firrtl.uint<1>,
    out %e: !firrtl.uint<1>
  ) {
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    %0 = firrtl.not %b : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %c, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %d, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %e, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
