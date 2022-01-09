// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-remove-unused-ports)' %s | FileCheck %s
firrtl.circuit "Top"   {
  // CHECK-LABEL: firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>)
  firrtl.module @Top(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c = firrtl.instance A  @UseBar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @Bar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>)
  firrtl.module @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    firrtl.connect %c, %b : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @UseBar(in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
  firrtl.module @UseBar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    %A_a, %A_b, %A_c, %A_d = firrtl.instance A  @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %c, %A_c : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @Foo(in %a: !firrtl.uint<1> sym @dntSym, in %b: !firrtl.uint<1> [{a = "a"}])
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) attributes {
    portAnnotations = [[], [{a = "a"}]], portSyms = ["dntSym", ""]}
  {}

  // CHECK-LABEL: firrtl.module @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>)
  firrtl.module @UseFoo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %A_a, %A_b = firrtl.instance A  @Foo(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>)
    firrtl.connect %A_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %A_b, %b : !firrtl.uint<1>, !firrtl.uint<1>
  }
}