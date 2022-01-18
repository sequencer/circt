//===- GAAOpInterfaces.td - GAA Dialect OpInterfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the GAA dialect operator interfaces.
//
//===----------------------------------------------------------------------===//
include "mlir/IR/OpBase.td"

def GAAModuleLike : OpInterface<"GAAModuleLike"> {
  let cppNamespace = "::circt::gaa";
  let description = "Provide base module interface.";
  let methods = [
    //===------------------------------------------------------------------===//
    // Module Name
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module name",
    "StringRef", "moduleName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,
  ];
  let verify = [{}];
}