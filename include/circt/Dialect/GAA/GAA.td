//===- GAA.td - GAA dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Guard-Atomic-Action dialect.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_TD
#define CIRCT_DIALECT_GAA_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/RegionKindInterface.td"

def GAADialect : Dialect {
  let name = "gaa";

  let summary = "Operations for gaa dialect";
  let description = [{
    This dialect defines the `gaa` dialect, which is used to lower from
    GAA IR to HW.
    For more information, see the [GAA GitHub page](https://github.com/sequencer/gaa).
  }];

  let cppNamespace = "::circt::gaa";
}

// Base class for the types in GAA dialect.(currently only rule instance type.)
class GAAType<string name> : TypeDef<GAADialect, name> {}

// Base class for the operation in this dialect.
class GAAOp<
  string mnemonic, 
  list<OpTrait> traits = []
  > : Op<GAADialect, mnemonic, traits>;

include "GAATypes.td"
include "GAAOps.td"
#endif // CIRCT_DIALECT_GAA_TD