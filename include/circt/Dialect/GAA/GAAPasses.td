//===-- GAAPasses.td - GAA pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the GAA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAAPASSES_TD
#define CIRCT_DIALECT_GAA_GAAPASSES_TD

include "mlir/Pass/PassBase.td"

def Analysis : Pass<"gaa-analysis", "gaa::CircuitOp"> {
  let summary = "analysis GAA Module, analyse method calling stacks";
  let description = [{
    This pass analyse the call stack of the GAA Circuit.
  }];
  let constructor = "circt::gaa::createAnalysisPass()";
}

#endif // CIRCT_DIALECT_GAA_GAAPASSES_TD
