//===- GAAOps.h - Definition of GAA dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAAOPS_H
#define CIRCT_DIALECT_GAA_GAAOPS_H

#include "circt/Dialect/GAA/GAADialect.h"
#include "circt/Dialect/GAA/GAATypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/GAA/GAA.h.inc"

#endif // CIRCT_DIALECT_GAA_GAAOPS_H
