//===- GAAOps.h - GAA Dialect Operators -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the GAA dialect operators.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_GAAOPS_H
#define CIRCT_DIALECT_GAA_GAAOPS_H

#include "llvm/ADT/Any.h"

#include "circt/Dialect/GAA/GAADialect.h"
#include "circt/Dialect/GAA/GAAOpInterfaces.h"

// provides implementations for FunctionInterface.td
#include "mlir/IR/FunctionInterfaces.h"
// LogicalResult
#include "mlir/IR/Diagnostics.h"
// provides implementations for OpAsmInterface.td
#include "mlir/IR/OpImplementation.h"
// provides implementations for SymbolInterfaces.td
#include "mlir/IR/SymbolTable.h"
// provides implementations for ControlFlowInterfaces.td
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/GAA/GAA.h.inc"

#endif // CIRCT_DIALECT_GAA_GAAOPS_H
