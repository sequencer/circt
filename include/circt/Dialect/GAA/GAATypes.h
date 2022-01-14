//===- GAATypes.h - Definition of GAA dialect types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAATYPES_H
#define CIRCT_DIALECT_GAA_GAATYPES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/GAA/GAATypes.h.inc"

#endif // CIRCT_DIALECT_GAA_GAATYPES_H
