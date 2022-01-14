//===- GAATypes.td - Definition of GAA dialect types ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_GAA_GAATYPES_TD
#define CIRCT_DIALECT_GAA_GAATYPES_TD

def RuleInstanceType : GAAType<"ruleinst"> {
  let summary = "An GAA instance type";
  let description = [{
    Represents an Rule instance.
  }];
  let mnemonic = "instance";
  let parameters = (ins);
}

#endif // CIRCT_DIALECT_GAA_GAATYPES_TD
