//===- InstanceGraph.h - Instance graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GAA CallGraph: Child vertex is a method from an
// instance. Father vertex is rule or other methods.
// The CallGraph must be a DAG.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_CALLGRAPH_H
#define CIRCT_DIALECT_GAA_CALLGRAPH_H
namespace circt {
namespace gaa {
/// This is an edge in the CallGraph. This tracks a method or rule under

}
}
#endif // CIRCT_DIALECT_GAA_CALLGRAPH_H