//===- Scheduling.cpp - Scheduling Pass ----------------------------C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Analysis pass.
// Rather than construction of
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/GAA/GAAOps.h"
#include "circt/Dialect/GAA/GAAPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace circt;
using namespace gaa;
using namespace mlir;


namespace {
class AnalysisPass : public AnalysisBase<AnalysisPass> {
  void runOnOperation() override;
};
} // end anonymous namespace


void AnalysisPass::runOnOperation() {
  auto circuit = mlir::OperationPass<circt::gaa::CircuitOp>::getOperation();
  auto circuitName = circuit->getAttr("sym_name");
  // There are only two kinds of modules: gaa::Module and gaa::ExtModule
  // construct graph:
  //   1. Instance Tree: from the top module(the module which has the same name as gaa::CircuitOp) to each instance.
  //                     for each instance, find to correspond gaa::Module operator, do the same thing.
  //                     until the module only contains the gaa::ExtModule.
  //   2. Method Graph: for each module inside the Instance Graph, get all rules and method from which. find the method graph.
  //                    the vertex is rule or method, the root are all rules.
  //
  for (auto op : circuit.getOps<circt::gaa::ModuleOp>()) {
    // Q: How to get operator from symbol?
    // gaa.instance -> graph
    // method call graph in

    llvm::outs() << "Found Top: " << op->getAttr("sym_name") << "\n";
  }
  return ;
}


std::unique_ptr<mlir::Pass> circt::gaa::createAnalysisPass() {
  return std::make_unique<AnalysisPass>();
}
