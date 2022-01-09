//===- RemoveUnusedPorts.cpp - Remove Dead Ports ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-unused-ports"

using namespace circt;
using namespace firrtl;

namespace {
struct RemoveUnusedPortsPass
    : public RemoveUnusedPortsBase<RemoveUnusedPortsPass> {
  void runOnOperation() override;
  void removeUnusedModulePorts(FModuleOp module,
                               InstanceGraphNode *instanceGraphNode);
};
} // namespace

void RemoveUnusedPortsPass::runOnOperation() {
  auto instanceGraph = getAnalysis<InstanceGraph>();
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  CircuitOp circuit = getOperation();
  // Iterate in the reverse order of instance graph iterator, i.e. from bottom
  // to top.
  for (auto node : llvm::reverse(instanceGraph))
    if (auto module = dyn_cast<FModuleOp>(node->getModule()))
      // Don't prune the main module.
      if (circuit.getMainModule() != module)
        removeUnusedModulePorts(module, node);
}

void RemoveUnusedPortsPass::removeUnusedModulePorts(
    FModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  SmallVector<unsigned> removalPortsIndexes;
  auto ports = module.getPorts();

  for (auto e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);
    // If the port is not input or there exists a user, we cannot remove the
    // port.
    if (!port.isInput() || !arg.use_empty())
      continue;

    // If the port has a symbol or unprocessed annotations, we cannot remove the
    // port.
    if ((port.sym && !port.sym.getValue().empty()) || !port.annotations.empty())
      continue;

    removalPortsIndexes.push_back(index);
  }

  // If there is nothing to remove already, abort.
  if (removalPortsIndexes.empty())
    return;

  module.erasePorts(removalPortsIndexes);

  LLVM_DEBUG(llvm::for_each(removalPortsIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  for (auto user : instanceGraphNode->uses()) {
    auto instance = user->getInstance();
    OpBuilder builder(instance);
    for (auto index : removalPortsIndexes) {
      auto result = instance.getResult(index);
      assert(ports[index].isInput() && "port must be an input port");
      // Replace with an unwritten wire so that we can remove use-chains in SV
      // dialect canonicalization.
      WireOp wire = builder.create<WireOp>(instance.getLoc(), result.getType());
      result.replaceUsesWithIf(wire, [&](OpOperand &op) -> bool {
        // Connects can be deleted directly.
        if (isa<ConnectOp>(op.getOwner())) {
          op.getOwner()->erase();
          return false;
        }
        return true;
      });

      if (wire.use_empty())
        wire.erase();
    }

    instance.erasePorts(builder, removalPortsIndexes);
    instance.erase();
  }

  numRemovedPorts += removalPortsIndexes.size();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRemoveUnusedPortsPass() {
  return std::make_unique<RemoveUnusedPortsPass>();
}
