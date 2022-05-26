//===- GenerateConflictMatrix.cpp - GenerateConflictMatrix Pass ----C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Use InstanceGraph and CallGraph to illustrate the method call information,
//
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/GAA/GAAOps.h"
#include "circt/Dialect/GAA/GAAPasses.h"
#include "circt/Dialect/GAA/InstanceGraph.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace gaa;
using namespace mlir;

class GenerateConflictMatrix
    : public GenerateConflictMatrixBase<GenerateConflictMatrix> {
  void runOnOperation() override;

private:
  // record an instance list from top
  struct StackElement {
    StackElement(InstanceGraphNode *node)
        : node(node), iterator(node->begin()), viewed(false) {}
    InstanceGraphNode *node;
    InstanceGraphNode::iterator iterator;
    bool viewed;
  };
  llvm::SmallVector<std::pair<StackElement, StringAttr>> rules;

  // (module -> (rule/method/value -> (instance, method/value)))
  using CacheMap = DenseMap<llvm::StringRef, DenseMap<llvm::StringRef, llvm::SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>, 4>>>;
  CacheMap moduleMethods = CacheMap {};
  CacheMap moduleValues = CacheMap {};
  CacheMap moduleRules = CacheMap {};
};


void GenerateConflictMatrix::runOnOperation() {
  auto circuit = OperationPass<CircuitOp>::getOperation();

  // firstly we gather all function call inside the rule/method/value for each module.
  circuit.walk([&](circt::gaa::ModuleOp module){
    // first walk the InstanceOp and store the corresponding FunctionLikeOp to each instance.
    // LHS is the SymbolRefAttr of the instance, RHS is the methods in the reference module of the instance.
    using MethodInInstance = std::pair<llvm::StringRef, llvm::SmallVector<llvm::StringRef, 4>>;

    /// a cache that stores all functions name which can be call on each instance.
    llvm::SmallVector<MethodInInstance> instanceAndMethod;

    auto instances = circt::gaa::getInstances(module);
    // walk the instance.
    for (auto instance : instances) {
      // get the reference module of an instance.
      auto instanceName = instance.instanceName();
      auto refModule = getReferenceModule(instance);
      // get all GAAFunctionLike Operator of the reference module.
      // ValueOp, MethodOp for GAAModule
      // BindValueOp, BindMethodOp for GAAExtModule
      auto instanceFunction = getFunctions(refModule);
      auto instanceFunctionNames = llvm::SmallVector<llvm::StringRef, 4>();
      llvm::for_each(instanceFunction, [&](GAAFunctionLike function){
        instanceFunctionNames.push_back(function->getName().getStringRef());
      });
      instanceAndMethod.push_back(std::pair(instanceName, instanceFunctionNames));
    }

    // walking Method/Value/RuleOp to show which function has been called.
    module.walk([&](Operation *op){
      mlir::TypeSwitch<Operation*>(op)
          // For each MethodOp and ValueOp in the module, check the
          .Case<MethodOp>([&](GAAFunctionLike function){
            function.walk([&](GAACallLike call){
              auto instanceToCall = call.instanceName();
              auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
              auto functionToCall = call.functionName();
              llvm::outs() << module.moduleName() << "." << function.functionName() << " call " << instanceToCall << "/" << instanceToCallModule << ":" << functionToCall
                           << "\n";
              moduleMethods[module.moduleName()][function.functionName()].push_back(std::pair(instanceToCall, functionToCall));
            });
          })
          .Case<ValueOp>([&](GAAFunctionLike function){
            function.walk([&](GAACallLike call){
              auto instanceToCall = call.instanceName();
              auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
              auto functionToCall = call.functionName();
              llvm::outs() << module.moduleName() << "." << function.functionName() << " call " << instanceToCall << "/" << instanceToCallModule << ":" << functionToCall
                           << "\n";
              moduleValues[module.moduleName()][function.functionName()].push_back(std::pair(instanceToCall, functionToCall));
            });
          })
          .Case<RuleOp>([&](GAARuleLike rule){
            rule.walk([&](GAACallLike call){
              auto instanceToCall = call.instanceName();
              auto instanceToCallModule = module.lookupSymbol<InstanceOp>(instanceToCall).moduleName();
              auto functionToCall = call.functionName();
              llvm::outs() << module.moduleName() << "." << rule.ruleName() << " call " << instanceToCall << "/" << instanceToCallModule << ":" << functionToCall
                           << "\n";
              moduleRules[module.moduleName()][rule.ruleName()].push_back(std::pair(instanceToCall, functionToCall));
            });
          });
    });
    auto moduleSymbolRef = mlir::SymbolRefAttr::get(module.getContext(), module.moduleName());
  });

  // then visiting all module to inspect the call map for each module.
  // get the instance graph of the circuit.
  auto *instanceGraph = &getAnalysis<InstanceGraph>();
  InstanceGraphNode *top = instanceGraph->getTopLevelNode();

  // DFS the circuit from the top to analyse the primitive call of each rule.
  // for the methods in the top module, they are regarded as rule, the interface
  // is out-ready->gaa gaa->enable->out.
  SmallVector<StackElement> instancePath;
  instancePath.emplace_back(top);
  while (!instancePath.empty()) {
    auto &element = instancePath.back();
    auto &node = element.node;
    auto &iterator = element.iterator;
    if(!element.viewed) {
      auto module = node->getModule();
      // regard methods in the top being rule
      if(node == top) {
        rules;
      }
      auto moduleSymbolRef = mlir::SymbolRefAttr::get(module.getContext(), module.moduleName());
    }
    element.viewed = true;

    if(iterator == node->end()) {
      instancePath.pop_back();
      continue;
    }
    auto *instanceNode = *iterator++;

    instancePath.emplace_back(instanceNode->getTarget());
  }

  exit(0);
  return;
}

std::unique_ptr<mlir::Pass> circt::gaa::createGenerateConflictMatrix() {
  return std::make_unique<GenerateConflictMatrix>();
}
