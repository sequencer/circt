//===- ModuleInliner.cpp - FIRRTL module inlining ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL module instance inlining.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::BitVector;

//===----------------------------------------------------------------------===//
// Module Inlining Support
//===----------------------------------------------------------------------===//

namespace {
/// A representation of an NLA that can be mutated.  This is intended to be used
/// in situations where you want to make a series of modifications to an NLA
/// while also being able to query information about it.  Finally, the NLA is
/// written back to the IR to replace the original NLA.
class MutableNLA {
  // Storage of the NLA this represents.
  NonLocalAnchor nla;

  // A namespace that can be used to generate new symbol names if needed.
  CircuitNamespace *circuitNamespace;

  /// A mapping of symbol to index in the NLA.
  DenseMap<Attribute, unsigned> symIdx;

  /// Records which elements of the path are inlined.
  BitVector inlinedSymbols;

  /// The point after which the NLA is flattened.
  signed flattenPoint = -1;

  /// Indicates if this NLA is dead and should be deleted.
  bool dead = false;

  /// Stores any new roots for the NLA.  This occurs when an NLA's root is
  /// inlined and one new NLA is created for each instance of the inlined root.
  SmallVector<std::pair<StringAttr, StringAttr>> newTops;

  /// Stores the size of the NLA path.
  unsigned int size;

public:
  MutableNLA(NonLocalAnchor nla, CircuitNamespace *circuitNamespace)
      : nla(nla), circuitNamespace(circuitNamespace),
        inlinedSymbols(BitVector(nla.namepath().size(), true)),
        size(nla.namepath().size()) {
    llvm::errs() << "Constructing Mutable NLA for: " << nla << "\n"
                 << "  symIdx:\n";
    for (size_t i = 0, e = size; i != e; ++i) {
      symIdx.insert({modPart(i), i});
      llvm::errs() << "    - " << modPart(i) << " -> " << i << "\n";
    }
  }

  /// This default constructor only exists so that the `[]` referential lookup
  /// can be used for `Inliner::nlaMap`.  The `nlaMap` is prepopulated with all
  /// NLAs, so this will always hit.
  MutableNLA() {
    llvm_unreachable(
        "the default constructor for MutableNLA should never be used");
  }

  /// Return the original NLA that this was pointing at.
  NonLocalAnchor getNLA() { return nla; }

  /// Writeback updates accumulated in this MutableNLA to the IR.  This method
  /// should only ever be called once and, if a writeback occurrs, the
  /// MutableNLA is NOT updated for further use.  Interacting with the
  /// MutableNLA in any way after calling this method may result in crashes.
  /// (This is done to save unnecessary state cleanup of a pass-private
  /// utility.)
  NonLocalAnchor applyUpdates() {
    // Delete an NLA which is either dead or has been made local.
    if (isLocal() || isDead()) {
      nla.erase();
      return nullptr;
    }

    // The NLA was never updated, just return the NLA and do not writeback
    // anything.
    if (inlinedSymbols.all() && newTops.empty() && flattenPoint == -1)
      return nla;

    // The NLA has updates.  Generate a new NLA with the same symbol and delete
    // the original NLA.
    OpBuilder b(nla);
    auto writeBack = [&](StringAttr root, StringAttr sym) -> NonLocalAnchor {
      SmallVector<Attribute> namepath;
      StringAttr lastMod;
      // This unrolls the first iteration of the loop because it has weird
      // behavior around using the root name.
      if (!inlinedSymbols.test(1)) {
        if (!lastMod)
          lastMod = root;
      } else {
        namepath.push_back(hw::InnerRefAttr::get(
            nla.getContext(), lastMod ? lastMod : root, refPart(0)));
      }
      // Rest of the loop.
      for (signed i = 1, e = inlinedSymbols.size() - 1; i != e; ++i) {
        if (i == flattenPoint) {
          lastMod = modPart(i);
          break;
        }

        if (!inlinedSymbols.test(i + 1)) {
          if (!lastMod)
            lastMod = modPart(i);
          continue;
        }

        namepath.push_back(hw::InnerRefAttr::get(
            nla.getContext(), lastMod ? lastMod : modPart(i), refPart(i)));
        lastMod = {};
      }

      if (flattenPoint == -1)
        namepath.push_back(nla.namepath()[size - 1]);
      else
        namepath.push_back(hw::InnerRefAttr::get(nla.getContext(), lastMod,
                                                 refPart(size - 1)));

      return b.create<NonLocalAnchor>(b.getUnknownLoc(), sym,
                                      b.getArrayAttr(namepath));
    };

    NonLocalAnchor last;
    assert(!dead || !newTops.empty());
    if (!dead)
      last = writeBack(root(), nla.getNameAttr());
    for (auto root : newTops)
      last = writeBack(root.first, root.second);

    nla.erase();
    return last;
  }

  /// Write this MutableNLA to a stream using FIRRTL target syntax.  The target
  /// is written with all updates applied.
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, MutableNLA &x) {
    auto writeOne = [&](StringRef root) {
      os << "~" << x.nla->getParentOfType<CircuitOp>().name() << "|";
      auto namepath = x.nla.namepath();
      os << root;

      SmallString<32> prefix;
      for (signed i = 1, e = x.inlinedSymbols.size(); i != e; ++i) {
        if (i - 1 == x.flattenPoint)
          break;

        StringAttr mod = x.modPart(i);
        StringAttr ref;
        if (auto innerRef = namepath[i - 1].dyn_cast<hw::InnerRefAttr>())
          ref = innerRef.getName();

        if (!prefix.empty() && ref)
          prefix.append("_");
        prefix.append(ref);
        if (!x.inlinedSymbols.test(i))
          continue;

        os << "/" << prefix << ":";
        os << mod.getValue();
        prefix.clear();
      }

      if (auto innerRef =
              namepath[namepath.size() - 1].dyn_cast<hw::InnerRefAttr>()) {
        if (!prefix.empty())
          prefix.append("_");
        prefix.append(innerRef.getName());
        os << ">" << prefix;
      }
    };

    SmallVector<Attribute> tops;
    if (!x.dead)
      tops.push_back(x.root());
    auto newTopNames = llvm::map_range(
        x.newTops, [](std::pair<Attribute, Attribute> a) { return a.first; });
    tops.append(newTopNames.begin(), newTopNames.end());

    bool multiary = !x.newTops.empty();
    if (multiary)
      os << "[";
    llvm::interleaveComma(tops, os,
                          [&](Attribute a) { writeOne(a.cast<StringAttr>()); });
    if (multiary)
      os << "]";

    return os;
  }

  /// Returns true if this NLA is dead.  There are several reasons why this
  /// could be dead:
  ///   1. This NLA has no uses and was not re-topped.
  ///   2. This NLA was flattened and its leaf reference is a Module.
  bool isDead() { return dead && newTops.empty(); }

  /// Returns true if this NLA is local.  For this to be local, every module
  /// after the root (up to the flatten point or the end) must be inlined.  The
  /// root is never truly inlined as inlined as inlining the root just sets a
  /// new root.
  bool isLocal() {
    unsigned end = flattenPoint > -1 ? flattenPoint + 1 : inlinedSymbols.size();
    return inlinedSymbols.find_first_in(1, end) == -1;
  }

  /// Return the module part at a specific index in the original NLA.
  StringAttr modPart(unsigned i) {
    return TypeSwitch<Attribute, StringAttr>(nla.namepath()[i])
        .Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
        .Case<hw::InnerRefAttr>([](auto a) { return a.getModule(); });
  }

  /// Return the module where the NLA starts.
  StringAttr root() {
    assert(!nla.namepath().empty());
    return modPart(0);
  }

  /// Return the reference part at a specific index in the original NLA.
  StringAttr refPart(unsigned i) {
    return TypeSwitch<Attribute, StringAttr>(nla.namepath()[i])
        .Case<FlatSymbolRefAttr>([](auto a) { return a.getAttr(); })
        .Case<hw::InnerRefAttr>([](auto a) { return a.getName(); });
  }

  /// Return the name of the reference pointed at by the NLA.
  StringRef ref() {
    assert(!nla.namepath().empty());
    return refPart(size - 1);
  }

  /// Mark a module as inlined.  This will remove it from the NLA.
  void inlineModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(sym != root() && "unable to inline the root module");
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym];
    llvm::errs() << "Inlining " << sym << ", index: " << idx << "\n"
                 << "  old state: ";
    for (unsigned long x : inlinedSymbols.getData())
      llvm::errs() << llvm::formatv("{0:x}", x);
    llvm::errs() << "\n  new state: ";
    inlinedSymbols.reset(idx);
    for (auto x : inlinedSymbols.getData())
      llvm::errs() << llvm::formatv("{0:x}", x);
    llvm::errs() << "\n";
    // If we inlined the last module in the path, then this NLA is dead.
    if (idx == size - 1)
      dead = true;
  }

  /// Mark a module as flattened.  This has the effect of inlining all of its
  /// children.  Also mark the NLA was dead if the leaf reference of this NLA is
  /// a module.
  void flattenModule(FModuleOp module) {
    auto sym = module.getNameAttr();
    assert(symIdx.count(sym) && "module is not in the symIdx map");
    auto idx = symIdx[sym] - 1;
    llvm::errs() << "Setting flatten point to: " << idx << "\n";
    flattenPoint = idx;
    // If the leaf reference is a module and we're flattening the NLA, then the
    // NLA must be dead.  Mark it as such.
    if (nla.namepath()[size - 1].isa<FlatSymbolRefAttr>())
      dead = true;
  }

  StringAttr reTop(FModuleOp module) {
    StringAttr sym = nla.sym_nameAttr();
    if (!newTops.empty())
      sym = StringAttr::get(nla.getContext(),
                            circuitNamespace->newName(sym.getValue()));
    llvm::errs() << nla.sym_name() << " adds top {" << module.getName() << ", "
                 << sym.getValue() << "}\n";
    newTops.push_back({module.getNameAttr(), sym});
    dead = true;
    return sym;
  }

  ArrayRef<std::pair<StringAttr, StringAttr>> getAdditionalSymbols() {
    return llvm::makeArrayRef(newTops);
  }

  void markDead() { dead = true; }
};
} // namespace

/// If this operation or any child operation has a name, add the prefix to that
/// operation's name.
static void rename(StringRef prefix, Operation *op) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<CombMemOp, InstanceOp, MemOp, MemoryPortOp, NodeOp, RegOp,
            RegResetOp, SeqMemOp, WireOp>([&](auto op) {
        op.nameAttr(
            StringAttr::get(op.getContext(), (prefix + op.name()).str()));
      });
  // Recursively rename any child operations.
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        rename(prefix, &op);
}

/// Clone an operation, mapping used values and results with the mapper, and
/// apply the prefix to the name of the operation. This will clone to the
/// insert point of the builder.
static void cloneAndRename(StringRef prefix, OpBuilder &b,
                           BlockAndValueMapping &mapper, Operation &op) {
  auto *newOp = b.clone(op, mapper);
  rename(prefix, newOp);
}

/// This function is used before inlining a module, to handle the conversion
/// between module ports and instance results. For every port in the target
/// module, create a wire, and assign a mapping from each module port to the
/// wire. When the body of the module is cloned, the value of the wire will be
/// used instead of the module's ports.
static SmallVector<Value> mapPortsToWires(StringRef prefix, OpBuilder &b,
                                          BlockAndValueMapping &mapper,
                                          FModuleOp target) {
  SmallVector<Value> wires;
  auto portInfo = target.getPorts();
  for (unsigned i = 0, e = target.getNumPorts(); i < e; ++i) {
    auto arg = target.getArgument(i);
    // Get the type of the wire.
    auto type = arg.getType().cast<FIRRTLType>();
    auto wire = b.create<WireOp>(target.getLoc(), type,
                                 (prefix + portInfo[i].getName()).str());
    wires.push_back(wire);
    mapper.map(arg, wire.getResult());
  }
  return wires;
}

/// This function is used after inlining a module, to handle the conversion
/// between module ports and instance results. This maps each wire to the
/// result of the instance operation.  When future operations are cloned from
/// the current block, they will use the value of the wire instead of the
/// instance results.
static void mapResultsToWires(BlockAndValueMapping &mapper,
                              SmallVectorImpl<Value> &wires,
                              InstanceOp instance) {
  for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i) {
    auto result = instance.getResult(i);
    auto wire = wires[i];
    mapper.map(result, wire);
  }
}

/// Inlines, flattens, and removes dead modules in a circuit.
///
/// The inliner works in a top down fashion, starting from the top level module,
/// and inlines every possible instance. With this method of recursive top-down
/// inlining, each operation will be cloned directly to its final location.
///
/// The inliner uses a worklist to track which modules need to be processed.
/// When an instance op is not inlined, the referenced module is added to the
/// worklist. When the inliner is complete, it deletes every un-processed
/// module: either all instances of the module were inlined, or it was not
/// reachable from the top level module.
///
/// During the inlining process, every cloned operation with a name must be
/// prefixed with the instance's name. The top-down process means that we know
/// the entire desired prefix when we clone an operation, and can set the name
/// attribute once. This means that we will not create any intermediate name
/// attributes (which will be interned by the compiler), and helps keep down the
/// total memory usage.
namespace {
class Inliner {
public:
  /// Initialize the inliner to run on this circuit.
  Inliner(CircuitOp circuit);

  /// Run the inliner.
  void run();

private:
  /// Returns true if the operation is annotated to be flattened.
  bool shouldFlatten(Operation *op);

  /// Returns true if the operation is annotated to be inlined.
  bool shouldInline(Operation *op);

  /// Flattens a target module in to the insertion point of the builder,
  /// renaming all operations using the prefix.  This clones all operations from
  /// the target, and does not trigger inlining on the target itself.
  void flattenInto(StringRef prefix, OpBuilder &b, BlockAndValueMapping &mapper,
                   FModuleOp target);

  /// Inlines a target module in to the location of the build, prefixing all
  /// operations with prefix.  This clones all operations from the target, and
  /// does not trigger inlining on the target itself.
  void inlineInto(StringRef prefix, OpBuilder &b, BlockAndValueMapping &mapper,
                  FModuleOp target);

  /// Recursively flatten all instances in a module.
  void flattenInstances(FModuleOp module);

  /// Inline any instances in the module which were marked for inlining.
  void inlineInstances(FModuleOp module);

  CircuitOp circuit;
  MLIRContext *context;

  // A symbol table with references to each module in a circuit.
  SymbolTable symbolTable;

  /// The set of live modules.  Anything not recorded in this set will be
  /// removed by dead code elimination.
  DenseSet<Operation *> liveModules;

  /// Worklist of modules to process for inlining or flattening.
  SmallVector<FModuleOp, 16> worklist;
};
} // namespace

bool Inliner::shouldFlatten(Operation *op) {
  return AnnotationSet(op).hasAnnotation("firrtl.transforms.FlattenAnnotation");
}

bool Inliner::shouldInline(Operation *op) {
  return AnnotationSet(op).hasAnnotation("firrtl.passes.InlineAnnotation");
}

void Inliner::flattenInto(StringRef prefix, OpBuilder &b,
                          BlockAndValueMapping &mapper, FModuleOp target) {
  for (auto &op : *target.getBody()) {
    // If its not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, b, mapper, op);
      continue;
    }

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      cloneAndRename(prefix, b, mapper, op);
      continue;
    }

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target);
    mapResultsToWires(mapper, wires, instance);

    // Unconditionally flatten all instance operations.
    flattenInto(nestedPrefix, b, mapper, target);
  }
}

void Inliner::flattenInstances(FModuleOp module) {
  for (auto &op : llvm::make_early_inc_range(*module.getBody())) {
    // If its not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      continue;
    }

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    BlockAndValueMapping mapper;
    OpBuilder b(instance);
    auto nestedPrefix = (instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(wires[i]);

    // Recursively flatten the target module.
    flattenInto(nestedPrefix, b, mapper, target);

    // Erase the replaced instance.
    instance.erase();
  }
}

void Inliner::inlineInto(StringRef prefix, OpBuilder &b,
                         BlockAndValueMapping &mapper, FModuleOp target) {
  for (auto &op : *target.getBody()) {
    // If its not an instance op, clone it and continue.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance) {
      cloneAndRename(prefix, b, mapper, op);
      continue;
    }

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      cloneAndRename(prefix, b, mapper, op);
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      cloneAndRename(prefix, b, mapper, op);
      continue;
    }

    // Create the wire mapping for results + ports.
    auto nestedPrefix = (prefix + instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target);
    mapResultsToWires(mapper, wires, instance);

    // Inline the module, it can be marked as flatten and inline.
    if (shouldFlatten(target)) {
      flattenInto(nestedPrefix, b, mapper, target);
    } else {
      inlineInto(nestedPrefix, b, mapper, target);
    }
  }
}

void Inliner::inlineInstances(FModuleOp module) {
  for (auto &op : llvm::make_early_inc_range(*module.getBody())) {
    // If its not an instance op, skip it.
    auto instance = dyn_cast<InstanceOp>(op);
    if (!instance)
      continue;

    // If its not a regular module we can't inline it. Mark is as live.
    auto *module = symbolTable.lookup(instance.moduleName());
    auto target = dyn_cast<FModuleOp>(module);
    if (!target) {
      liveModules.insert(module);
      continue;
    }

    // If we aren't inlining the target, add it to the work list.
    if (!shouldInline(target)) {
      if (liveModules.insert(target).second) {
        worklist.push_back(target);
      }
      continue;
    }

    // Create the wire mapping for results + ports. We RAUW the results instead
    // of mapping them.
    BlockAndValueMapping mapper;
    OpBuilder b(instance);
    auto nestedPrefix = (instance.name() + "_").str();
    auto wires = mapPortsToWires(nestedPrefix, b, mapper, target);
    for (unsigned i = 0, e = instance.getNumResults(); i < e; ++i)
      instance.getResult(i).replaceAllUsesWith(wires[i]);

    // Inline the module, it can be marked as flatten and inline.
    if (shouldFlatten(target)) {
      flattenInto(nestedPrefix, b, mapper, target);
    } else {
      inlineInto(nestedPrefix, b, mapper, target);
    }

    // Erase the replaced instance.
    instance.erase();
  }
}

Inliner::Inliner(CircuitOp circuit)
    : circuit(circuit), context(circuit.getContext()), symbolTable(circuit) {}

void Inliner::run() {
  auto *topModule = circuit.getMainModule();
  // Mark the top module as live, so it doesn't get deleted.
  liveModules.insert(topModule);

  // If the top module is not a regular module, there is nothing to do.
  if (auto fmodule = dyn_cast<FModuleOp>(topModule))
    worklist.push_back(fmodule);

  // If the module is marked for flattening, flatten it. Otherwise, inline
  // every instance marked to be inlined.
  while (!worklist.empty()) {
    auto module = worklist.pop_back_val();
    if (shouldFlatten(module)) {
      flattenInstances(module);
      continue;
    }
    inlineInstances(module);

    // Delete the flatten annotations. Any module with the inline annotation
    // will be deleted, as there won't be any remaining instances of it.
    AnnotationSet(module).removeAnnotationsWithClass(
        "firrtl.transforms.FlattenAnnotation");
  }

  // Delete all unreferenced modules.
  for (auto &op : llvm::make_early_inc_range(*circuit.getBody())) {
    if (isa<FExtModuleOp, FModuleOp>(op) && !liveModules.count(&op))
      op.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public InlinerBase<InlinerPass> {
  void runOnOperation() override {
    Inliner inliner(getOperation());
    inliner.run();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
