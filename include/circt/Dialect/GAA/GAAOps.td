//===- GAA.td - GAA dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Guard-Atomic-Action dialect.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_GAATYPES_TD
#define CIRCT_DIALECT_GAA_GAATYPES_TD
// depends on HW Dialect to express the stateless function
// and it will forward depends on HWIntegerTypes.
include "circt/Dialect/HW/HWTypes.td"
// This will finally be lowering to firrtl.regreset since GAA should always 
// have explicit reset value. 
def EHROp : GAAOp<"ehr", [
    HasParent<"RuleDefOp">
  ]> {
  let summary = "Define a new EHR";
  let description = [{
    Declare a new ephemeral history register:
    ```
      %name = gaa.ehr %resetValue : !firrtl.uint
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    HWIntegerType:$resetValue 
  );
  let results = (outs HWIntegerType:$result);

  let builders = [
    OpBuilder<(ins 
      CArg<"StringRef">:$name,
      "::mlir::Type":$elementType,
      "::mlir::Value":$resetValue
    )>
  ];
}

def ReadOp : GAAOp<"read", [
    HasParent<"RuleDefOp">
  ]> {
  let summary = "Read from EHR";
  let description = [{
    Read with priority:
    if priority is 0, mezans unscheduled yet.
    ```
      %result = gaa.read[priority] %ehr : t1
    ```
  }];
  let arguments = (ins HWIntegerType:$ehr, I32Attr:$priority);
  let results = (outs HWIntegerType:$result);
}

def WriteOp : GAAOp<"write", [
    HasParent<"RuleDefOp">
  ]> {
  let summary = "Write to EHR";
  let description = [{
    Write with priority:
    if priority is 0, means unscheduled yet.
    ```
      gaa.write[priority] %ehr, %src : t1
    ```
  }];
  let arguments = (ins HWIntegerType:$ehr, HWIntegerType:$src, I32Attr:$priority);
}

// Same design as FModuleOp, we don't support Annotation at this time.
def RuleDefOp : GAAOp<
  "defrule", [
    // same as FModuleOp, RuleDefOp is defined under a CircuitOp
    HasParent<"mlir::ModuleOp">,
    // it has its own local region.
    IsolatedFromAbove,
    // it has and only has 1 block.
    SingleBlock,
    Symbol, 
    SymbolTable,
    NoTerminator, 
    NoRegionArguments
  ]> {
  let summary = "Rule Definition";
  let description = [{
    The "gaa.defrule" operation represents a guarded-atomic-action rule 
    definition. It essentially works as what "firrtl.when" does, but
    it's non-deterministic execution.
  }];
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    // Rather than FIRRTL FModule, rule definition is quite loose
    // it will dynamiclly infer state read/write via body. 
    OpBuilder<(ins
      "StringAttr":$name
    )>
  ];
  let extraClassDeclaration = [{
    Block *getBody() { return &body().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBody()->begin(); }
    iterator end() { return getBody()->end(); }

    Block::BlockArgListType getArguments() {
      return getBody()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }
  }];
}

def RuleInstOp : GAAOp<
  "instrule", [
    // TODO: not so sure is this correct or not. Rule can be instantiate at Rule or Module
    HasParent<"RuleDefOp">,
    // expose `LogicalResult RuleInstOp::verifySymbolUses(SymbolTableCollection &symbolTable)` function
    // this is used to find correspond rule, check arguments created.
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Instantiate an rule";
  let description = [{
    This represents an instance of a rule.
    If a rule is instantiated in a module, instantiate that rule directly.
    If a rule is in another a rule, copy `body` to current rule, `AND` guard together.

    Examples:
    ```mlir
    %someInstance = gaa.instrule instantName @RuleName(state0: !firrtl.uint, state1: !firrtl.uint)
    ```
  }];

  let arguments = (ins 
    FlatSymbolRefAttr:$moduleName, 
    StrAttr:$name
    // TODO<High>: parameter (state input/output) arguments.(should be list of HWIntegerType)
    // TODO<Low>: if gaa is instantiated in a Module, clock and reset should be assigned to this instance as well.
    //            since it is not possible to maintain clock-domain crossing logic via GAA.
    // TODO<Low>: in the future, if instantiating FModuleList is allowed, RuleDef/RuleInst should also take normal
    //            port as parameter as well.
    );
  let builders = [
    OpBuilder<(ins
      "RuleInstOp":$module,
      "mlir::StringRef":$name
    )>
  ];
  let extraClassDeclaration = [{
    /// Lookup the rule for the symbol. This returns null on invalid IR.
    FModuleLike getReferencedRule();
    FModuleLike getReferencedRule(SymbolTable& symtbl);
  }];
}

#endif // CIRCT_DIALECT_GAA_GAATYPES_TD