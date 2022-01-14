# GAA Dialect Rationale

This document describes various design points of the GAA dialect, why they are
the way they are, and current status.  This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

[Guarded Atomic Action](https://ieeexplore.ieee.org/document/1560170/) provides
a state transition based RTL design paradigm. Like any other RTL, it is defined
by modules, and instants modules to construct the hierarchy of circuit.

It uses primitive operator like add/sub/mul/div from HW Dialect to express the
combinational behaviors. However, for sequential logics, It defines a special
register: [EHR]([Ephemeral History Register](https://ieeexplore.ieee.org/document/1459853))
as sequential logic primitives to be operated inside the _Rule_ block,
which has two component _body_ and _guard_ to express the _atomic_ transition,
formally defined below:

`S` is all possible states that system can reach, `S0` is the initial state of
the system. EHR can be `read` to a HW data type, and do some operations via HW
Dialect, any `Value` can be marked as `guard`, any `Value` can be written to 
EHR. `write` manipulates the `states`, this `manipulation` should only happen
under the case that `guard` being true.

`rules` is a list of `rule` that will be executed in sequence, for each rule,
it must guarantee the state that this rule observes satisfies the `guard`
predicate.

0. _body_ is a stateless function expressed by `Comb` Dialect.
0. _guard_ is a boolean predication indicates if this rule *may* be executed or
   not.
0. For each cycle, each rule can only be executed or not executed. Intermediate
   state never exist.
0. Each possible states can recurrent by executing a list of rules in sequence.

Comparing to Chisel/FIRRTL based designs. Rather than constructing hardware
directly, GAA uses the rule-based design methodology to express the atomic
logic in transition systems. It will essentially eliminate the multi-write bug
by nondeterministic execution model. It will also speed up simulation, based on
another pass to lower the GAA Dialect IR to llvm IR.
Comparing to HLS, GAA provides a fine granularity PPA control to circuit,
without introducing too much pragma to compiler.

In the future, GAA Dialect can also become a intermediate IR supporting HLS,
since `body` of rule only need to be a stateless function, if any other IRs can
be lowered to `Comb` Dialect, then only need to declare the guard(condition) of
this rule, it will can be automatically scheduled to high parallelized and
correct RTL.

### Scheduling
The most important part of GAA Dialect is scheduling pass, which maintains the
atomic semantic for each rules by constructing a `Scheduler` circuit.

#### Rule Relationship
This section defines three relationship for rules, assuming circuit state is
`S`, `guard` of rule `r0` under state `S0` is `r0.guard(S0)`, `fire` of rule
under state `S0` is `r0.fire(S0)`, after `r0` transition, new state is
`r0.trans(S0)`.  

1. Conflict Free
`r0` and `r1` is _Conflict Free_ iff:
```
CF(r0, r1) =
  foreach S, r0.guard(S) && r1.guard(S) =>                   // For guard of rules being true at the same time:   
    r1.guard(r0.trans(S))                                 && //   After r0 transition, r1 guard still fit.
    r0.guard(r1.trans(S))                                 && //   After r1 transition, r0 guard still fit.
    r1.trans(r0.trans(S)) == r0.trans(r1.trans(S))           //   Order of r0 and r1 transition doesn't affect result.
      == CFT(S)                                              //   And generated circuit should match final state.
```

1. Mutually Exclusive
`r0` and `r1` are _Mutually Exclusive_ iff:
```
ME(r0, r1) =
  foreach S, r0.guard(S) && r1.guard(S) => // For guard of rules being true at the same time:
    !(r1.fire(S) && r1.fire(S))            //   They are never being able to fire at the same time.
```
Here are two types of ME:
- Implicitly ME: If two guards cannot be true at the same time, they are Implicitly ME,
                 this can be checked via SAT tools, or adding user assertions.
- Explicitly ME: If two guards maybe true at the same time, but for area and logic reuse
                 concerns, user explicitly mark two rules are mutually exclusive,
                 they are Explicitly ME.

1. Sequential Before
`r0` is _Sequential Before_ to `r1` iff:
```
SB(r0, r1) =
  foreach S, r0.guard(S) && r1.guard(S) =>     // For guard of rules being true at the same time:
    r1.guard(r0.trans(S))                   && //   After r0 transition, r1 guard still fit.
    r1.trans(r0.trans(S)) == SBT(S)            //   r1 sequential after r0
                                               //   And generated circuit should match the final state.
```
for the cases that `r0.guard(S) && r1.guard(S)`, hardware scheduler must
guarantee `r0` scheduled before `r1`, which means when `guard` of `r0` `r1` are
both true  read from `r1` can only observe write from `r0`, in hardware designs
this means the data goes to `r1` via `EHR`.

#### GAA Scheduling Compiler
In order to increase the execution parallelism of rules(schedule as much as
possible rules in one cycle), rules are statically scheduled based on the below
constraints: 

Unlike [Hardware synthesis from guarded atomic actions with performance specifications](https://ieeexplore.ieee.org/document/1560170/),
CIRCT GAA decides not to support FIFO and memory, here is the reason why:
1. FIFO as primitive adds too much redundancy to GAA design, different fifo
   implementations has different trade-off, but doesn't affect the essential of
   GAA, in the future design FIFO should be implemented via decoupled blackbox.
2. Supporting memory needs to export memory port information to scheduler,
   which increases complexity of scheduler designing. Asynchronous memory is 
   not widely used in forwarding design, it can essentially be replaced by 
   decoupled blackbox.

a rule specific _read_, _write_ operator is defined to support state forward
operation under GAA semantic: `rule0` is possible to read a state that being
written by an `rule1`, if and only if this writing won't break the `guard` of
`rule0`.

Based on the read-write graph to `EHR`, an analysis pass will solve the CF
(conflict free) and SC(sequential composable) relationship between different
rules, detecting which rule can be scheduled in the same cycle.

a SAT-based solving pass will be used for check the ME(mutually exclusive)
relationship among CF rules to detect rule result that won't be executed at the
same cycle to find the possible data path reuseable among different rules.

After rule scheduled, hardware constructing pass will construct the RTL based
on previous metadata to construct a scheduler and each data paths.

Here are detail scheduling algorithm:(TBD)
1. `r0` will block `r1` if `guard` in `r1` is written by `r0`.

1. RAW: `r0` write `s`, `r1` read `s`(`r0` and `r1` only shares same state `s`)
- `r0` CF `f1` will fail in compiler:
  reason TBD
- `r0` SB `r1` leads to `write[1]` in `r0`, while `read[0]` in `r1`:
    1       1  -> `s` being write,     `r1` read register value.
    0       1  -> `s` being write,     `r1` doesn't execute.
    1       0  -> `s` not being write, `r1` read register value.
- `r1` SB `r0` leads to `write[0]` in `r0`, while `read[1]` in `r1`:
    1       1  -> `s` being write,     `r1` read write value.
    0       1  -> `s` being write,     `r1` doesn't execute.
    1       0  -> `s` not being write, `r1` read register value.
1. WAW: `r0` write `s`, `r1` write `s`(`r0` and `r1` only shares same state `s`)
- `r0` CF `f1` will fail in compiler:
  reason TBD
    1       1  -> `s` being write
    0       1  -> `s` being write
    1       0  -> `s` not being write

### Hardware Implementation of Scheduler
Essentially the scheduler is the hardware takes `guard` as input, generating
`fire` as output. A scheduler is a decoding logic, which essentially is a truth
table(generated based on the rule constraint) interpreted PLA(Programable Logic
Array), terms with more `1` in the output leads to higher parallelism but
longer latency of decoding logic and larger PLA area:
```mlir
%fire_0, %fire_1, ... , %fire_n = decode(%guard_0, %guard_1, ... %guard_n)
```
the hardware generator needs to make tradeoffs among different design choices:
tradeoff between parallelism(how many rules can be fired at the same cycle) and
clock latency of scheduler(combinational logic that used by scheduling logic).

## Operations
1. EHR IR
EHR is the primitive of GAA design paradigm. In the circuit design, it can be
regarded as register with forward ports.
`%someEHR = gaa.ehr !HW.types`

2. EHR Read/Write IR
rather than providing another version of `firrtl.connect` or
`firrtl.partialconnect`, GAA doesn't support `connect` semantic at all.

_read_ and _write_ are provided with constant _priority_ for both _connect_
and _forward_ semantic:

EHR read is defined below:
`%readResult = gaa.read[priority] %someEHR : !HW.types`

EHR write is defined below:
`gaa.write[priority] %src, %destEHR`

`read[n]`, `write[n]` are primitive of state manipulation and accessing. `n`
represents priority, less `n` means higher priority in read and lower priority
in write. `n` can be given by user or calculated by scheduling pass:
The witness of `n` follow the order below, LSB is earlier, MSB is later.
```
read(1) -> write(1) -> read(2) -> write(2) -> ... -> read(n) -> write(n)
```
These rules can be inferred from the order list
1. `read(n)`(`n` > 1) will observe previous nearest `write(m)`(`n > m`).
2. `read(1)` will observe current EHR value.(no forward by default)
3. `write(n)` will override `write(m)`(`n > m`)
4. The latest(highest `n`) write will be written to EHR.

3. Rule IR
Rule definition is a stateless function like any other `Module`:
`gaa.defrule @RuleName (state0: !HW.types, state1: !HW.types ...) {body}`

A rule can be instantiated below, which will return the rule instance.
`%someInstance = gaa.instrule instantName @RuleName(%state0, %state1): !gaa.rule`.

If a rule is instantiated in a module from `HW` Dialect, the rule will be
instantiated directly.

If a rule is in another rule, `body` of this rule will be copied out, then
`Comb.and` `guard` together to become the new `guard`.

4. Scheduling IR
Rather than explicitly providing priority `n` in to each `read`, `write`, user
can also constraint rule priority with Rule Scheduling IR:

`gaa.setcf %r0, %r1` indicates two rules should be conflict free, constrainting
two rules can fire in the same cycle. If the analysis pass detects a conflict
condition, an exception should be thrown.

`gaa.setme %r0, %r1` indicates two rules should be mutually exclusive, which
predicts the guard of `%r0` and `%r1` should never be true at the same cycle,
making datapath possible to be shared between different rules. If the analysis
pass detects a case guard of `%r0` and `%r1` can both be true, a warning should
be raised to give the counter example to user.

`gaa.setsb %r0, %r1` indicates if both `guard` of `%r0` and `%r1` are true,
they should be scheduled to the same cycle, and `%r0` should be scheduled
before `%r1`. If the analysis pass detects a conflict free case, a warning
should be raised to indicate user that rules are conflict free.


## GCD Example
```mlir
gaa.module @GCD {
  %x = gaa.ehr 0 : !u32
  %y = gaa.ehr 0 : !u32
  %swapRule = gaa.instRule Swap (%r1, %r2)
  %subRule = gaa.instRule Sub (%r1, %r2)

  gaa.defrule @Swap (s1: !u32, %s2: !u32) {
    %r1 = gaa.read[0] %s1 : !u32
    %r2 = gaa.read[0] %s2 : !u32
    %0 = comb.neq %r2, 0 : !u1
    %1 = comb.gt %r1, %r2 : !u1
    %2 = comb.and %0 %1 : !u1
    gaa.guard %2
    gaa.write %1 %2
    gaa.write %2 %1
  }

  gaa.defrule @Sub (s1: !u32, %s2: !u32) {
    %r1 = gaa.read[0] %s1 : !u32
    %r2 = gaa.read[0] %s2 : !u32
    %0 = comb.leq %r1, %r2 : !u1
    %1 = comb.neq %r2, 0 : !u1
    %2 = comb.and %0, %1 : !u1
    %3 = comb.sub %s2, %s1 : !u32
    gaa.write[0] %s2 %3
  }

  // This rule will be invoked by other rules.
  gaa.defrule @start (s1: !u32, s2: !u32, in1: !u32, in2: !u32) {
    %r1 = gaa.read[0] %s1 : !u32
    %r2 = gaa.read[0] %s2 : !u32
    %0 = comb.eq 0 : !u1
    gaa.guard(%0)
    gaa.write[0] %in1 %r1
    gaa.write[0] %in2 %r2
  }

  // This rule will be invoked by other rules.
  gaa.defrule @result (s1: !u32, s2: !u32, target: !u32) {
    %r1 = gaa.read[0] %s1 : !u32
    %0 = comb.eq %r1, 0
    gaa.write %s1 %target
  }
}
```

## Internal designs
TBD
### Pass
1. Flatten GAA Modules.
Rather than creating ports automatically, firstly inline all GAAModules for method API.
2. EHR RW normalization
normalize priorities for EHR.
3. For each state, find its sources and sinks, create a mapping from state to rule.
Generate CF/SC Graphs and check it with `markSC` and `markCF`
4. Extract ME path.
Generate all ME paths. SAT them.
5. Construct Scheduler
6. Merge ME data path.
