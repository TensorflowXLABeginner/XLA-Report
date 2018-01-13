# XLA analyze

Basically, XLA involves in `tensorflow/compiler/tf2xla` and `tensorflow/compiler/xla` these two directories.

```
- <tensorflow source code>
  | tensorflow
    | compiler
      | tf2xla					Codes for turing Tensorflow operations into
      							HLO for XLA to use
      							
      	| kernels				each class matches a Tensorflow operations
      | xla						XLA source code for its optimizations
      	| service
```



Codes in `tf2xla/` are responsible for using Tensorflow operations to generate XLA codes. In `tf2xla/kernels`, there are source code such as `depthtospace_op.cc` and `retuction_ops.h` that match with specified Tensorflow operations.

Dir `xla/` is responsible for optimizing HLO input (target-independent & Terget-dependent optimizations & analyses), and generating target-specific codes. Inside `xla/service`, there are some basic optimizations such as `hlo_cse.h` for *common-subexpression elimination* and `hlo_dce.h` for *dead code elimination*.

## XLA Interpreter Backend

The XLA Interpreter backend operates at HLO-level by ingesting a HloModule and evaluating the result of the HLO graph directly with HloEvaluator, without lowering it further (to LLVM IR for example) before execution as other backends (CPU and GPU for example) do.

XLA Interpreter;s key componenets:

* [`InterpreterCompiler`] despite the inherited naming of "compiler", all `InterpreterCompiler` really does is the following:

  1. Runs certain HLO optimization passes on the given HLO graph.
  2. Generates an `InterpreterExecutable` from the optimized HLO graph.
  3. Registers itself in the global compiler factory registry.

* [`InterpreterExecutable`]: responsible for running input HLO graph through the `HloEvaluator`, allocating output buffer and finally copying evaluated Literal result over.

* [`HloEvaluator`]: traverses a HLO graph and evaluates each node in DFS ordering along the way.

By checking `xla/service/interpreter/compiler.h`, we find the class `InterpreterCompiler`.

```c++
class InterpreterCompiler : public Compiler {
  ...
 private:
  Status RunHloOptimization(HloModule* hlo_module);
  ...
};
```

Most HLO optimizations are done in this function.

```c++
Status InterpreterCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("Interpreter");
  
  /* perfom inline replace */
  // A pass which performs inlining.
  // Which can result, for example, in functions
  // that were previously being mapped by Map 
  // instead directly applied to the forwarded operands 
  // (i.e., map({X, Y}, max) -> max(X, Y)).
  pipeline.AddPass<Inliner>();
  
  /* unify sub computations */
  // Unify subcomputations of a `HloModule`: 
  // if any computations are equal, choose
  // one arbitrarily to use and delete the others.
  pipeline.AddPass<HloSubcomputationUnification>();
  
  /* common subexpression elimination */
  // A pass which performs common-subexpression 
  // elimination. Identical constants and identical 
  // instructions with the same operands are commoned. 
  // The pass iterates over the instructions in 
  // topological order which enables the pass to
  // find arbitrarily large common expressions.
  //
  // The bool parameter is named is_layout_sensitive.
  // If is_layout_sensitive is true, then the simplifier
  // preserves layout during transformation. Otherwise, 
  // layout is ignored.
  pipeline.AddPass<HloCSE>(false);

 
  // Do an HLO pass to a fix point.
  // AlgebraicSimplifier performs Algebraic Simplications
  // The second parameter is named valid_bit_cast_callback.
  // If is_layout_sensitive is true, then the simplifier 
  // preserves layout during transformation. 
  // Otherwise, layout is ignored. 
  // If valid_bitcast_callback returns true, then the pass 
  // will replace reshapes and transposes with bitcasts.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      false, [](const Shape&, const Shape&) { return false; });
  
  // HLO pass that makes the following transformations on while loops:
  //
  //  - A while loop with static trip count of 0 is deleted.
  //  - A while loops with static trip count of 1 is replaced 
  //    by its body (sans loop).
  //  - Elements of a while loop's tuple that the loop doesn't 
  //    use are removed from the tuple.
  //
  pipeline.AddPass<WhileLoopSimplifier>();
  
  // A pass which moves Reshapes and Transposes to let later
  // passes combine them. This now only moves them outputward 
  // across elementwise ops all whose operands are equivalent 
  // Reshapes or Transposes, but in future could potentially 
  // move them inputward also.
  pipeline.AddPass<ReshapeMover>();
  
  
  // A pass which performs constant folding in order to avoid 
  // unnecessary computation on constants.
  pipeline.AddPass<HloConstantFolding>();
  
  /* Another CSE, layout sensitive. */
  pipeline.AddPass<HloCSE>(true);
  
  // Abstract base class for layout constraints. These constraint 
  // objects are gathered together in LayoutConstraints object.
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());
  
  /* dead code elimination */
  pipeline.AddPass<HloDCE>();
  
  // Flatten the call graph for an HLO module into a tree.
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(hlo_module).status();
}
```

