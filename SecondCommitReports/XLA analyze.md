# XLA source code analyze

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

Dir `xla/` is responsible for optimizing HLO input (target-independent & Terget-dependent optimizations & analyses), and generating target-specific codes. Inside `xla/service`, there are some basic optimizations such as `hlo_cse.h` for *common-subexpression elimination* and `hlo_dce.h` for *dead code elimination*