# Accelerated Linear Algebra

The Accelerated Linear Algebra (XLA) is a domain specific compiler developed by TensorFlow for optimizing its computations. The XLA framework is experimental and in active development. In particular, while it is unlikely that the semantics of existing operations will change, it is expected that more operations will be added to cover important use cases. The team welcomes feedback from the community about missing functionality and community contributions via GitHub.

## Existence and Feature of XLA

The following the key strengths for the feature of XLA:

- **Server side speedups**: Through this JIT compilation and specialization that we have mentioned, we see that some in-house models that TensorFlow team has win up to 60%. Also, SyntaxNet gets latency reductions from around 200 microseconds to 5 microseconds. And the reason for this was SyntaxNet had a lot of small operations in its graph so the interpreter has to go grab each little operation and this process of going and grabbing these small operations and running those causes some latency overhead but by compiling you're actually able to eliminate all that latency overhead away. 
- **Improve memory usage**: By eliminating many intermediate storage buffers, TensorFlow managed to improve the memory usage and hence you can target more architecture like mobile architecture that has limited capabilities. 
- Mobile footprint reductions: With XLA's ahead-of-time compilation you can go through a build process to turn models into executables if you want to do that from the beginning. This executable you can run it on the command line and it's able to eliminate much of TensorFlow runtime and by this; you will get to reduce the binary size of your program. So TensorFlow team has tried this feature on an long short-term memory (LSTM) model for mobile and they were able to reduce the binary size from 2.6MiB to less than 600KiB which means 4 times reduction in the deployment footprint and that is because of using XLA and following some best practices for writing TensorFlow code (https://www.tensorflow.org/performance/performance_guide). 
- Whole program analysis made easy: The general thing that is exciting about this XLA approach is that analyzing a whole graph or program is made easy by this compiler infrastructure so we have this thing that's call XLA's high-level optimizer that able to look at a linear algebra level graph and create a reusable toolkit of global optimizations to work on it. So even though you compile for different platforms CPU, GPU or other devices, TensorFlow has parameterized out the things that are specific to a given platform at this high-level optimization toolkit level. 

## Architecture of XLA

So XLA takes an input language called HLO IR or just High-Level Optimizer (HLO). So XLA takes a graph defined in this HLO and then compiles it into machine instructions for different architectures.

The following diagram shows the compilation process in XLA: 

![image](/image/XLA.png)

TensorFlow uses HLO in order to provide target-independent code, so in this step TensorFlow is only trying to optimize your program without any target constraints, then TensorFlow uses another HLO in order to emit target dependent optimized and analyzed code that will be finally fed to XLA backend for target-specific code generation. 

## Still Experimental

XLA is still under develop and the usage of it is still experimental. So there are not much documents and resources about it on the Internet. Choosing this topic as our project, we are ready to dig into the source code to discover the essence of XLA.



