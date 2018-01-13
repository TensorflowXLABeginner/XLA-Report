# XLA Summary

## Project link

Detailed Reports are included in our [XLA-Report repository](https://github.com/TensorflowXLABeginner/XLA-Report).

## Group Members

| Name | Github Account | ID         |
| ---- | -------------- | ---------- |
| 宋小牛  | Jeffery-Song   | PB15000301 |
| 王若冰  | MalcolmUnWang  | PB15121735 |
| 陈翊辉  | cyh-ustc       | PB15111656 |

## Introduction

**XLA** (Accelerated Linear Algebra)([Github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler)) is a domain-specific compiler for linear algebra that optimizes TensorFlow computations. The results are improvements in speed, memory usage, and portability on server and mobile platforms. Users can use XLA via JIT (just-in-time) compilation or AOT (ahead-of-time) compilation.

In this project, we will do a small research on XLA, which covers in

- What kind of acceleration can XLA do
- How does XLA do it
- Where does JIT and AOT came in

## Summary of Our Work

### 12.23 Commit

- [x] Discussed what will do and assign the research tasks.
- [x] Get start with TensorFlow and learning the background knowledge. We have wrote a report for [TensorFlow Programming](https://github.com/TensorflowXLABeginner/XLA-Report/blob/master/FirstCommitReports/Accelerated%20Linear%20Algebra%20Intro.md).
- [x] Installed TensorFlow with GPU support and start toying with the basic operation.
- [x] Learned the general concepts and implementation about XLA. We have wrote two report individually for [XLA Overview](https://github.com/TensorflowXLABeginner/XLA-Report/blob/master/FirstCommitReports/Accelerated%20Linear%20Algebra%20Intro.md) and [XLA Just-in-time Compilation](https://github.com/TensorflowXLABeginner/XLA-Report/blob/master/FirstCommitReports/xla_Just-in-time%20compilation.md).

### 1.3 Commit

- [x] Compiled TensorFlow from source since XLA can only be included when TensorFlow is compiled from source.
- [x] Turned on JIT compilation to include XLA and run some optimized example.
- [x] Did some basic profiling with CUDA Profiling Tools Interface (CUPTI).
- [x] Report the test and profiling we have done in the [repository](https://github.com/TensorflowXLABeginner/XLA-Report/tree/master/SecondCommitReports).

### 1.13 Commit

- [x] Build our XLA on multiple platform. Including CPU version on PC, GPU version by **CUDA 8.0** on PC (**Nvidia GTX 970M**), GPU version by CUDA 9.0 on high performance cluster (**Nvidia Tesla V100**), GPU version AOT compilation for x86-64 and **ARM**.


- [x] We have done full benchmark and profiling for JIT compilation on V100 cluster. Compiler optimization