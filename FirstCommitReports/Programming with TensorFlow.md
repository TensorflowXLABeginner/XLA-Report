# Programming with TensorFlow

TensorFlow is an open source software library for numerical computation using data flow graphs that enables machine learning practitioners to do more data-intensive computing. It provides some robust implementations of widely used deep learning algorithms. Nodes in the flow graph represent mathematical operations. On the other hand, the edges represent multidimensional tensors that ensure communication between edges and nodes. TensorFlow offers you a very flexible architecture that enables you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.

 ## Feature

The main features offered by the latest release of TensorFlow are as follows: 

- **Faster computing**: The major versioning upgrade to TensorFlow has made its capability incredibly faster including a 7.3x speedup on 8 GPUs for inception v3 and 58x speedup for distributed Inception (v3 training on 64 GPUs).
- **Flexibility**: TensorFlow is not just a deep learning or machine learning software library but also great a library full with powerful mathematical functions with which you can solve most different problems. 
- **Portability**: TensorFlow runs on Windows, Linux, and Mac machines and on mobile computing platforms (that is, Android).
- **Unified API**: TensorFlow offers you a very flexible architecture that enables you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 
- **Transparent use of GPU computing**: Automating management and optimization of the same memory and the data used. You can now use your machine for largescale and data-intensive GPU computing with NVIDIA, cuDNN, and CUDA tool kits.

## Computing Model

### Tensors

The central unit of data in TensorFlow is the **tensor**. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's **rank** is its number of dimensions. Here are some examples of tensors:

```
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

### Computational Graph

When performing an operation, for example training a neural network, or the sum of two integers, TensorFlow internally represent, its computation using a *data flow graph* (or *computational graph*). 

The TensorFlow implementation defines *control dependencies* to enforce orderings between otherwise independent operations as a way of controlling peak memory usage. A computational graph is basically like a *flow chart*; the following is the computational graph for a simple computation: *z=d×c=(a+b) ×c*. 

![image](/image/graph.png)

## Programming Model

A TensorFlow program is generally divided into three phases: 

- Construction of the computational graph
- Running a session, which is performed for the operations defined in the graph
- Resulting data collection and analysis

These main steps define the *programming model* in TensorFlow.
Consider the following example, in which we want to multiply two numbers. 

```python
import tensorflow as tf
with tf.Session() as session:
	x = tf.placeholder(tf.float32,[1],name="x")
	y = tf.placeholder(tf.float32,[1],name="y")
	z = tf.constant(2.0)
	y = x * z
x_in = [100]
y_output = session.run(y,{x:x_in})
print(y_output)
```

For this example, the computation graph will be the following:

![image](/image/constructed.png)

## Data Model

The data model in TensorFlow is represented by tensors. Without using complex mathematical definitions, we can say that a tensor (in TensorFlow) identifies a *multidimensional numerical array*. This data structure is characterized by three parameters--*Rank*, *Shape*, and *Type*. 

### Rank

Each tensor is described by a unit of dimensionality called **rank**. It identifies the number of dimensions of the tensor, for this reason, a rank is a known-as order or n-dimensions of a tensor. A rank zero tensor is a **scalar**, a rank one tensor ID a vector, while a rank two tensor is a matrix. 

### Shape

The *shape* of a tensor is the number of rows and columns it has. Now we see how to relate the shape to a rank of a tensor: 

```python
>>scalar1.get_shape()
TensorShape([])

>>vector1.get_shape()
TensorShape([Dimension(5)])

>>matrix1.get_shape()
TensorShape([Dimension(2), Dimension(3)])

>>cube1.get_shape()
TensorShape([Dimension(3), Dimension(3), Dimension(1)])
```

### Data types

In addition to rank and shape, tensors have a *data type*. The following is a list of the data types: 

| Data type     | Python type   | Description                              |
| :------------ | :------------ | :--------------------------------------- |
| DT_FLOAT      | tf.float32    | 32-bit floating point.                   |
| DT_DOUBLE     | tf.float64    | 64-bit floating point.                   |
| DT_INT8       | tf.int8       | 8-bit signed integer.                    |
| DT_INT16      | tf.int16      | 16-bit signed integer.                   |
| DT_INT32      | tf.int32      | 32-bit signed integer.                   |
| DT_INT64      | tf.int64      | 64-bit signed integer.                   |
| DT_UINT8      | tf.uint8      | 8-bit unsigned integer.                  |
| DT_STRING     | tf.string     | Variable length byte arrays.             |
| DT_BOOL       | tf.bool       | Boolean.                                 |
| DT_COMPLEX64  | tf.complex64  | Complex number.                          |
| DT_COMPLEX128 | tf.complex128 | Complex number made up of two 64 bits floating points, |
| DT_QINT8      | tf.qint8      | 8-bit signed integer used in quantized Ops. |
| DT_QINT32     | tf.qint32     | 32-bit signed integer used in quantized ops. |
| DT_QUINT8     | tf.quint8     | 8-bit unsigned integer used in quantized ops. |

