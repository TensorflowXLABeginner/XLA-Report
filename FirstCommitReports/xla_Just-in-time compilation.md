#Just-in-time compilation

## Just In Time -JIT introduction

### What does 'Just In Time -JIT' mean?

Just-in-time (JIT) is an inventory strategy companies employ to increase efficiency and decrease waste by receiving goods only as they are needed in the production process, thereby reducing inventory costs. This method requires producers to forecast demand accurately.

This inventory supply system represents a shift away from the older just-in-case strategy, in which producers carried large inventories in case higher demand had to be met.

### Advantages

Just-in-time inventory control has several advantages over traditional models. Production runs remain short, which means manufacturers can move from one type of product to another very easily. This method reduces costs by eliminating warehouse storage needs. Companies also spend less money on raw materials because they buy just enough to make the products and no more.

### Disadvantages

The disadvantages of just-in-time inventories involve disruptions in the supply chain. If a supplier of raw materials has a breakdown and cannot deliver the goods on time, one supplier can shut down the entire production process. A sudden order for goods that surpasses expectations may delay delivery of finished products to clients.

##Just-in-time compilation via XLA

The way TensorFlow is going to increase the speed of its programs and incorporate more
devices that can run TensorFlow is with this JIT compilation via XLA.

The way XLA is working is summarized in the following figure:

![XLA life cycle](http://img.ctolib.com/uploadImg/20170307/20170307064228_586.png)

TensorFlow team has developed this compiler infrastructure such that you can hand it a TensorFlow graph and then an optimized and specialized assembly comes out.
This is the compiler infrastructure that TensorFlow team have developed such that it can take a TensorFlow graph and spit out optimized and specialized assembly for that graph.
This is a great feature added by the TensorFlow team as it enables you to produce compiled code that's not architecture specific rather it will be optimized and specialized for the underlying architecture that you are using.

### an example

To show how XLA is working, let's demonstrate this by using the TensorFlow shell.
You need to open a TensorFlow shell in order to run the following code snippet. First, you can choose the paste mode by issuing the following command:

```
	%cpaste
```

Next, you can paste the following example:

```python
with tf.Session() as sess:
	x = tf.placeholder(tf.float32, [4])
	with tf.device("device:XLA_CPU:0"):
		y = x*x
	result = sess.run(y,{x:[1.5,0.5,-0.5,-1.5]})
```

You need to pass the following parameter to TensorFlow shell:

```
	--xla_dump_assembly=true
```

Here we are passing this flag to say spit out the XLA assembly that that's produced.
The output of the previous code will be the following:

```
0x00000000 movq (%rdx), %rax
0x00000003 vmovaps (%rax), %xmm0
0x00000007 vmulps %xmm0, %xmm0, %xmm0
0x0000000b vmovaps %xmm0, (%rdi)
0x0000000f retq
```

Now we are going to elaborate more on this example in order to understand the code snippet and the assembly output.
The previous example is just taking four floating point numbers and multiplying them.
What's special about this example is that we are putting it explicitly onto the XLA CPU device so the compiler is exposed as a device in one mode inside of TensorFlow.
After running the previous code snippet you will see just a couple of assembly instructions get emitted and the special thing about these instructions is that there are no loops because XLA knew that you are only going to multiply four numbers. So the emitted assembly instructions is specialized and optimized for the graph or the program that you fed in with
your TensorFlow expression.

Also, you put the previous code snippet explicitly onto the XLA GPU device, but we are not
going to cover this here. So as we mentioned that XLA can work for CPU and GPU in a
standard TensorFlow shell.