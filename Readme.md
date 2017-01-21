# MNIST For ML Beginners

Implementation of the [tutorial](https://www.tensorflow.org/tutorials/mnist/beginners/) plus some code to evaluate single files at the end.

Type `make` on the command line to install dependencies and run the notebook.


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
import tensorflow as tf
import numpy as np
```


```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
```


```python
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```


```python
init = tf.global_variables_initializer()
```


```python
session = tf.Session()
session.run(init)
```


```python
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```


```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

    0.917


_(end of the tutorial)_
___________________

## Parse single files using the tensors above


```python
import operator

def judge_file(filename, comp=None):
    with open(filename, 'rb') as f:
        pixels = []
        while True:
            try:
                pixels.append(f.read(1)[0])
            except IndexError:
                break

    result = session.run(y, feed_dict={x: [np.asfarray([1 - (pixel / 255) for pixel in pixels])]})

    d, p = max(enumerate(result[0]), key=operator.itemgetter(1))
    if d == comp:
        print("{}: {}, {:.4}%".format(filename, d, p * 100))
    return max(enumerate(result[0]), key=operator.itemgetter(1))

```


```python
dict(enumerate([judge_file('single_files/test{}.data'.format(n), n) for n in range(10)]))
```

    single_files/test0.data: 0, 99.89%
    single_files/test2.data: 2, 99.85%
    single_files/test3.data: 3, 100.0%
    single_files/test4.data: 4, 97.75%
    single_files/test5.data: 5, 66.26%





    {0: (0, 0.99892777),
     1: (8, 0.86384517),
     2: (2, 0.99845934),
     3: (3, 0.99999487),
     4: (4, 0.97749537),
     5: (5, 0.66259474),
     6: (5, 0.97694141),
     7: (3, 0.79184127),
     8: (3, 0.98323917),
     9: (3, 0.49540526)}




```python
judge_file('single_files/Untitled.data')
```




    (5, 0.73703039)




