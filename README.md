# Iceberg or boat?

This program is used to demonstrate the use of a convolutional neural network in tensorflow used on the Statoil/C-CORE Iceberg Classifier Challenge dataset found at kaggle.com ( https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data).
The purpose of this is to demonstrate the use of tensorflow to create a convolutional neural net that creates a straightforward graph for new users.

## Getting Started

This scirpt uses python 3.5 and tensorflow 1.1.0
Supporting libraries are :
pandas for data structuring
numpy for linear algebra (mostly dealing with matricies)
sklearn for log loss metrics for this particular competition

## Usage

I reccomend using a python virtual environment specifically for use of tensorflow. I am using anaconda with a conda virtual environment

```python
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
```

These are the basis for our backbone of the CNN, just creating our weight variables/ bias variables of specific shapes to handle neurons math.
We also have the two layers essential to a basic CNN the convolution layer and the max pooling layer.

x: Input

W: The weight variable for the layer

## Training

![](https://i.imgur.com/iBjFnyj.png)

## Graph Creation

### Full model:
![](https://media.giphy.com/media/xUNda23psAEN55g8yQ/giphy.gif)


### Convolution Layer:

![](https://i.imgur.com/3oLGVmU.png)

Each run builds a clean tensorboard graph that is used to demonstrate the framework of the network
it will be put into a folder graph and can be run from command line by:

> activate tensorflow virtual environment

tensorboard --logdir graph

From here you can go to your localhost:6006 and under the "graphs" section you may view your interactive board

## Author

Jacob Biloki : bilokij@gmail.com

## License

This project is licensed under the MIT License
