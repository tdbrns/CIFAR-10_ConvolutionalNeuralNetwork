# Overview
This program is meant to demonstrate and explain how convolutional neural networks are trained to recognize digital images. The code written for version of the CIFAR-10 convolutional neural network is based on course assignments and materials from [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/).  
### Requirements
- Python 3.10 or later 
- Python libraries: numpy and matplotlib
- Jupyter notebook (for viewing .ipynb files)

# Neural Networks Explained
Neural networks typically consist of multiple layers of interconnected nodes, or **neurons**, that iteratively process each portion of a data sample derived from a larger dataset. Before processing starts, each neuron in every **hidden layer** (a network layer in which the output of the preceding layer is processed) is assigned an arbitrary **weight** value and an arbritrary **bias** value - these values are adjusted during neural network training to improve the overall accuracy of the network when identifying images.  
The neural network model for this program uses **stochastic gradient descent (SGD)** during training to "learn" the most optimal way to minimize inaccuracy, or **loss**, when identifying a given sample image. Here is a step-by-step explanation of how SGD is used: 
1. The CIFAR-10 dataset is divided into batches of randomly selected labeled images to improve training efficiency. One batch is to be passed through the neural network with each iteration.
2. Each pixel of a sample image from the current batch is assigned a grayscale value between 1.0 and 0.0 (pixels closer to black have a value closer to 0.0; pixels closer to white have a value closer to 1.0).
3.  Each neuron in the **input layer** (the first network layer) is assigned the value of a pixel from the given sample image, and each neuron in the hidden layer is assigned a random weight and bias value.
4.  Each neuron in the input layer passes its value to a neuron in the first hidden layer to be processed by an **activation function** that determines how "activated" a neuron should be (i.e., how close to 1.0 a neuron's value is after adding bias). Activation functions essentially improve the network's learning capability by introducing non-linearity to the network so that the weights and biases are not simply adjusted in an undesirably linear fashion. The output of the activation function is passed to the next hidden layer as input to be processed by an another activation function. This recursive process of passing data through the hidden layers is called **forward progpagation**.
> **Neuron Output with Activation Function**:  
> $f()$ =  activation function  
> $w$ = weight assigned to the neuron in a hidden layer.  
> $x$ = output of the preceding neuron.  
> $b$ = bias asigned to a neuron in a hidden layer.  
> $y$ = output of the current neuron.
> ### $$f(wx + b) = y$$
5.  When the neuron output values have been passed forward through all the hidden layers, they will reach the **output layer** (the last network layer) where the final output values will be processed by a **loss function** (a.k.a. cost function or error function) that measures how far off the **actual output** is from the **expected output** of the neural network. Typically the expected output has one neuron in the output layer with a value of 1.0, or *true*, and the other neurons in the output layer will have a value of 0, or *false*. The loss function essentially takes the difference of the actual value and the expected value of each output layer neuron and averages them to produce an average loss value. The lower the average loss value, the more accurate the network is.
6.  The negative derivative of the loss function, or the negative gradient, is calculated. The negative gradient indicates how the weights and biases should be adjusted to get the average loss value as close to 0 as possible.
7.  After the negative gradient has been calculated using the loss function and the values of the output layer neurons, the weights and biases of the neurons in the hidden layer adjacent to the output layer are adjusted in a way that should bring the average loss value closer to 0. Then, the weights and biases of the next hidden layer are updated to accomodate the the updated weights and biases of the previous hidden layer. This recursive process called **backpropagation** continues until all the weights and biases of the hidden layers have been updated.  

# The CIFAR-10 Convolutional Neural Network
A **convolutional neural network**, or a CNN, is a neural network that is typically designed to process and identify images.  
The CIFAR-10 CNN accepts 60,000 32x32 color images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) as input. There are 10 image classes; the class that each image is placed into depends on the object the image displays (e.g., automobile, cat, ship, etc.). 50,000 labeled images are used for training and 10,000 unlabeled images are used for testing.  
The most notable features of this neural network are listed below.  
- Either 2 or 3 hidden neuron layers can be utilized in this neural network.
- Either **ReLU** or **sigmoid** activation functions can be used to add non-linearity to the output of the hidden layers. The derivatives of the ReLU and sigmoid activation functions are used to compute the negative gradient of the network during backpropagation.  
> ReLU Activation Function  
> $x$ = neuron value  
> $max(0,x)$ = return $x$ if $x$ > 0; otherwise, return 0
> ### $$f(x) = max(0,x)$$  

> Sigmoid Activation Function  
> $x$ = neuron value
> ### $$f(x) = \frac{1}{1 + e^{-x}}$$
- **Softmax** function is used in the output layer to convert the neuron output values into a probability distribution representing the probability of each image class being the correct class.
> Softmax Function   
> $y_i$ = neuron output values for a specified image class  
> $\sum_{k=1}^n e^{y_k}$ = sum of all neuron output values for all image classes  
> $P$ = predicted probability distribution  
> $c$ = total number of image classes
> ### $$f(y_i) = \frac{e^{y_i}}{\sum_{k=1}^c e^{y_k}} = P$$
- **Softmax Loss** is used to calculate the average loss. Basically, softmax loss is the softmax function with the addition of a **cross-entropy** function, which calculates the loss by taking the sum of the negative logarithms of the probabilities produced by the softmax function.  
> Cross-Entropy Loss Function    
> $P(x)$ = true probability distribution  
> $\bar{P}(x)$ = predicted probability distribution given by softmax  
> $x_i$ = sameple image from a specified image class  
> $c$ = total number of classes
> ### $$CE = -\sum_{i=1}^c P(x_i)⋅\log(\bar{P}(x_i))$$
- **L2 Regularization**, or **ridge regression**, is used to prevent overfitting by penalizing the average loss value with the sum of the squares of the model weights. **Overfitting** occurs when the model fails to generalize, or accurately identify images that are not included in the training dataset.
