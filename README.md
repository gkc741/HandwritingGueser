Building a Neural Network in C to guess the handwitten number.
Dataset from MNIST

Im using 784 input pixels, 100 neuons in the hidden layer, and 10 in the output for the ints 0-9.

Usin ReLU on the hidden layer, and softmax on the output layer.

Using Categorial Corss-Entropy Loss for the 10 classes i have to calculate the loss.

For optimization im using Stochastic Gradien Descent for one sample at a time, with a fixed learning rate of 0.1 rn.

