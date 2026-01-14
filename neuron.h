#ifndef NEURON_H
#define NEURON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// THE STRUCTURES

/*
    @struct neuron
    Represents a single neuron in a neural network
*/
struct neuron {
    double bias;          
    double* weights;      
    double* inputs;       
    int size;             
};

/**
    @struct network
    Represents a layer of neurons (or entire network)
*/
struct network {
    struct neuron* neuron; 
    int size;               
};

/**
    @struct dataset
    Represents a loaded dataset from CSV
    @param data is a 2d array of inputs
*/
struct dataset {
    double** data;   
    int* labels;     
    int rows;        
    int cols;       
};

/**
    @struct training_result
    Result returned from a single training pass
 */
struct training_result {
    double loss;          
    int predicted_class;   
};

// Neuron stuff

/**
    Creates and initializes a single neuron with default values
    @return A neuron with bias=0, weights=NULL, inputs=NULL, size=0
*/
struct neuron create_neuron();

/**
    Frees the weights array of a neuron from memory
    @param neuron Pointer to the neuron to free
*/
void free_neuron(struct neuron* neuron);

// Network stuff

/**
    Creates an empty network structure with given size
    @param neurons Number of neurons in this network
    @return A network with neuron array allocated but not initialized
*/
struct network create_network(int neurons);

/**
    Creates a network with neurons fully initialized
    @param neurons Number of neurons to create
    @return A network with all neurons allocated and initialized to zero
*/
struct network create_network_with_neurons(int neurons);

/**
    Frees an entire network and all its neurons from memory
    @param network Pointer to the network to free
*/
void free_network(struct network* network);

// Weight initilization 

/**
    Creates a random weight array with small values between -0.01 and 0.01
    @param cols Number of weights to generate
    @return Dynamically allocated array of random weights
*/
double* create_ran_weights(int cols);

/**
    Creates a small random bias value between -0.001 and 0.001
    @return A random bias value
*/
double create_ran_bias();

/**
    Frees a weights array from memory
    @param weights Pointer to the weights array to free
*/
void free_weights(double* weights);

// Dataset calculations

/**
    Reads a CSV file into a dataset structure
    Assumes first column is the label, remaining columns are pixels (0-255 pixel values)
    @param filename Path to the CSV file
    @return A dataset structure with loaded data and labels
*/
struct dataset read_csv_2d(const char* filename);

/**
    Frees all memory associated with a dataset
    @param ds Pointer to the dataset to free
*/
void free_dataset(struct dataset* ds);

// Activation functions

/**
    ReLU (Rectified Linear Unit) activation function
    Returns max(0, x)
    @param x Input value
    @return Activated value
*/
double relu(double x);

/**
    Sigmoid activation function
    Returns 1 / (1 + e^(-x))
    @param x Input value
    @return Sigmoid output (0-1)
*/
double sigmoid(double x);

// Forward pass

/**
    Computes output of hidden layer neurons using ReLU activation
    @param ds Pointer to dataset
    @param network Pointer to hidden network
    @param image_inx Index of the image to process
    @return Array of hidden layer outputs (must be freed by caller)
*/
double* calc_output_hidden(struct dataset* ds, struct network* network, int image_inx);

/**
    Computes output probabilities using softmax
    @param hidden_out Output from hidden layer
    @param out_network Output network
    @param hidden_size Number of hidden neurons
    @return Array of probabilities for each class 0-9 (must be freed by caller)
*/
double* calc_output(double* hidden_out, struct network* out_network, int hidden_size);

// Loss and error calculations

/**
    Calculates cross-entropy loss
    Loss = -log(P[correct_class])
    @param ds Pointer to dataset
    @param pred Predicted probabilities for all classes
    @param img_inx Index of the image
    @return The loss value
*/
double calc_loss(struct dataset* ds, double* pred, int img_inx);

/**
    Calculates error deltas for output layer neurons
    delta = prediction - target
    @param ds Pointer to dataset
    @param prediction Predicted probabilities
    @param img_inx Index of the image
    @return Array of deltas for each output neuron (must be freed by caller)
*/
double* output_layer_errors(struct dataset* ds, double* prediction, int img_inx);

/**
    Calculates error deltas for hidden layer neurons using backpropagation
    @param hidden_out Output values from hidden layer
    @param hidden_size Number of hidden neurons
    @param delta_output Deltas from output layer
    @param output_network Pointer to output network
    @return Array of deltas for each hidden neuron (must be freed by caller)
*/
double* hidden_layer_errors(double* hidden_out, int hidden_size, double* delta_output, struct network* output_network);

// Weight updates

/**
    Updates weights and bias of output layer neurons using gradient descent
    @param out_network Pointer to output network
    @param delta_output Deltas from output layer
    @param eta Learning rate
    @param hidden_out Output from hidden layer (used as inputs to output layer)
*/
void update_output_weights_bias(struct network* out_network, double* delta_output, double eta, double* hidden_out);

/**
    Updates weights and bias of hidden layer neurons using gradient descent
    @param hidden_network Pointer to hidden network
    @param delta_hidden Deltas from hidden layer
    @param ds Pointer to dataset
    @param img_inx Index of the image
    @param hidden_size Number of hidden neurons
    @param eta Learning rate
*/
void update_hidden_weights_bias(struct network* hidden_network, double* delta_hidden, struct dataset* ds, int img_inx, int hidden_size, double eta);

// Actual trainging

/**
    Performs one training pass (forward + backward + weight update)
    @param ds Pointer to dataset
    @param hidden_network Pointer to hidden layer
    @param output_network Pointer to output layer
    @param img_inx Index of the training image
    @param eta Learning rate (e.g., 0.01)
    @return training_result containing loss and predicted class
*/
struct training_result one_pass(struct dataset* ds, struct network* hidden_network, struct network* output_network, int img_inx, double eta);

#endif // NEURON_H





