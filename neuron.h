#ifndef NEURON_H
#define NEURON_H


struct neuron {
    double bias;
    double* weights;
    double* inputs;
    int size;
};

struct network{
    struct neuron* neuron;
    int size;
};



/*
    Calculates the dot product of inputs and weights + bias for a given neuron
    @param neuron The neuron containing weights, inputs, and bias
    @return The computed output (sum of products + bias)
 */
double dot_prod(struct neuron neuron);

/*
    Creates a neuron with size initialized as INPUT
*/
struct neuron create_neuron();

// Creates a network with size initialized as NR_NEURONS
struct network create_network();

/*
    Assigns input from an input vector to every neuron in the network
    @param inputs An input vector
    @param network A network of neurons
*/
void assign_inputs(double* inputs, struct network* network);

/*
    Assigns weights from a weight matrix to the corresponding neuron in the network
    @param weights A weight matrix of size (NR_NEURONS, INPUTS)
    @param network A network of neurons
*/
void assign_weights(double weights[], struct network* network);

/*
    Assigns bias from a bias vector to the corresponding neuron in the network
    @param bias A vector of biases
    @param network A network of neurons
*/
void assign_bias(double* bias, struct network* network);

// Frees a neuron from memory
void free_neuron(struct neuron neuron);

// Print the output of every neuron in a network
void network_output(struct network* network);






#endif