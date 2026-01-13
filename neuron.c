#include <stdlib.h>
#include <stdio.h>

#include "neuron.h"

#define NR_NEURONS 3  // Antal Neurons for the weight
#define INPUT 4  // Antal input pr. neuron



double dot_prod(struct neuron neuron){
    double calc = 0;
    for(int i = 0; i < neuron.size; i++){
        calc += (neuron.inputs[i] * neuron.weights[i]);
    } 
    calc += neuron.bias;
    return calc;
}

struct neuron create_neuron(){
    struct neuron n;
    n.size = INPUT;
    n.bias = 0;
    n.weights = NULL;
    n.inputs = NULL;
    return n;
}

struct network create_network(){
    struct network n;
    n.size = NR_NEURONS;
    n.neuron = NULL;
    return n;
}

void assign_inputs(double* inputs, struct network* network){
    int size = network->size;
    int len = network->neuron[0].size;

    for(int i = 0; i < size; i++){
        network->neuron[i].inputs = malloc(len* sizeof(double));

        for(int j = 0; j < len; j++){
            network->neuron[i].inputs[j] = inputs[j];
        }
    }
}

void assign_weights(double weights[][INPUT], struct network* network){
    int size = network->size;
    int len = network->neuron[0].size;
    // Så for hver neuron
    for(int i = 0; i < size; i++){
        network->neuron[i].weights = malloc(len * sizeof(double));
        // For hver weight
        for(int j = 0; j < len; j++){
            network->neuron[i].weights[j] = weights[i][j];
        }
    }
}

void assign_bias(double* bias, struct network* network){
    int size = network->size;

    for(int i = 0; i < size; i++){
            network->neuron[i].bias = bias[i];
    }
}

void free_neuron(struct neuron neuron){
    free(neuron.inputs);
    free(neuron.weights);
    return;
}

void network_output(struct network* network){
    printf("[");
    for(int i = 0; i < network->size; i++){
        printf("%f", dot_prod(network->neuron[i]));
        if(i < network->size - 1){
            printf(", ");
        }
    }
    printf("]\n");
}


int main(){
    double inputs[INPUT] = {1, 2, 3, 2.5};

    double weights[NR_NEURONS][INPUT] = {
        {  0.2,   0.8,  -0.5,  1.0 },
        {  0.5,  -0.91,  0.26, -0.5 },
        { -0.26, -0.27,  0.17,  0.87 }
    };

    double bias[INPUT] = {2, 3, 0.5};

    struct neuron neuron1 = create_neuron();

    struct neuron neuron2 = create_neuron();

    struct neuron neuron3 = create_neuron();
    

    struct network network = create_network();


    network.neuron = malloc(sizeof(struct neuron) * network.size);
    network.neuron[0] = neuron1;
    network.neuron[1] = neuron2;
    network.neuron[2] = neuron3;

    assign_bias(bias, &network); 
    assign_inputs(inputs, &network);
    assign_weights(weights, &network);
    network_output(&network);
    return 0;

    
}

