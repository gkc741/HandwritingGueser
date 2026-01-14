#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "neuron.h"




struct neuron create_neuron(){
    struct neuron n;
    n.bias = 0;
    n.weights = NULL;
    n.inputs = NULL;
    return n;
}



struct network create_network(int neurons){
    struct network n;
    n.size = neurons;
    n.neuron = NULL;
    return n;
}



struct network create_network_with_neurons(int neurons){
    struct network network = create_network(neurons);
    network.neuron = malloc(sizeof(struct neuron) * neurons);  // Allocate for the actual number of neurons
    for(int i = 0; i < neurons; i++){
        struct neuron neuron = create_neuron();
        network.neuron[i] = neuron;
    }
    return network;
}



double* create_ran_weights(int cols){ 
    double* weights = malloc(cols * sizeof(double));
    for(int i = 0; i < cols; i++){
        double random_weight = ((double)rand() / RAND_MAX) * 0.02 - 0.01;
        weights[i] = random_weight;
    }
    return weights;
}


// Small random bias between -0.1 and 0.1
double create_ran_bias(){
    return ((double)rand() / RAND_MAX) * 0.002 - 0.001;
}




// Frees a neuron from memory
void free_neuron(struct neuron* neuron){
    free(neuron->weights);
}



// Frees an entire network from memory
void free_network(struct network* network){
    for(int i = 0; i < network->size; i++){
        free_neuron(&network->neuron[i]);
    }
    free(network->neuron);
}



void free_dataset(struct dataset* ds){
    for(int i = 0; i < ds->rows; i++){
        free(ds->data[i]);
    }
    free(ds->labels);
    free(ds->data);
}


void free_weights(double* weights){
    free(weights);
}




struct dataset read_csv_2d(const char* filename){
    struct dataset ds = {NULL, NULL, 0, 0};
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error: Couldnt find the file\n");
        return ds;
    }

    char buffer[8192]; // a buffer to skip the header (needs to be big enough!)
    
    // Skip header row and check if it was read successfully
    if(fgets(buffer, sizeof(buffer), f) == NULL){
        printf("Error: Could not read header\n");
        fclose(f);
        return ds;
    }

    ds.cols = 785;  // 1 pr pixel and a label

    int capacity = 10;
    ds.data = malloc(capacity * sizeof(double*));
    ds.labels = malloc(capacity * sizeof(int));

    
    while(fgets(buffer, sizeof(buffer), f) != NULL){

        if (ds.rows >= capacity){ 
            capacity *= 2;
            ds.data = realloc(ds.data, capacity * sizeof(double*));
            ds.labels = realloc(ds.labels, capacity * sizeof(int));
        }

        ds.data[ds.rows] = malloc((ds.cols - 1) * sizeof(double));

        char* current_number = strtok(buffer, ",");
        ds.labels[ds.rows] = atoi(current_number);  // first item in the column is the label
        current_number = strtok(NULL, ",");  // read next int aka the first pixel
        int column = 0;
        
        
        while(current_number != NULL){
            ds.data[ds.rows][column] = atof(current_number) / 255.0;  // normalize to 0-1
            column++;
            current_number = strtok(NULL, ",");
        }

        ds.rows++;
    }
    ds.cols -= 1;
    fclose(f);

    return ds;
}

double relu(double x){
    if (x > 0){
        return x;
    }
    return 0;
}


double* calc_output_hidden(struct dataset* ds, struct network* network, int image_inx){
    double* output = malloc(network->size * sizeof(double));

    for(int j = 0; j < network->size; j++){  // so for every neuron
        double sum = network->neuron[j].bias;
        for(int i = 0; i < ds->cols; i++){  // for every input
            sum += network->neuron[j].weights[i] * ds->data[image_inx][i];
        }
        output[j] = relu(sum);
    }

    return output;
}



double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

// Calculates the probability of it being every number 0-9
double* calc_output(double* hidden_out, struct network* out_network, int hidden_size){
    double* z = malloc(sizeof(double) * out_network->size);
    // Find the raw output for the 10 neurons
    for(int i = 0; i < out_network->size; i++){
        z[i] = out_network->neuron[i].bias;
        for(int j = 0; j < hidden_size; j++){
            z[i] += out_network->neuron[i].weights[j] * hidden_out[j];
        }
    }


    // find the largest z
    double max_z = z[0];
    for(int i = 1; i < out_network->size; i++){
        if (z[i] > max_z){
            max_z = z[i];
        }
    }

    // exponentiate and sum
    double sum_exp = 0;
    for(int i = 0; i < out_network->size; i++){
        z[i] = exp(z[i] - max_z);
        sum_exp += z[i];
    }

    // devide by the sum to get the probability
    for(int i = 0; i < out_network->size; i++){
        z[i] /= sum_exp;
    }

    return z;
}



double calc_loss(struct dataset* ds, double* pred, int img_inx) {
    int label = ds->labels[img_inx];  
    double loss = -log(pred[label]);  // take softmax probability of correct class
    return loss;
}

// the error contribution of the k-th output neuron
double* output_layer_errors(struct dataset* ds, double* prediction, int img_inx){
    int label = ds->labels[img_inx];

    double* detla_output = malloc(sizeof(double) * 10);
    
    for(int i = 0; i < 10; i++){
        double t = (i == label) ? 1.0 : 0.0;
        detla_output[i] = prediction[i] - t;
    }
    return detla_output;
}


// compute how much the j-th hidden neuron contributed to the total error
double* hidden_layer_errors(double* hidden_out, int hidden_size, double* delta_output, struct network* output_netwok){
    
    double* delta_hidden = malloc(sizeof(double) * hidden_size);

    for(int i = 0; i < hidden_size; i++){
        if(hidden_out[i] <= 0){  // cuz then the derivative is = 0
            delta_hidden[i] = 0;
            continue;
        }
        double sum = 0;
        for(int j = 0; j < 10; j++){
            sum += delta_output[j] * output_netwok->neuron[j].weights[i];
        }
        delta_hidden[i] = sum;
    }
    return delta_hidden;
}


void update_output_weights_bias(struct network* out_network, double* delta_output, double eta, double* hidden_out){
    // so for every neuron
    for(int i = 0; i < out_network->size; i++){
        // for every weight for every neuron
        for(int j = 0; j < out_network->neuron[i].size; j++){
            out_network->neuron[i].weights[j] -= eta * delta_output[i] * hidden_out[j];
        }
        // also the bias
        out_network->neuron[i].bias -= eta * delta_output[i];
    }
    return;
}

// hidden_size is 100 
void update_hidden_weights_bias(struct network* hidden_network, double* delta_hidden, struct dataset* ds, int img_inx, int hidden_size, double eta){
    double* x = ds->data[img_inx];

    for(int i = 0; i < hidden_size; i++){
        for(int j = 0; j < hidden_network->neuron[i].size; j++){
            hidden_network->neuron[i].weights[j] -= eta * delta_hidden[i] * x[j];
        }
        hidden_network->neuron[i].bias -= eta * delta_hidden[i];
    }
    return;
}


struct training_result one_pass(struct dataset* ds, struct network* hidden_network, struct network* output_network, int img_inx, double eta){
    // Now i have a hidden layer of 100 neurons all with 784 weights and their own bias
    // And a output layer of 10 neurons with 100 weights and bias
    double* hidden_out = calc_output_hidden(ds, hidden_network, img_inx);
    


    // Calculate the prediction using relu on hidden layers and sigmoid on output
    double* pred = calc_output(hidden_out, output_network, hidden_network->size);

    double loss = calc_loss(ds, pred, img_inx);
    
    // Find predicted class
    int pred_class = 0;
    double max_prob = pred[0];
    for(int j = 1; j < 10; j++){
        if(pred[j] > max_prob){
            max_prob = pred[j];
            pred_class = j;
        }
    }

    // now for backward pass

    // calculate deltas
    double* delta_output = output_layer_errors(ds, pred, img_inx);
    double* delta_hidden = hidden_layer_errors(hidden_out, hidden_network->size, delta_output, output_network);

    // update weights
    update_hidden_weights_bias(hidden_network, delta_hidden, ds, img_inx, hidden_network->size, eta);
    update_output_weights_bias(output_network, delta_output, eta, hidden_out);


    free(delta_output);
    free(delta_hidden);
    free(hidden_out);
    free(pred);
    
    struct training_result result = {loss, pred_class};
    return result;
}


/*
784 inputs 
100 hidden neurons that have 784 inputs, 784 weights and 1 bias pr neuron
1 output neuron with 100 inputs 100 weights and 1 bias
*/



int main(){
    srand(13);
    struct dataset ds = read_csv_2d("archive/mnist_test.csv");

    struct network hidden_network = create_network_with_neurons(100);  // hidden layer

    // create the output network
    struct network output_network = create_network_with_neurons(10);

    // Initialize hidden layer weights and biases
    for(int i = 0; i < hidden_network.size; i++){  // assign 784 weights to every neuron
        hidden_network.neuron[i].weights = create_ran_weights(ds.cols);
        hidden_network.neuron[i].size = ds.cols;
        hidden_network.neuron[i].bias = create_ran_bias();
    }

    // Initialize output layer weights and biases
    for(int i = 0; i < output_network.size; i++){ // assign 100 weights to 10 neurons
        output_network.neuron[i].weights = create_ran_weights(hidden_network.size);
        output_network.neuron[i].size = hidden_network.size;
        output_network.neuron[i].bias = create_ran_bias();
    }

    double eta = 0.01;

    for(int run = 0; run < 10; run++){
        double total_loss = 0;
        int correct = 0;
        
        for(int i = 0; i < ds.rows; i++){
            struct training_result result = one_pass(&ds, &hidden_network, &output_network, i, eta);
            
            if(result.predicted_class == ds.labels[i]) correct++;
            total_loss += result.loss;
        }
        
        printf("Run %d: Loss = %.4f, Accuracy = %.1f%%\n", run + 1, total_loss / ds.rows, (100.0 * correct) / ds.rows);
    }




    free_network(&output_network);
    free_network(&hidden_network);
    free_dataset(&ds);
    return 0;
}

