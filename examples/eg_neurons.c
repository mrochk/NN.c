#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "../loss/loss.h"
#include "../utils/utils.h"
#include "../tensors/tensors.h"

#include "../models/neuron/neuron.h"

void run_neurons_example() {
    /* create data */
    Vector x = new_random_float_vector_(20);

    Vector y = new_zeros_vector_(20);
    for (int i = 0; i < y->n; i++) { 
        double xd = (double) x->d[i];
        y->d[i] = (float) sin(2.0 * PI * xd); 
    }

    /* printing the data */
    puts("X:");   print_vector(x);
    puts("\nY:"); print_vector(y);

    /* creating two neurons */
    Neuron h = new_neuron_(1, Sigmoid);
    Neuron o = new_neuron_(1, Linear);

    /* forward pass */

    Vector preds = new_zeros_vector_(x->n);
    Vector hiddens = new_zeros_vector_(x->n);
    Vector x_ = new_zeros_vector_(1);

    for (int i = 0; i < x->n; i++) {
        x_->d[0] = x->d[i];

        // hidden layer
        float hidden = neuron_forward(h, x_);
        hiddens->d[i] = hidden;
        x_->d[0] = hidden; 

        // output layer
        float out = neuron_forward(o, x_);

        preds->d[i] = out;
    }

    puts("\nModel Predictions:");
    print_vector(preds);

    float loss = MSE(preds, y);
    printf("\nLoss: %.2f\n", loss);

    float dw_o = 0.0;
    for (int i = 0; i < x->n; i++) {
        float error = y->d[i] - preds->d[i];

        dw_o += -2.0F * error * hiddens->d[i]; 
    }
    dw_o = dw_o / x->n;

    printf("dw_o = %.2f\n", dw_o);

    print_vector(o->w);
    o->w->d[0] -= 0.1f * dw_o;
    print_vector(o->w);

    for (int j = 0; j < 5; j++) {

        for (int i = 0; i < x->n; i++) {
            x_->d[0] = x->d[i];

            // hidden layer
            float hidden = neuron_forward(h, x_);
            hiddens->d[i] = hidden;
            x_->d[0] = hidden; 

            printf("HIDDEN: %.2f\n", hidden);

            // output layer
            float out = neuron_forward(o, x_);

            preds->d[i] = out;
        }

        puts("\nModel Predictions:");
        print_vector(preds);

        loss = MSE(preds, y);
        printf("\nLoss: %.2f\n", loss);

        dw_o = 0.0;
        for (int i = 0; i < x->n; i++) {
            float error = y->d[i] - preds->d[i];

            dw_o += -2.0F * error * hiddens->d[i]; 
        }
        dw_o = dw_o / x->n;

        printf("dw_o = %.2f\n", dw_o);

        print_vector(o->w);
        o->w->d[0] -= 0.1f * dw_o;
        print_vector(o->w);
    }

    free_vector(x); free_vector(y); free_vector(preds); free_vector(hiddens); free_vector(x_);
    free_neuron(h); free_neuron(o);
}

