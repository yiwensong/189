#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define NUM_FEATURES 784
#define NUM_HIDDEN 200
#define NUM_OUTPUTS 10

/* NEURAL NET STRUCTURES */

/* hidden layers are represented by a (F+1)xH double array */
/* output layers are represented by a (H+1)xO double array */
/* inputs are represented by arrays of size F */


/* BASIC UPDATE FUNCTIONS */
/* The sigmoid function and its derivative*/
double sigmoid(double x);

double dsigmoid(double x);


/* The tanh funciton and its derivative */
/* The tanh function is already defined in math.h */
double dtanh(double x);



/* ERROR CALCS */
double mean_sq_loss(double* pred, double* actual);
double cross_ent_loss(double* pred, double* actual);

/* ERROR CALC GRADIENTS */
void mean_sq_loss_deriv(double* pred, double* actual, double* dst);
void cross_ent_loss_deriv(double* pred, double* actual, double* dst);



/* HELPER FUNCTIONS FOR NN CALCULATIONS */
void hidden_output(double* hidden, double* finput, double* dst);
void output_output(double* output, double* hinput, double* dst);
void nn_outputs(double* inputs, double* hidden, double* output, double* dst);
void backprop(double* inputs, double* hidden, double* outputs, double* labels);
void init_units(double* hidden, double* outputs);



/* Other util functions */
void shuffle(size_t* idx, size_t len);
void dgemv(double* A, double* x, double* y, int rows, int columns);
off_t xy2off(int row, int col, size_t row_size);
