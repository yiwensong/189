#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <cblas.h>

#include "nnhelp.h"

#define NUM_FEATURES 784
#define NUM_HIDDEN 200
#define NUM_OUTPUTS 10

#ifdef MEAN_SQ_LOSS
#define LOSS(x,y) mean_sq_loss(x,y)
#define LOSS_G(x,y,z) mean_sq_loss(x,y,z)
#else
#define LOSS(x,y) cross_ent_loss(x,y)
#define LOSS_G(x,y,z) cross_ent_loss(x,y,z)
#endif

/* BASIC UPDATE FUNCTIONS */
/* The sigmoid function and its derivative*/
double sigmoid(double x)
{
  return 1.0/(1.0 + exp(x));
}

double dsigmoid(double x)
{
  return sigmoid(x) * (1 - sigmoid(x));
}


/* The tanh funciton and its derivative */
/* The tanh function is already defined in math.h */
double dtanh(double x)
{
  register double tmp = tanh(x);
  return 1 - tmp * tmp;
}




/* ERROR CALCS */
double mean_sq_loss(double* pred, double* actual)
{
  double sum = 0.0;
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    sum += pow(pred[i] - actual[i],2.0)/2.0;
  }
  return sum;
}

double cross_ent_loss(double* pred, double* actual)
{
  double sum = 0.0;
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    double s1 = pred[i] > .001 ? pred[i] : .001;
    double s2 = (1.0 - pred[i]) > .001 ? pred[i] : .999;
    sum -= actual[i] * s1 + (1.0 - actual[i]) * (1.0 - s2);
  }
  return sum;
}



/* ERROR CALC GRADIENTS */
void mean_sq_loss_deriv(double* pred, double* actual, double* dst)
{
  for(int i=0;i<NUM_OUTPUTS+1;i++)
  {
    dst[i] = pred - actual;
  }
}
void cross_ent_loss_deriv(double* pred, double* actual, double* dst)
{
  for(int i=0;i<NUM_OUTPUTS+1;i++)
  {
    double s1 = pred[i] > .001 ? pred[i] : .001;
    double s2 = (1.0 - pred[i]) > .001 ? pred[i] : .999;
    dst[i] = actual[i]/s1 - (1.0 - actual[i])/(1.0 - s2);
  }
}



/* HELPER FUNCTIONS FOR NN CALCULATIONS */
void hidden_output(double* hidden, double* finput, double* dst)
{
  double x[NUM_FEATURES + 1];
  memcpy(x,finput,(NUM_FEATURES + 1) * sizeof(double));
  x[NUM_FEATURES] = 0.0;

  dgemv(hidden,x,dst,NUM_HIDDEN,NUM_FEATURES + 1);
}

void output_output(double* output, double* hinput, double* dst)
{
  double x[NUM_HIDDEN + 1];
  memcpy(x,hinput,(NUM_HIDDEN + 1 ) * sizeof(double));
  x[NUM_HIDDEN] = 0.0;

  dgemv(output,x,dst,NUM_OUTPUTS,NUM_HIDDEN + 1);
}

void nn_outputs(double* inputs, double* hidden, double* output, double* dst)
{
  double hout[NUM_HIDDEN];
  hidden_output(hidden, inputs, hout);
  for(int i=0;i<NUM_HIDDEN;i++)
  {
    hout[i] = tanh(hout[i]);
  }

  output_output(output,hout,dst);
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    dst[i] = sigmoid(dst[i]);
  }
}

void backprop(double* inputs, double* hidden, double* outputs)
{
  
}

void init_units(double* hidden, double* outputs)
{
  srand(time(NULL));
  for(int i=0;i<NUM_HIDDEN * (NUM_FEATURES + 1);i++)
  {
    hidden[i] = (((double) rand()) / 1.0)/ RAND_MAX;
  }

  for(int i=0;i<NUM_OUTPUTS * (NUM_HIDDEN + 1);i++)
  {
    hidden[i] = (((double) rand()) / 1.0)/ RAND_MAX;
  }
}



/* Other util functions */
void shuffle(size_t* idx, size_t len)
{
  for(size_t i=0;i<len;i++)
  {
    idx[i] = i;
  }
  for(size_t i=0;i<len;i++)
  {
    int r = rand() % (len - i);
    int tmp = idx[i + r];
    idx[i + r] = idx[i];
    idx[i] = tmp;
  }
}

void dgemv(double* A, double* x, double* y, int rows, int columns)
{
  const enum CBLAS_ORDER order = CblasRowMajor;
  const enum CBLAS_TRANSPOSE transpose = CblasNoTrans;
  int M = rows;
  int N = columns;
  double alpha = 1.0;
  int lda = N;
  int incX = 1;
  double beta = 1.0;
  int incY = 1;

  cblas_dgemv(order,transpose,M,N,alpha,A,lda,x,incX,beta,y,incY);
}

off_t xy2off(int row, int col, size_t row_size)
{
  return col * row_size + row;
}
