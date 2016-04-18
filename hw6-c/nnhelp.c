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
#define LOSS_G(x,y,z) mean_sq_loss_deriv(x,y,z)
#else
#define LOSS(x,y) cross_ent_loss(x,y)
#define LOSS_G(x,y,z) cross_ent_loss_deriv(x,y,z)
#endif

double epsilon;

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
  /*
  double x[NUM_FEATURES + 1];
  memcpy(x,finput,(NUM_FEATURES + 1) * sizeof(double));
  */
  memset(dst,0,(NUM_FEATURES + 1) * sizeof(double));
  finput[NUM_FEATURES] = 1.0;

  dgemv(hidden,finput,dst,NUM_HIDDEN,NUM_FEATURES + 1);
}

void output_output(double* output, double* hinput, double* dst)
{
  /*
  double x[NUM_HIDDEN + 1];
  memcpy(x,hinput,(NUM_HIDDEN + 1 ) * sizeof(double));
  */
  memset(dst,0,(NUM_HIDDEN + 1) * sizeof(double));
  hinput[NUM_HIDDEN] = 1.0;

  dgemv(output,hinput,dst,NUM_OUTPUTS,NUM_HIDDEN + 1);
}

/* Make sure inputs has an extra spot to add the 1 vector */
void nn_outputs(double* inputs, double* hidden, double* output, double* dst)
{
  hidden_output(hidden, inputs, dst);
  double* d1 = dst + NUM_HIDDEN + 1;
  for(int i=0;i<NUM_HIDDEN;i++)
  {
    d1[i] = tanh(d1[i]);
  }

  double* d2 = dst + 2 * (NUM_HIDDEN + 1);
  output_output(output,dst,d2);
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    d2[i] = sigmoid(d2[i]);
  }
}

void output_weight_grad(double* y, double* z, double* loss_gradient, double* h, double* dst)
{
  /* sum( (z_i - y_i) ^ 2 ) * z(1-z) * h(vec) */
  double sum_diffsq = 0.0;
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    sum_diffsq += loss_gradient[i];
  }
  double s_deriv[NUM_OUTPUTS];
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    s_deriv[i] = z[i] * (1 - z[i]);
  }

  outerprod(h,s_deriv,dst,sum_diffsq,epsilon,NUM_HIDDEN + 1,NUM_OUTPUTS);
}

void hidden_out_grad(double* y, double* z, double* loss_gradient, double* weights, double* dst)
{
  /* sum( sum( (z_i - y_i) ^2 ) z(1-z) * w_i(vec) ) */
  double sum_diffsq = 0.0;
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    sum_diffsq += loss_gradient[i];
  }

  double s_deriv[NUM_OUTPUTS];
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    s_deriv[i] = z[i] * (1 - z[i]);
  }

  dgemv_rev(weights,s_deriv,dst,NUM_HIDDEN + 1,NUM_OUTPUTS,sum_diffsq);
}

void hidden_weight_grad(double* h_grad, double* q, double* f, double* dst)
{
  /* (1 - q^2) * f OUTERPROD dL/dh */
  double row[NUM_HIDDEN];
  for(int i=0;i<NUM_HIDDEN;i++)
  {
    row[i] = (1 - q[i] * q[i]) * h_grad[i];
  }

  outerprod(row,f,dst,1.0,epsilon,NUM_HIDDEN,NUM_FEATURES+1);
}

void backprop(double* inputs, double* hidden, double* output, double* labels)
{
  double buf[NUM_HIDDEN + 2 * (NUM_OUTPUTS + 1)];
  double *hidden_out = buf;
  double *output_out = buf + NUM_HIDDEN + 1;
  nn_outputs(inputs,hidden,output,buf);

  double gradient[NUM_OUTPUTS];
  LOSS_G(output_out,labels,gradient);

  double out_g[NUM_OUTPUTS];
  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    out_g[i] = gradient[i] * (output_out[i] * (1-output_out[i]));
  }

  double hidden_g[NUM_HIDDEN + 1];
  for(int i=0;i<NUM_HIDDEN + 1;i++)
  {
    hidden_g[i] = 0.0;
    for(int j=0;j<NUM_OUTPUTS;j++)
    {
      hidden_g[i] += out_g[j] * (1 - hidden_out[i] * hidden_out[i]);
    }
  }
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

void dgemv_rev(double* A, double* x, double* y, int rows, int columns, double scale)
{
  const enum CBLAS_ORDER order = CblasRowMajor;
  const enum CBLAS_TRANSPOSE transpose = CblasTrans;
  int M = rows;
  int N = columns;
  double alpha = scale;
  int lda = N;
  int incX = 1;
  double beta = 1.0;
  int incY = 1;

  cblas_dgemv(order,transpose,M,N,alpha,A,lda,x,incX,beta,y,incY);
}

void outerprod(double* A, double* B, double* C, double scale, double learning, int A_len, int B_len)
{
  const enum CBLAS_ORDER order = CblasRowMajor;
  const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
  const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
  int M = A_len;
  int N = B_len;
  int K = 1;
  double alpha = scale * learning;
  int lda = 1;
  int ldb = B_len;
  int ldc = B_len;
  double beta = 1.0;

  cblas_dgemm(order,TransA,TransB,M,N,K,alpha,A,lda,B,ldb,beta,C,ldc);
}

off_t xy2off(int row, int col, size_t row_size)
{
  return col * row_size + row;
}
