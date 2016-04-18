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

#define MEAN_SQ_LOSS

int main()
{
  double A[(NUM_FEATURES + 1) * NUM_HIDDEN];
  double X[NUM_FEATURES + 1];
  double Y[NUM_FEATURES + 1];


  /* TEST 1 */
  for(int y=0;y<NUM_OUTPUTS;y++)
  {
    for(int x=0;x<NUM_OUTPUTS + 1;x++)
    {
      off_t idx = xy2off(x,y,NUM_OUTPUTS);
      /*
      printf("x %d, y %d, idx %d\n",x,y,(int)idx);
      */
      A[idx] = (x==y)? 1.0 : 0.0;
    }
  }

  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    X[i] = i * 1.0;
    Y[i] = 0.0;
  }

  dgemv(A,X,Y,NUM_OUTPUTS,NUM_OUTPUTS);

  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    printf("Y[%d] = %lf\n",i,Y[i]);
  }

  printf("\n\n\n\n");






  /* TEST 2 */

  double Z[6];
  for(int i=0;i<6;i++)
  {
    Z[i] = 0.0;
    X[i] = 1.0;
    Y[i] = 1.0;
  }
  outerprod(X,Y,Z,0.5,0.5,2,3);
  for(int i=0;i<6;i++)
  {
    printf("Z[%d] = %lf\n",i,Z[i]);
  }

  printf("\n\n\n\n");











  /* TEST 3 */



  for(int y=0;y<NUM_OUTPUTS;y++)
  {
    for(int x=0;x<NUM_OUTPUTS + 1;x++)
    {
      off_t idx = xy2off(x,y,NUM_OUTPUTS);
      A[idx] = 0.0;
    }
  }

  for(int y=0;y<3;y++)
  {
    for(int x=0;x<2;x++)
    {
      off_t idx = xy2off(x,y,2);
      A[idx] = idx + 1.0;
    }
  }

  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    X[i] = 0.0;
    Y[i] = 0.0;
  }

  for(int i=0;i<3;i++)
  {
    X[i] = 1.0;
  }

  dgemv_rev(A,X,Y,3,2,1);

  for(int i=0;i<2;i++)
  {
    printf("Y[%d] = %lf\n",i,Y[i]);
  }

  printf("\n\n\n\n");

  return 0;
}
