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
  double A[NUM_OUTPUTS * NUM_OUTPUTS];
  double x[NUM_OUTPUTS];
  double y[NUM_OUTPUTS];

  for(int y=0;y<NUM_OUTPUTS;y++)
  {
    for(int x=0;x<NUM_OUTPUTS + 1;x++)
    {
      off_t idx = xy2off(x,y,NUM_OUTPUTS);
      printf("x %d, y %d, idx %d\n",x,y,(int)idx);
      A[idx] = (x==y)? 1.0 : 0.0;
    }
  }

  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    x[i] = i * 1.0;
    y[i] = 0.0;
  }

  dgemv(A,x,y,NUM_OUTPUTS,NUM_OUTPUTS);

  for(int i=0;i<NUM_OUTPUTS;i++)
  {
    printf("y[%d] = %lf\n",i,y[i]);
  }
  return 0;
}
