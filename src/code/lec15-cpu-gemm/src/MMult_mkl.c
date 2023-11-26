/* Create macros so that the matrices are stored in column-major order */
#include "mkl.h"


/* Routine for computing C = A * B + C */

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,1.0,a,m,b,k,1.0,c,m);
}


  
