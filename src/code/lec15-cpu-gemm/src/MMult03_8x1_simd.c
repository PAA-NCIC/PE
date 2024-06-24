#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x1( int, double *, int,  double *, int, double *, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=1 ){        /* Loop over the rows of C */
    for ( i=0; i<m; i+=8 ){        /* Loop over the columns */
      AddDot8x1( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}


void AddDot8x1( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  __m512d c_reg = _mm512_load_pd(&C( 0, 0 ));
  for (int p=0; p<k; p++ ){
    __m512d b_reg = _mm512_set1_pd(B( p, 0 ));
    __m512d a_reg = _mm512_load_pd(&A( 0, p ));
    c_reg = _mm512_fmadd_pd(a_reg, b_reg, c_reg);    
  }
  _mm512_store_pd(&C( 0, 0 ), c_reg);
}
