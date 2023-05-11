#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot8x8( int, double *, int,  double *, int, double *, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=8 ){        /* Loop over the rows of C */
    for ( i=0; i<m; i+=8 ){        /* Loop over the columns of C, unrolled by 4 */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot8x8( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}


void AddDot8x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  __m512d a_reg;
  __m512d c_reg_j0 = _mm512_load_pd(&C( 0, 0 ));  //C(0:7,0) 
  __m512d c_reg_j1 = _mm512_load_pd(&C( 0, 1 ));  //C(0:7,1)
  __m512d c_reg_j2 = _mm512_load_pd(&C( 0, 2 ));  //C(0:7,2)
  __m512d c_reg_j3 = _mm512_load_pd(&C( 0, 3 ));  //C(0:7,3)
  __m512d c_reg_j4 = _mm512_load_pd(&C( 0, 4 ));  //C(0:7,4)
  __m512d c_reg_j5 = _mm512_load_pd(&C( 0, 5 ));  //C(0:7,5)
  __m512d c_reg_j6 = _mm512_load_pd(&C( 0, 6 ));  //C(0:7,6)
  __m512d c_reg_j7 = _mm512_load_pd(&C( 0, 7 ));  //C(0:7,7)
//	  a_reg, a_reg_j5, a_reg_j6, a_reg_j7;
  __m512d b_reg_j0, b_reg_j1, b_reg_j2, b_reg_j3,\
	  b_reg_j4, b_reg_j5, b_reg_j6, b_reg_j7;
  //unroll 8
  for ( int p=0; p<k; p++ ){ 
    //compute C(0:7, 0)
    a_reg = _mm512_load_pd(&A( 0, p ));
    b_reg_j0 = _mm512_set1_pd(B( p, 0 ));
    c_reg_j0 = _mm512_fmadd_pd(a_reg, b_reg_j0, c_reg_j0);
    //compute C(0:7, 1)
    b_reg_j1 = _mm512_set1_pd(B( p, 1 ));
    //a_reg = _mm512_load_pd(&A( 1, p ));
    c_reg_j1 = _mm512_fmadd_pd(a_reg, b_reg_j1, c_reg_j1);
    //compute C(0:7, 2)
    b_reg_j2 = _mm512_set1_pd(B( p, 2 ));
    //a_reg = _mm512_load_pd(&A( 2, p ));
    c_reg_j2 = _mm512_fmadd_pd(a_reg, b_reg_j2, c_reg_j2);
    //compute C(0:7, 3)
    b_reg_j3 = _mm512_set1_pd(B( p, 3 ));
    //a_reg = _mm512_load_pd(&A( 3, p ));
    c_reg_j3 = _mm512_fmadd_pd(a_reg, b_reg_j3, c_reg_j3);
    //compute C(0:7, 4)
    b_reg_j4 = _mm512_set1_pd(B( p, 4 ));
    //a_reg = _mm512_load_pd(&A( 4, p ));
    c_reg_j4 = _mm512_fmadd_pd(a_reg, b_reg_j4, c_reg_j4);
    //compute C(0:7, 5)
    b_reg_j5 = _mm512_set1_pd(B( p, 5 ));
    //a_reg = _mm512_load_pd(&A( 5, p ));
    c_reg_j5 = _mm512_fmadd_pd(a_reg, b_reg_j5, c_reg_j5);
    //compute C(0:7, 6)
    b_reg_j6 = _mm512_set1_pd(B( p, 6 ));
    //a_reg = _mm512_load_pd(&A( 6, p ));
    c_reg_j6 = _mm512_fmadd_pd(a_reg, b_reg_j6, c_reg_j6);
    //compute C(0:7, 7)
    b_reg_j7 = _mm512_set1_pd(B( p, 7 ));
    //a_reg = _mm512_load_pd(&A( 7, p ));
    c_reg_j7 = _mm512_fmadd_pd(a_reg, b_reg_j7, c_reg_j7);
  }
  _mm512_store_pd(&C( 0, 0 ), c_reg_j0);
  _mm512_store_pd(&C( 0, 1 ), c_reg_j1);
  _mm512_store_pd(&C( 0, 2 ), c_reg_j2);
  _mm512_store_pd(&C( 0, 3 ), c_reg_j3);
  _mm512_store_pd(&C( 0, 4 ), c_reg_j4);
  _mm512_store_pd(&C( 0, 5 ), c_reg_j5);
  _mm512_store_pd(&C( 0, 6 ), c_reg_j6);
  _mm512_store_pd(&C( 0, 7 ), c_reg_j7);
}
