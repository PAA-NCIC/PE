#include <immintrin.h>
//1024K private L2
//half for C blocking, e.g., 512KB, can store
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#define min( i, j ) ( (i)<(j) ? (i): (j) )
/* Routine for computing C = A * B + C */
#define mb 240
#define kb 256

static void AddDot24x8( int, double *, int,  double *, int, double *, int );
static void InnerKernel( int m, int n, int k, double *a, int lda,  
                                       double *b, int ldb, 
                                       double *c, int ldc);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j, p, pb, ib;

  for(p = 0; p < k; p += kb){
    pb = min(k-p, kb);
    for ( i = 0; i < m; i += mb ){        
      ib = min(m-i, mb);
      InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc );
    }
  }
}


void InnerKernel( int m, int n, int k, double *a, int lda,  
                                       double *b, int ldb, 
                                       double *c, int ldc) {
	for(int i = 0; i < m; i += 24) {
    for(int j = 0; j < n; j += 8) {
      AddDot24x8(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
    }
  }
}

void AddDot24x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  __m512d c_00 = _mm512_load_pd(&C( 0*8, 0 ));  //C(0:7,0) 
  __m512d c_01 = _mm512_load_pd(&C( 0*8, 1 ));  //C(0:7,1)
  __m512d c_02 = _mm512_load_pd(&C( 0*8, 2 ));  //C(0:7,2)
  __m512d c_03 = _mm512_load_pd(&C( 0*8, 3 ));  //C(0:7,3)
  __m512d c_04 = _mm512_load_pd(&C( 0*8, 4 ));  //C(0:7,4)
  __m512d c_05 = _mm512_load_pd(&C( 0*8, 5 ));  //C(0:7,5)
  __m512d c_06 = _mm512_load_pd(&C( 0*8, 6 ));  //C(0:7,6)
  __m512d c_07 = _mm512_load_pd(&C( 0*8, 7 ));  //C(0:7,7)
  __m512d c_10 = _mm512_load_pd(&C( 1*8, 0 ));  //C(8:15,0)
  __m512d c_11 = _mm512_load_pd(&C( 1*8, 1 ));  //C(8:15,1)
  __m512d c_12 = _mm512_load_pd(&C( 1*8, 2 ));  //C(8:15,2)
  __m512d c_13 = _mm512_load_pd(&C( 1*8, 3 ));  //C(8:15,3)
  __m512d c_14 = _mm512_load_pd(&C( 1*8, 4 ));  //C(8:15,4)
  __m512d c_15 = _mm512_load_pd(&C( 1*8, 5 ));  //C(8:15,5)
  __m512d c_16 = _mm512_load_pd(&C( 1*8, 6 ));  //C(8:15,6)
  __m512d c_17 = _mm512_load_pd(&C( 1*8, 7 ));  //C(8:15,7)
  __m512d c_20 = _mm512_load_pd(&C( 2*8, 0 ));  //C(16:23,0)
  __m512d c_21 = _mm512_load_pd(&C( 2*8, 1 ));  //C(16:23,1)
  __m512d c_22 = _mm512_load_pd(&C( 2*8, 2 ));  //C(16:23,2)
  __m512d c_23 = _mm512_load_pd(&C( 2*8, 3 ));  //C(16:23,3)
  __m512d c_24 = _mm512_load_pd(&C( 2*8, 4 ));  //C(16:23,4)
  __m512d c_25 = _mm512_load_pd(&C( 2*8, 5 ));  //C(16:23,5)
  __m512d c_26 = _mm512_load_pd(&C( 2*8, 6 ));  //C(16:23,6)
  __m512d c_27 = _mm512_load_pd(&C( 2*8, 7 ));  //C(16:23,7)
  __m512d b_0, b_1, b_2, b_3;
  __m512d a_0, a_1, a_2;

  for ( int p=0; p<k; p++ ){ 
    a_0 = _mm512_load_pd(&A( 0 * 8, p ));
    b_0 = _mm512_set1_pd(B(p , 0 ));
    c_00 = _mm512_fmadd_pd(a_0, b_0, c_00);
    b_1 = _mm512_set1_pd(B(p , 1 ));
    a_1 = _mm512_load_pd(&A( 1 * 8, p ));
    c_01 = _mm512_fmadd_pd(a_0, b_1, c_01);
    b_2 = _mm512_set1_pd(B(p , 2 ));
    c_02 =  _mm512_fmadd_pd(a_0, b_2, c_02);
    b_3 = _mm512_set1_pd(B(p , 3 ));
    c_03 =  _mm512_fmadd_pd(a_0, b_3, c_03);

    a_2 =  _mm512_load_pd(&A( 2 * 8, p ));
    c_10 =  _mm512_fmadd_pd(a_1, b_0, c_10);
    c_11 =  _mm512_fmadd_pd(a_1, b_1, c_11);
    c_12 =  _mm512_fmadd_pd(a_1, b_2, c_12);
    c_13 =  _mm512_fmadd_pd(a_1, b_3, c_13);

    c_20 =  _mm512_fmadd_pd(a_2, b_0, c_20);
    b_0 = _mm512_set1_pd(B(p , 4 ));
    c_21 =  _mm512_fmadd_pd(a_2, b_1, c_21);
    b_1 = _mm512_set1_pd(B(p , 5 ));
    c_22 =  _mm512_fmadd_pd(a_2, b_2, c_22);
    b_2 = _mm512_set1_pd(B(p , 6 ));
    c_23 =  _mm512_fmadd_pd(a_2, b_3, c_23);
    b_3 = _mm512_set1_pd(B(p , 7 ));

    c_04 = _mm512_fmadd_pd(a_0, b_0, c_04);
    c_05 = _mm512_fmadd_pd(a_0, b_1, c_05);
    c_06 = _mm512_fmadd_pd(a_0, b_2, c_06);
    c_07 = _mm512_fmadd_pd(a_0, b_3, c_07);
    c_14 = _mm512_fmadd_pd(a_1, b_0, c_14);
    c_15 = _mm512_fmadd_pd(a_1, b_1, c_15);
    c_16 = _mm512_fmadd_pd(a_1, b_2, c_16);
    c_17 = _mm512_fmadd_pd(a_1, b_3, c_17);
    c_24 = _mm512_fmadd_pd(a_2, b_0, c_24);
    c_25 = _mm512_fmadd_pd(a_2, b_1, c_25);
    c_26 = _mm512_fmadd_pd(a_2, b_2, c_26);
    c_27 = _mm512_fmadd_pd(a_2, b_3, c_27);
  }
  _mm512_store_pd(&C( 0, 0 ), c_00);
  _mm512_store_pd(&C( 0, 1 ), c_01);
  _mm512_store_pd(&C( 0, 2 ), c_02);
  _mm512_store_pd(&C( 0, 3 ), c_03);
  _mm512_store_pd(&C( 0, 4 ), c_04);
  _mm512_store_pd(&C( 0, 5 ), c_05);
  _mm512_store_pd(&C( 0, 6 ), c_06);
  _mm512_store_pd(&C( 0, 7 ), c_07);
  _mm512_store_pd(&C( 8, 0 ), c_10);
  _mm512_store_pd(&C( 8, 1 ), c_11);
  _mm512_store_pd(&C( 8, 2 ), c_12);
  _mm512_store_pd(&C( 8, 3 ), c_13);
  _mm512_store_pd(&C( 8, 4 ), c_14);
  _mm512_store_pd(&C( 8, 5 ), c_15);
  _mm512_store_pd(&C( 8, 6 ), c_16);
  _mm512_store_pd(&C( 8, 7 ), c_17);
  _mm512_store_pd(&C( 16, 0 ), c_20);
  _mm512_store_pd(&C( 16, 1 ), c_21);
  _mm512_store_pd(&C( 16, 2 ), c_22);
  _mm512_store_pd(&C( 16, 3 ), c_23);
  _mm512_store_pd(&C( 16, 4 ), c_24);
  _mm512_store_pd(&C( 16, 5 ), c_25);
  _mm512_store_pd(&C( 16, 6 ), c_26);
  _mm512_store_pd(&C( 16, 7 ), c_27);
}
