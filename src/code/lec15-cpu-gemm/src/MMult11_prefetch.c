#include <immintrin.h>
//1024K private L2
//half for C blocking, e.g., 512KB, can store
/* Create macros so that the matrices are stored in column-major order */
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define BT(i,j) b[(i)*ldb + (j)] //row major for packedB

#define min( i, j ) ( (i)<(j) ? (i): (j) )
/* Routine for computing C = A * B + C */
#define mc 192
#define kc 384
#define mr 24
#define nr 8


static inline void AddDot24x8( int, double *, int,  double *, int, double *, int );
static void InnerKernel( int m, int n, int k, double *a, int lda,  
                                       double *b, int ldb, 
                                       double *c, int ldc, int first_time);
static void PackMatrixA( int, double *, int, double * );
static void PackMatrixB( int, double *, int, double * );


void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc)
{
  int i, j, p, pb, ib;

  for(p = 0; p < k; p += kc){
    pb = min(k-p, kc);
    for ( i=0; i<m; i+=mc ){       
      ib = min(m-i, mc);
      InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0);
    }
  }
}


void InnerKernel( int m, int n, int k, double *a, int lda,  
                                       double *b, int ldb, 
                                       double *c, int ldc,
                                       int first_time) {
  double __attribute__((aligned(64))) packedA[m * k];
  static double __attribute__((aligned(64))) packedB[kc * 1300];
  int i, j;
  for(j = 0; j < n; j+=nr) {
    if ( first_time )
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
      //PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
	  for(i = 0; i < m; i+=mr) {
      if(j == 0) 
        PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      //AddDot24x8(k, &A(i, 0), lda,  &B(0,j), ldb, &C(i,j), ldc);
      AddDot24x8(k, &packedA[i * k], mr, &packedB[ j*k ], nr, &C(i,j), ldc);
    }
  }
}

#define LOAD_C  \
  __m512d c_00 = _mm512_load_pd(&C( 0*8, 0 ));  \
  __m512d c_01 = _mm512_load_pd(&C( 0*8, 1 ));  \
  __m512d c_02 = _mm512_load_pd(&C( 0*8, 2 ));  \
  __m512d c_03 = _mm512_load_pd(&C( 0*8, 3 ));  \
  __m512d c_04 = _mm512_load_pd(&C( 0*8, 4 ));  \
  __m512d c_05 = _mm512_load_pd(&C( 0*8, 5 ));  \
  __m512d c_06 = _mm512_load_pd(&C( 0*8, 6 ));  \
  __m512d c_07 = _mm512_load_pd(&C( 0*8, 7 ));  \
  __m512d c_10 = _mm512_load_pd(&C( 1*8, 0 ));  \
  __m512d c_11 = _mm512_load_pd(&C( 1*8, 1 ));  \
  __m512d c_12 = _mm512_load_pd(&C( 1*8, 2 ));  \
  __m512d c_13 = _mm512_load_pd(&C( 1*8, 3 ));  \
  __m512d c_14 = _mm512_load_pd(&C( 1*8, 4 ));  \
  __m512d c_15 = _mm512_load_pd(&C( 1*8, 5 ));  \
  __m512d c_16 = _mm512_load_pd(&C( 1*8, 6 ));  \
  __m512d c_17 = _mm512_load_pd(&C( 1*8, 7 ));  \
  __m512d c_20 = _mm512_load_pd(&C( 2*8, 0 ));  \
  __m512d c_21 = _mm512_load_pd(&C( 2*8, 1 ));  \
  __m512d c_22 = _mm512_load_pd(&C( 2*8, 2 ));  \
  __m512d c_23 = _mm512_load_pd(&C( 2*8, 3 ));  \
  __m512d c_24 = _mm512_load_pd(&C( 2*8, 4 ));  \
  __m512d c_25 = _mm512_load_pd(&C( 2*8, 5 ));  \
  __m512d c_26 = _mm512_load_pd(&C( 2*8, 6 ));  \
  __m512d c_27 = _mm512_load_pd(&C( 2*8, 7 ))  \
   

#define ZERO_C  \
  __m512d c_00 = _mm512_setzero_pd(); \ 
  __m512d c_01 = _mm512_setzero_pd(); \ 
  __m512d c_02 = _mm512_setzero_pd(); \ 
  __m512d c_03 = _mm512_setzero_pd(); \ 
  __m512d c_04 = _mm512_setzero_pd(); \ 
  __m512d c_05 = _mm512_setzero_pd(); \ 
  __m512d c_06 = _mm512_setzero_pd(); \ 
  __m512d c_07 = _mm512_setzero_pd(); \ 
  __m512d c_10 = _mm512_setzero_pd(); \ 
  __m512d c_11 = _mm512_setzero_pd(); \ 
  __m512d c_12 = _mm512_setzero_pd(); \ 
  __m512d c_13 = _mm512_setzero_pd(); \ 
  __m512d c_14 = _mm512_setzero_pd(); \ 
  __m512d c_15 = _mm512_setzero_pd(); \ 
  __m512d c_16 = _mm512_setzero_pd(); \ 
  __m512d c_17 = _mm512_setzero_pd(); \ 
  __m512d c_20 = _mm512_setzero_pd(); \ 
  __m512d c_21 = _mm512_setzero_pd(); \ 
  __m512d c_22 = _mm512_setzero_pd(); \ 
  __m512d c_23 = _mm512_setzero_pd(); \ 
  __m512d c_24 = _mm512_setzero_pd(); \ 
  __m512d c_25 = _mm512_setzero_pd(); \ 
  __m512d c_26 = _mm512_setzero_pd(); \ 
  __m512d c_27 = _mm512_setzero_pd() \ 

#define PREFETCH_C(i,j) _mm_prefetch((char *)&C( (i)*8, j ), 0)



 void AddDot24x8( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  ZERO_C;
  __m512d b_0, b_1;
  __m512d a_00, a_01, a_02, a_10, a_11, a_12;
  int p;
  //double *a_ptr = &()
  a_00 = _mm512_load_pd(&A( 0 * 8, 0 ));
  a_01 = _mm512_load_pd(&A( 1 * 8, 0 ));
  a_02 = _mm512_load_pd(&A( 2 * 8, 0 ));
  b_0  = _mm512_set1_pd(BT(p , 0 ));

  for (p=0; p<k-2;){ 
    //load p+1
    a_10 = _mm512_load_pd(&A( 0 * 8, p+1 ));
    a_11 = _mm512_load_pd(&A( 1 * 8, p+1 ));
    a_12 = _mm512_load_pd(&A( 2 * 8, p+1 ));
    //compute p
    b_1  = _mm512_set1_pd(BT(p , 1 ));
    c_00 = _mm512_fmadd_pd(a_00, b_0, c_00);
    c_10 = _mm512_fmadd_pd(a_01, b_0, c_10);
    c_20 = _mm512_fmadd_pd(a_02, b_0, c_20);
    b_0  = _mm512_set1_pd(BT(p , 2 ));
    c_01 = _mm512_fmadd_pd(a_00, b_1, c_01);
    c_11 = _mm512_fmadd_pd(a_01, b_1, c_11);
    c_21 = _mm512_fmadd_pd(a_02, b_1, c_21); 
    b_1  = _mm512_set1_pd(BT(p , 3 ));
    c_02 = _mm512_fmadd_pd(a_00, b_0, c_02);
    c_12 = _mm512_fmadd_pd(a_01, b_0, c_12);
    c_22 = _mm512_fmadd_pd(a_02, b_0, c_22);
    b_0  = _mm512_set1_pd(BT(p , 4 ));
    c_03 = _mm512_fmadd_pd(a_00, b_1, c_03);
    c_13 = _mm512_fmadd_pd(a_01, b_1, c_13);
    c_23 = _mm512_fmadd_pd(a_02, b_1, c_23);
    b_1  = _mm512_set1_pd(BT(p , 5 ));
    c_04 = _mm512_fmadd_pd(a_00, b_0, c_04);
    c_14 = _mm512_fmadd_pd(a_01, b_0, c_14);
    c_24 = _mm512_fmadd_pd(a_02, b_0, c_24);
    b_0  = _mm512_set1_pd(BT(p , 6 ));
    c_05 = _mm512_fmadd_pd(a_00, b_1, c_05);
    c_15 = _mm512_fmadd_pd(a_01, b_1, c_15);
    c_25 = _mm512_fmadd_pd(a_02, b_1, c_25);
    b_1  = _mm512_set1_pd(BT(p , 7 ));
    c_06 = _mm512_fmadd_pd(a_00, b_0, c_06);
    c_16 = _mm512_fmadd_pd(a_01, b_0, c_16);
    c_26 = _mm512_fmadd_pd(a_02, b_0, c_26);
    b_0  = _mm512_set1_pd(BT(++p , 0 ));
    c_07 = _mm512_fmadd_pd(a_00, b_1, c_07);
    c_17 = _mm512_fmadd_pd(a_01, b_1, c_17);
    c_27 = _mm512_fmadd_pd(a_02, b_1, c_27);
    //load p+2
    a_00 = _mm512_load_pd(&A( 0 * 8, p+1 ));
    a_01 = _mm512_load_pd(&A( 1 * 8, p+1 ));
    a_02 = _mm512_load_pd(&A( 2 * 8, p+1 ));
    b_1  = _mm512_set1_pd(BT(p , 1 ));
    c_00 = _mm512_fmadd_pd(a_10, b_0, c_00);
    c_10 = _mm512_fmadd_pd(a_11, b_0, c_10);
    c_20 = _mm512_fmadd_pd(a_12, b_0, c_20);
    b_0  = _mm512_set1_pd(BT(p , 2 ));
    c_01 = _mm512_fmadd_pd(a_10, b_1, c_01);
    c_11 = _mm512_fmadd_pd(a_11, b_1, c_11);
    c_21 = _mm512_fmadd_pd(a_12, b_1, c_21);
    b_1  = _mm512_set1_pd(BT(p , 3 ));
    c_02 = _mm512_fmadd_pd(a_10, b_0, c_02);
    c_12 = _mm512_fmadd_pd(a_11, b_0, c_12);
    c_22 = _mm512_fmadd_pd(a_12, b_0, c_22);
    b_0  = _mm512_set1_pd(BT(p , 4 ));
    c_03 = _mm512_fmadd_pd(a_10, b_1, c_03);
    c_13 = _mm512_fmadd_pd(a_11, b_1, c_13);
    c_23 = _mm512_fmadd_pd(a_12, b_1, c_23);
    b_1  = _mm512_set1_pd(BT(p , 5 ));
    c_04 = _mm512_fmadd_pd(a_10, b_0, c_04);
    c_14 = _mm512_fmadd_pd(a_11, b_0, c_14);
    c_24 = _mm512_fmadd_pd(a_12, b_0, c_24);
    b_0  = _mm512_set1_pd(BT(p , 6 ));
    c_05 = _mm512_fmadd_pd(a_10, b_1, c_05);
    c_15 = _mm512_fmadd_pd(a_11, b_1, c_15);
    c_25 = _mm512_fmadd_pd(a_12, b_1, c_25);
    b_1  = _mm512_set1_pd(BT(p , 7 ));
    c_06 = _mm512_fmadd_pd(a_10, b_0, c_06);
    c_16 = _mm512_fmadd_pd(a_11, b_0, c_16);
    c_26 = _mm512_fmadd_pd(a_12, b_0, c_26);
    b_0  = _mm512_set1_pd(BT(++p , 0 ));
    c_07 = _mm512_fmadd_pd(a_10, b_1, c_07);
    c_17 = _mm512_fmadd_pd(a_11, b_1, c_17);
    c_27 = _mm512_fmadd_pd(a_12, b_1, c_27);
  }
  a_10 = _mm512_load_pd(&A( 0 * 8, p+1 ));
  a_11 = _mm512_load_pd(&A( 1 * 8, p+1 ));
  a_12 = _mm512_load_pd(&A( 2 * 8, p+1 ));
  b_1  = _mm512_set1_pd(BT(p , 1 ));
  //_mm_prefetch((char *)p, int i)
  c_00 = _mm512_fmadd_pd(a_00, b_0, c_00);
  c_10 = _mm512_fmadd_pd(a_01, b_0, c_10);
  c_20 = _mm512_fmadd_pd(a_02, b_0, c_20);
  PREFETCH_C(0,0);
  PREFETCH_C(1,0);
  PREFETCH_C(2,0);
  b_0  = _mm512_set1_pd(BT(p , 2 ));
  c_01 = _mm512_fmadd_pd(a_00, b_1, c_01);
  c_11 = _mm512_fmadd_pd(a_01, b_1, c_11);
  c_21 = _mm512_fmadd_pd(a_02, b_1, c_21); 
  PREFETCH_C(0,1);
  PREFETCH_C(1,1);
  PREFETCH_C(2,1);
  b_1  = _mm512_set1_pd(BT(p , 3 ));
  c_02 = _mm512_fmadd_pd(a_00, b_0, c_02);
  c_12 = _mm512_fmadd_pd(a_01, b_0, c_12);
  c_22 = _mm512_fmadd_pd(a_02, b_0, c_22);
  PREFETCH_C(0,2);
  PREFETCH_C(1,2);
  PREFETCH_C(2,2);
  b_0  = _mm512_set1_pd(BT(p , 4 ));
  c_03 = _mm512_fmadd_pd(a_00, b_1, c_03);
  c_13 = _mm512_fmadd_pd(a_01, b_1, c_13);
  c_23 = _mm512_fmadd_pd(a_02, b_1, c_23);
  PREFETCH_C(0,3);
  PREFETCH_C(1,3);
  PREFETCH_C(2,3);
  b_1  = _mm512_set1_pd(BT(p , 5 ));
  c_04 = _mm512_fmadd_pd(a_00, b_0, c_04);
  c_14 = _mm512_fmadd_pd(a_01, b_0, c_14);
  c_24 = _mm512_fmadd_pd(a_02, b_0, c_24);
  PREFETCH_C(0,4);
  PREFETCH_C(1,4);
  PREFETCH_C(2,4);
  b_0  = _mm512_set1_pd(BT(p , 6 ));
  c_05 = _mm512_fmadd_pd(a_00, b_1, c_05);
  c_15 = _mm512_fmadd_pd(a_01, b_1, c_15);
  c_25 = _mm512_fmadd_pd(a_02, b_1, c_25);
  PREFETCH_C(0,5);
  PREFETCH_C(1,5);
  PREFETCH_C(2,5);
  b_1  = _mm512_set1_pd(BT(p , 7 ));
  c_06 = _mm512_fmadd_pd(a_00, b_0, c_06);
  c_16 = _mm512_fmadd_pd(a_01, b_0, c_16);
  c_26 = _mm512_fmadd_pd(a_02, b_0, c_26);
  PREFETCH_C(0,6);
  PREFETCH_C(1,6);
  PREFETCH_C(2,6);
  b_0  = _mm512_set1_pd(BT(++p , 0 ));
  c_07 = _mm512_fmadd_pd(a_00, b_1, c_07);
  c_17 = _mm512_fmadd_pd(a_01, b_1, c_17);
  c_27 = _mm512_fmadd_pd(a_02, b_1, c_27);
  PREFETCH_C(0,7);
  PREFETCH_C(1,7);
  PREFETCH_C(2,7);
  b_1  = _mm512_set1_pd(BT(p , 1 ));
  c_00 = _mm512_fmadd_pd(a_10, b_0, c_00);
  c_10 = _mm512_fmadd_pd(a_11, b_0, c_10);
  c_20 = _mm512_fmadd_pd(a_12, b_0, c_20);
  b_0  = _mm512_set1_pd(BT(p , 2 ));
  c_01 = _mm512_fmadd_pd(a_10, b_1, c_01);
  c_11 = _mm512_fmadd_pd(a_11, b_1, c_11);
  c_21 = _mm512_fmadd_pd(a_12, b_1, c_21);
  b_1  = _mm512_set1_pd(BT(p , 3 ));
  c_02 = _mm512_fmadd_pd(a_10, b_0, c_02);
  c_12 = _mm512_fmadd_pd(a_11, b_0, c_12);
  c_22 = _mm512_fmadd_pd(a_12, b_0, c_22);
  b_0  = _mm512_set1_pd(BT(p , 4 ));
  c_03 = _mm512_fmadd_pd(a_10, b_1, c_03);
  c_13 = _mm512_fmadd_pd(a_11, b_1, c_13);
  c_23 = _mm512_fmadd_pd(a_12, b_1, c_23);
  b_1  = _mm512_set1_pd(BT(p , 5 ));
  c_04 = _mm512_fmadd_pd(a_10, b_0, c_04);
  c_14 = _mm512_fmadd_pd(a_11, b_0, c_14);
  c_24 = _mm512_fmadd_pd(a_12, b_0, c_24);
  b_0  = _mm512_set1_pd(BT(p , 6 ));
  c_05 = _mm512_fmadd_pd(a_10, b_1, c_05);
  c_15 = _mm512_fmadd_pd(a_11, b_1, c_15);
  c_25 = _mm512_fmadd_pd(a_12, b_1, c_25);
  b_1  = _mm512_set1_pd(BT(p , 7 ));
  c_06 = _mm512_fmadd_pd(a_10, b_0, c_06);
  c_16 = _mm512_fmadd_pd(a_11, b_0, c_16);
  c_26 = _mm512_fmadd_pd(a_12, b_0, c_26);
  c_07 = _mm512_fmadd_pd(a_10, b_1, c_07);
  c_17 = _mm512_fmadd_pd(a_11, b_1, c_17);
  c_27 = _mm512_fmadd_pd(a_12, b_1, c_27);
  _mm512_store_pd(&C( 0, 0 ), _mm512_add_pd(c_00,_mm512_loadu_pd(&C( 0, 0 ))));
  _mm512_store_pd(&C( 0, 1 ), _mm512_add_pd(c_01,_mm512_loadu_pd(&C( 0, 1 ))));
  _mm512_store_pd(&C( 0, 2 ), _mm512_add_pd(c_02,_mm512_loadu_pd(&C( 0, 2 ))));
  _mm512_store_pd(&C( 0, 3 ), _mm512_add_pd(c_03,_mm512_loadu_pd(&C( 0, 3 ))));
  _mm512_store_pd(&C( 0, 4 ), _mm512_add_pd(c_04,_mm512_loadu_pd(&C( 0, 4 ))));
  _mm512_store_pd(&C( 0, 5 ), _mm512_add_pd(c_05,_mm512_loadu_pd(&C( 0, 5 ))));
  _mm512_store_pd(&C( 0, 6 ), _mm512_add_pd(c_06,_mm512_loadu_pd(&C( 0, 6 ))));
  _mm512_store_pd(&C( 0, 7 ), _mm512_add_pd(c_07,_mm512_loadu_pd(&C( 0, 7 ))));
  _mm512_store_pd(&C( 8, 0 ), _mm512_add_pd(c_10,_mm512_loadu_pd(&C( 8, 0 ))));
  _mm512_store_pd(&C( 8, 1 ), _mm512_add_pd(c_11,_mm512_loadu_pd(&C( 8, 1 ))));
  _mm512_store_pd(&C( 8, 2 ), _mm512_add_pd(c_12,_mm512_loadu_pd(&C( 8, 2 ))));
  _mm512_store_pd(&C( 8, 3 ), _mm512_add_pd(c_13,_mm512_loadu_pd(&C( 8, 3 ))));
  _mm512_store_pd(&C( 8, 4 ), _mm512_add_pd(c_14,_mm512_loadu_pd(&C( 8, 4 ))));
  _mm512_store_pd(&C( 8, 5 ), _mm512_add_pd(c_15,_mm512_loadu_pd(&C( 8, 5 ))));
  _mm512_store_pd(&C( 8, 6 ), _mm512_add_pd(c_16,_mm512_loadu_pd(&C( 8, 6 ))));
  _mm512_store_pd(&C( 8, 7 ), _mm512_add_pd(c_17,_mm512_loadu_pd(&C( 8, 7 ))));
  _mm512_store_pd(&C( 16, 0 ), _mm512_add_pd(c_20,_mm512_loadu_pd(&C( 16, 0 ))));
  _mm512_store_pd(&C( 16, 1 ), _mm512_add_pd(c_21,_mm512_loadu_pd(&C( 16, 1 ))));
  _mm512_store_pd(&C( 16, 2 ), _mm512_add_pd(c_22,_mm512_loadu_pd(&C( 16, 2 ))));
  _mm512_store_pd(&C( 16, 3 ), _mm512_add_pd(c_23,_mm512_loadu_pd(&C( 16, 3 ))));
  _mm512_store_pd(&C( 16, 4 ), _mm512_add_pd(c_24,_mm512_loadu_pd(&C( 16, 4 ))));
  _mm512_store_pd(&C( 16, 5 ), _mm512_add_pd(c_25,_mm512_loadu_pd(&C( 16, 5 ))));
  _mm512_store_pd(&C( 16, 6 ), _mm512_add_pd(c_26,_mm512_loadu_pd(&C( 16, 6 ))));
  _mm512_store_pd(&C( 16, 7 ), _mm512_add_pd(c_27,_mm512_loadu_pd(&C( 16, 7 ))));
  // _mm512_store_pd(&C( 0, 0 ), c_00);
  // _mm512_store_pd(&C( 0, 1 ), c_01);
  // _mm512_store_pd(&C( 0, 2 ), c_02);
  // _mm512_store_pd(&C( 0, 3 ), c_03);
  // _mm512_store_pd(&C( 0, 4 ), c_04);
  // _mm512_store_pd(&C( 0, 5 ), c_05);
  // _mm512_store_pd(&C( 0, 6 ), c_06);
  // _mm512_store_pd(&C( 0, 7 ), c_07);
  // _mm512_store_pd(&C( 8, 0 ), c_10);
  // _mm512_store_pd(&C( 8, 1 ), c_11);
  // _mm512_store_pd(&C( 8, 2 ), c_12);
  // _mm512_store_pd(&C( 8, 3 ), c_13);
  // _mm512_store_pd(&C( 8, 4 ), c_14);
  // _mm512_store_pd(&C( 8, 5 ), c_15);
  // _mm512_store_pd(&C( 8, 6 ), c_16);
  // _mm512_store_pd(&C( 8, 7 ), c_17);
  // _mm512_store_pd(&C( 16, 0 ), c_20);
  // _mm512_store_pd(&C( 16, 1 ), c_21);
  // _mm512_store_pd(&C( 16, 2 ), c_22);
  // _mm512_store_pd(&C( 16, 3 ), c_23);
  // _mm512_store_pd(&C( 16, 4 ), c_24);
  // _mm512_store_pd(&C( 16, 5 ), c_25);
  // _mm512_store_pd(&C( 16, 6 ), c_26);
  // _mm512_store_pd(&C( 16, 7 ), c_27);
}

void PackMatrixA( int k, double *a, int lda, double *a_to) {
  int j;
  // __m512d tmp1 = _mm512_load_pd(&A( 0, 0 ));
  // __m512d tmp2 = _mm512_load_pd(&A( 0, 0 ));
  // __m512d tmp3 = _mm512_load_pd(&A( 0, 0 ));
  for( j=0; j<k; j++){  
    double *a_ptr = &A( 0, j );
    __m512d tmp1 = _mm512_load_pd(a_ptr);
    __m512d tmp2 = _mm512_load_pd(a_ptr+8);
    __m512d tmp3 = _mm512_load_pd(a_ptr+16);
    _mm512_store_pd(a_to, tmp1);
    _mm512_store_pd(a_to+8, tmp2);
    _mm512_store_pd(a_to+16, tmp3);
    a_to += mr;
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to ){
  double *bj0 = &B(0, 0), 
         *bj1 = &B(0, 1), 
         *bj2 = &B(0, 2), 
         *bj3 = &B(0, 3), 
         *bj4 = &B(0, 4), 
         *bj5 = &B(0, 5), 
         *bj6 = &B(0, 6), 
         *bj7 = &B(0, 7);
  for(int i = 0; i < k; i++){
    *b_to++ = *bj0++;
    *b_to++ = *bj1++;
    *b_to++ = *bj2++;
    *b_to++ = *bj3++;
    *b_to++ = *bj4++;
    *b_to++ = *bj5++;
    *b_to++ = *bj6++;
    *b_to++ = *bj7++;
  }
}
