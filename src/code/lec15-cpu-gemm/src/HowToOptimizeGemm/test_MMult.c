#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>

#include "parameters.h"

void REF_MMult(int, int, int, double *, int, double *, int, double *, int );
void MY_MMult(int, int, int, double *, int, double *, int, double *, int );
void copy_matrix(int, int, double *, int, double *, int );
void random_matrix(int, int, double *, int);
double compare_matrices( int, int, double *, int, double *, int );


double dclock();


//align the memory address to 64bytes for futher simd optimization
const int64_t ALIGNMENT = 64;
#define ALIGN(ptr) (double *)(((int64_t)(ptr) +ALIGNMENT) & (~(ALIGNMENT - 1)))
//#define ALIGN(ptr) ptr

int main()
{
  int 
    p, 
    m, n, k,
    lda, ldb, ldc, 
    rep;

  double
    dtime, dtime_best,        
    gflops, 
    diff;

  double 
    *a, *b, *c, *cref, *cold;    
  
  printf( "MY_MMult = [\n" );
    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( double * ) malloc( ALIGNMENT + lda * (k+1) * sizeof( double ) );  
    b = ( double * ) malloc( ALIGNMENT + ldb * n * sizeof( double ) );
    c = ( double * ) malloc( ALIGNMENT + ldc * n * sizeof( double ) );
    cold = ( double * ) malloc( ALIGNMENT + ldc * n * sizeof( double ) );
    cref = ( double * ) malloc( ALIGNMENT + ldc * n * sizeof( double ) );
    //printf("%p, %p, %p, %p, %p\n", a, b, c, cold, cref);
    //printf("%p, %p, %p, %p, %p\n", ALIGN(a), ALIGN(b), ALIGN(c), ALIGN(cold), ALIGN(cref));
    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, ALIGN(a), lda );
    random_matrix( k, n, ALIGN(b), ldb );
    random_matrix( m, n, ALIGN(cold), ldc );

    copy_matrix( m, n, ALIGN(cold), ldc, ALIGN(cref), ldc );

    /* Run the reference implementation so the answers can be compared */

    REF_MMult( m, n, k, ALIGN(a), lda, ALIGN(b), ldb, ALIGN(cref), ldc );

    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, n, ALIGN(cold), ldc, ALIGN(c), ldc );

      // Time your implementation 
      dtime = dclock();

      MY_MMult( m, n, k, ALIGN(a), lda, ALIGN(b), ldb, ALIGN(c), ldc );
      
      dtime = dclock() - dtime;

      if ( rep==0 )
	      dtime_best = dtime;
      else
	      dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }

    diff = compare_matrices( m, n, ALIGN(c), ldc, ALIGN(cref), ldc );

    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( cold );
    free( cref );
  }

  printf( "];\n" );

  exit( 0 );
}

