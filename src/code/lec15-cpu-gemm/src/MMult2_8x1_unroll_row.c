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
    for ( i=0; i<m; i+=8 ){        /* Loop over the columns of C, unrolled by 4 */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot8x1( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}


void AddDot8x1( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes four elements of C: 

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i, j ), C( i, j+1 ), C( i, j+2 ), C( i, j+3 ) 
	  
     in the original matrix C.

     In this version, we merge the four loops, computing four inner
     products simultaneously. */
  double c0 = C(0,0), c1 = C(1,0), 
         c2 = C(2,0), c3 = C(3,0), 
         c4 = C(4,0), c5 = C(5,0),  
         c6 = C(6,0), c7 = C(7,0);

  asm volatile ("# simd begin");
  for (int p=0; p<k; p++ ){
    c0 += A( 0, p ) * B( p, 0 );     
    c1 += A( 1, p ) * B( p, 0 );     
    c2 += A( 2, p ) * B( p, 0 );     
    c3 += A( 3, p ) * B( p, 0 );     
    c4 += A( 4, p ) * B( p, 0 );     
    c5 += A( 5, p ) * B( p, 0 );     
    c6 += A( 6, p ) * B( p, 0 );     
    c7 += A( 7, p ) * B( p, 0 );     
  }
  asm volatile ("# simd begin");
  C(0,0)=c0;C(1,0)=c1;C(2,0)=c2;C(3,0)=c3;
  C(4,0)=c4;C(5,0)=c5;C(6,0)=c6;C(7,0)=c7;
}
