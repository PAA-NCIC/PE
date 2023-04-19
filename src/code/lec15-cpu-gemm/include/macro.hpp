#ifndef _MACRO_HPP_
#define _MACRO_HPP_

//matrix element visit
#define COL_MAJOR(A, row, ldI, col, ldJ) A[row + col * ldJ] 
#define ROW_MAJOR(A, row, ldI, col, ldJ) A[col + row * ldI] 

#endif
