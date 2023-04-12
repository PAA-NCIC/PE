#ifndef _MACRO_HPP_
#define _MACRO_HPP_

//matrix element visit
#define ARRAY_2D(A, d1, d2, D1, D2) A[(d1) * D2 + (d2)] 
#define ARRAY_3D(A, d1, d2, d3, D1, D2, D3) A[(d1) * D2 * D3 + (d2) * D3 + (d3)] 

#endif
