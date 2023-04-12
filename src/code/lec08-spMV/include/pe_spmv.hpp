#ifndef _PE_SPMV_HPP_
#define _PE_SPMV_HPP_
#include<cstdint>
#include"macro.hpp"

template <typename T>
void pe_spmv_csr(int *row_ptr, int *colind, T *val, int m, T *x, T *y)
{
  int i, j;
  T temp;
  //loop over rows
  for(i = 0; i < m ; i++)
  {
    //dot product 
    temp = y[i];
    for(j = row_ptr[i]; j < row_ptr[i+1]; j++){
      temp += val[j] * x[colind[j]];
    }
    y[i] = temp;
  }
}

#endif
