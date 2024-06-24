#ifndef CSR_FORMATTER_H__
#define CSR_FORMATTER_H__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h> 

using namespace std;

struct CSR {
  vector<double> val;
  vector<int>    col_idx;
  vector<int>    row_ptr;
  int            rows;		// m dim of sparse matrix
  int            cols;		// n dim of sparse matrix
  int            nnz;		// num of nonezeros
};

void printMatrix(CSR csr);

//Sparse matrix bandwidth refers to a measure of how widely spread the non-zero 
//elements of a sparse matrix are along its main diagonal.
int getBandwidth(CSR csr);

CSR assemble_csr_matrix(std::string filePath);

CSR assemble_symmetric_csr_matrix(std::string filePath);


#endif
