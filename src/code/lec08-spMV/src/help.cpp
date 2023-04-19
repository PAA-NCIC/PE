#include"../include/help.hpp"
#include<fstream>


double get_time(struct timespec *start,
  struct timespec *end)
{
  return end->tv_sec - start->tv_sec +
    (end->tv_nsec - start->tv_nsec) * 1e-9;
}

CSR load_matrix(std::string filePath){
  int M, N, L;
  CSR matrix;
  std::ifstream fin(filePath);
  // Ignore headers and comments:
  while (fin.peek() == '%') fin.ignore(2048, '\n');
  // Read defining parameters:
  fin >> M >> N >> L;
  
  int last_row = 1;
  matrix.row_ptr.push_back(1);
  for (int l = 0; l < L; l++){
    int row, col;
    double data;
    fin >> row >> col >> data;
    matrix.col_ind.push_back(col);
    matrix.val.push_back(data);
    if (row > last_row){
    	last_row = row;
        matrix.row_ptr.push_back(matrix.col_ind.size());
    }	
  }
  matrix.row_ptr.push_back(matrix.col_ind.size() + 1);
  fin.close();
  return matrix;
}


