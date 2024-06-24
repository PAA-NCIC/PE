#include "../include/csr_formatter.h"

void printMatrix(CSR csr){
  int cont = 0;
  for(int i = 1; i < csr.row_ptr.size(); i++){
    int row_start = csr.row_ptr[i-1] - 1;
    int row_end = csr.row_ptr[i] - 1;
    vector<int>::const_iterator first = csr.col_idx.begin() + row_start;
    vector<int>::const_iterator last = csr.col_idx.begin() + row_end;  
    vector<int> row(first, last);    
    for(int j = 1; j < csr.row_ptr.size(); j++){
      if(std::count(row.begin(), row.end(), j) == 0)
        cout << '0' << ' ';
      else{
        cout << csr.val[cont] << ' ';
        cont++;
      }
    }
    std::cout << std::endl;
  }
}

//Sparse matrix bandwidth refers to a measure of how widely spread the non-zero 
//elements of a sparse matrix are along its main diagonal.
int getBandwidth(CSR csr){
  int bandwidth = std::numeric_limits<int>::min();
  for(int i = 1; i < csr.row_ptr.size() - 1; i++){ // i = current row id
    int row_start = csr.row_ptr[i-1];
    int row_end = csr.row_ptr[i];
    if (row_end - row_start == 1)
      continue;
    for (int j = row_start; j < row_end;j++){
      if (abs(csr.col_idx[j] - i) > bandwidth){
        bandwidth = abs(csr.col_idx[j] - i);
      }
        
    }
  }
  return bandwidth;
}

CSR assemble_csr_matrix(std::string filePath){
  int M, N, L;
  CSR matrix;
  std::ifstream fin(filePath);
  // Ignore headers and comments:
  while (fin.peek() == '%') fin.ignore(2048, '\n');
  // Read defining parameters:
  fin >> M >> N >> L;
  matrix.rows = M;
  matrix.cols = N;
  matrix.nnz  = 0;
  int last_row = 1;
  matrix.row_ptr.push_back(1);
  for (int l = 0; l < L; l++){
    int row, col;
    double data;
    fin >> row >> col >> data;
    matrix.col_idx.push_back(col);
    matrix.val.push_back(data);
    matrix.nnz++;
    if (row > last_row){
      last_row = row;
      matrix.row_ptr.push_back(matrix.col_idx.size());
    }  
  }
  //matrix.row_ptr.push_back(matrix.col_idx.size() + 1);
  fin.close();
  return matrix;
}

CSR assemble_symmetric_csr_matrix(std::string filePath){
  int M, N, L;
  vector<int> rows, cols;
  vector<double> data;
  CSR matrix;
  std::ifstream fin(filePath);
  // Ignore headers and comments:
  while (fin.peek() == '%') fin.ignore(2048, '\n');
  // Read defining parameters:
  fin >> M >> N >> L;  
  matrix.rows = M;
  matrix.cols = N;
  matrix.nnz  = 0;
  matrix.row_ptr.push_back(0);
  for (int l = 0; l < L; l++){
    int row, col;
    double d;
    fin >> row >> col >> d;
    rows.push_back(row);
    cols.push_back(col);
    data.push_back(d);
  }
  fin.close();
  for (int l = 1; l <= M; l++){
    for (int k = 0; k < L; k++){
      if (cols[k] == l){
        matrix.col_idx.push_back(rows[k]);
        matrix.val.push_back(data[k]);          
        matrix.nnz++;
      }  
      else if (rows[k] == l){
        matrix.col_idx.push_back(cols[k]);
        matrix.val.push_back(data[k]);        
        matrix.nnz++;
      }
    }
    matrix.row_ptr.push_back(matrix.col_idx.size());
  }
  
  //matrix.row_ptr.push_back(matrix.col_idx.size() + 1);
  
  return matrix;
}
