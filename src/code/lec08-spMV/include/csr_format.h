#include<iostream>

using namespace std;

struct CSR {
	double* val;
	int* col_ind;
	int* row_ptr;
};
