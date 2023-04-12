#include"../../include/pe_jacobi.hpp"

void pe_jacobi3d(double *y, double *x, int64_t imax, int64_t jmax, 
		 int64_t kmax, double scale) {
	pe_jacobi3d_template<double>(x, y, imax, jmax, kmax, scale);
}
