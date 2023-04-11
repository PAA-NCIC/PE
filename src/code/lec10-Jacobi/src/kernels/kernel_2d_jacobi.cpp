#include"../../include/pe_jacobi.hpp"

void pe_jacobi2d(double *y, double *x, int64_t jmax, 
		 int64_t kmax, double scale) {
	pe_jacobi2d_template<double>(x, y, jmax, kmax, scale);
}
