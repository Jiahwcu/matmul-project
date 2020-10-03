const char* dgemm_desc = "My awesome dgemm loop";


void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (j = 0; j < M; ++j) {
        for (k = 0; k < M; ++k) {
	    double s = B[j*M+k];
            for (i = 0; i < M; ++i) {
		double cij = C[j*M+i];
		cij += A[k*M+i] * s;
		C[j*M+i] = cij;
            }
        }
    }
}
