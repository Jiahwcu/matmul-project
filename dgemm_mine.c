const char* dgemm_desc = "My awesome dgemm loop with blocks.";

// #include <nmmintrin.h>


// #define MM 8
// #define NN 8
// #define PP 8

// int DIM_M = MM;
// int DIM_N = NN;
// int DIM_K = PP;

// //BLOCK_SIZE need to have factor 8 !!!!
// #ifndef SMALL_BLOCK_SIZE
// #define SMALL_BLOCK_SIZE ((int) 8)
// #endif

// #ifndef BLOCK_SIZE
// #define BLOCK_SIZE ((int) 128)
// #endif

// /*
//  * On the Nehalem architecture, shufpd and multiplication use the same port.
//  * 32-bit integer shuffle is a different matter.  If we want to try to make
//  * it as easy as possible for the compiler to schedule multiplies along
//  * with adds, it therefore makes sense to abuse the integer shuffle
//  * instruction.  See also
//  *   http://locklessinc.com/articles/interval_arithmetic/
//  */
// #define USE_SHUFPD
// #ifdef USE_SHUFPD
// #  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
// #else
// #  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
// #endif


// /*
//  * Block matrix multiply kernel.
//  * Inputs:
//  *    A: 2-by-P matrix in column major format.
//  *    B: P-by-2 matrix in row major format.
//  * Outputs:
//  *    C: 2-by-2 matrix with element order [c11, c22, c12, c21]
//  *       (diagonals stored first, then off-diagonals)
//  */
// void kdgemm2P2(double * restrict C,
//                const double * restrict A,
//                const double * restrict B)
// {
//     // This is really implicit in using the aligned ops...
//     __builtin_assume_aligned(A, 16);
//     __builtin_assume_aligned(B, 16);
//     __builtin_assume_aligned(C, 16);

//     // Load diagonal and off-diagonals
//     __m128d cd = _mm_load_pd(C+0);
//     __m128d co = _mm_load_pd(C+2);

//     /*
//      * Do block dot product.  Each iteration adds the result of a two-by-two
//      * matrix multiply into the accumulated 2-by-2 product matrix, which is
//      * stored in the registers cd (diagonal part) and co (off-diagonal part).
//      */
//     for (int k = 0; k < PP; k += 2) {

//         __m128d a0 = _mm_load_pd(A+2*k+0);
//         __m128d b0 = _mm_load_pd(B+2*k+0);
//         __m128d td0 = _mm_mul_pd(a0, b0);
//         __m128d bs0 = swap_sse_doubles(b0);
//         __m128d to0 = _mm_mul_pd(a0, bs0);

//         __m128d a1 = _mm_load_pd(A+2*k+2);
//         __m128d b1 = _mm_load_pd(B+2*k+2);
//         __m128d td1 = _mm_mul_pd(a1, b1);
//         __m128d bs1 = swap_sse_doubles(b1);
//         __m128d to1 = _mm_mul_pd(a1, bs1);

//         __m128d td_sum = _mm_add_pd(td0, td1);
//         __m128d to_sum = _mm_add_pd(to0, to1);

//         cd = _mm_add_pd(cd, td_sum);
//         co = _mm_add_pd(co, to_sum);
//     }

//     // Write back sum
//     _mm_store_pd(C+0, cd);
//     _mm_store_pd(C+2, co);
// }


// /*
//  * Block matrix multiply kernel.
//  * Inputs:
//  *    A: 4-by-P matrix in column major format.
//  *    B: P-by-4 matrix in row major format.
//  * Outputs:
//  *    C: 4-by-4 matrix with element order 
//  *       [c11, c22, c12, c21,   c31, c42, c32, c41,
//  *        c13, c24, c14, c23,   c33, c44, c34, c43]
//  *       That is, C is broken into 2-by-2 sub-blocks, and is stored
//  *       in column-major order at the block level and diagonal/off-diagonal
//  *       within blocks.
//  */
// void kdgemm4P4(double * restrict C,
//                const double * restrict A,
//                const double * restrict B)
// {
//     __builtin_assume_aligned(A, 16);
//     __builtin_assume_aligned(B, 16);
//     __builtin_assume_aligned(C, 16);

//     kdgemm2P2(C,    A+0,   B+0);
//     kdgemm2P2(C+4,  A+2*PP, B+0);
//     kdgemm2P2(C+8,  A+0,   B+2*PP);
//     kdgemm2P2(C+12, A+2*PP, B+2*PP);
// }

// /*
//  * Block matrix multiply kernel.
//  * Inputs:
//  *    A: 8-by-P matrix in column major format.
//  *    B: P-by-8 matrix in row major format.
//  * Outputs:
//  *    C: 8-by-8 matrix viewed as a 2-by-2 block matrix.  Each block has
//  *       the layout from kdgemm4P4.
//  */
// void kdgemm8P8(double * restrict C,
//                const double * restrict A,
//                const double * restrict B)
// {
//     __builtin_assume_aligned(A, 16);
//     __builtin_assume_aligned(B, 16);
//     __builtin_assume_aligned(C, 16);

//     kdgemm4P4(C,    A+0,   B+0);
//     kdgemm4P4(C+16, A+4*PP, B+0);
//     kdgemm4P4(C+32, A+0,   B+4*PP);
//     kdgemm4P4(C+48, A+4*PP, B+4*PP);
// }

// void kdgemm(const double * restrict A,
//             const double * restrict B,
//             double * restrict C)
// {
//     kdgemm8P8(C, A, B);
// }


/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    int i, j, k;
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            double s = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                double cij = C[j*lda+i];
                cij += A[k*lda+i] * s;
                C[j*lda+i] = cij;
            }
        }
    }
}

//register blocking

void dgemm_small(const double * restrict A, const double * restrict B, double * restrict C){
    const int M = SMALL_BLOCK_SIZE;
    const int lda = BLOCK_SIZE;
    int i,j,k;
    for (j = 0; j < M; ++j) {
        for (k = 0; k < M; ++k) {
            double s = B[j*lda+k];
            for (i = 0; i < M; ++i) {
                double cij = C[j*lda+i];
                cij += A[k*lda+i] * s;
                C[j*lda+i] = cij;
            }
        }
    }
}

void basic_dgemm_square(const double * restrict A, const double * restrict B, double * restrict C)
{
    const int M = BLOCK_SIZE;
    const int n_blocks = BLOCK_SIZE/SMALL_BLOCK_SIZE;
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * SMALL_BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * SMALL_BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * SMALL_BLOCK_SIZE;
                //basic_dgemm(M, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE, A + i + k*M, B + k + j*M, C + i + j*M);
                dgemm_small(A+i+k*M, B+k+j*M, C+i+j*M);
                //kdgemm(A+i+k*M, B+k+j*M, C+i+j*M);
            }
        }
    }
}

void do_copy_square_in(const int lda, const double * restrict A,  double * restrict AA){
    int i, j;
    const int M = BLOCK_SIZE;
    for(j = 0; j < M; ++j){
        for(i = 0; i < M; ++i){
            AA[j*M+i] = A[j*lda+i];
        }
    }
}

void do_copy_square_out(const int lda, double * restrict A, const double *restrict  AA){
    int i, j;
    const int M = BLOCK_SIZE;
    for(j = 0; j < M; ++j){
        for(i = 0; i < M; ++i){
             A[j*lda+i] = AA[j*M+i];
        }
    }
}

void do_block_square(const int lda,
              const double * restrict A, const double * restrict B, double * restrict C,
              double * restrict AA, double * restrict BB, double * restrict CC,
              const int i, const int j, const int k)
{

    const int M = BLOCK_SIZE;
    do_copy_square_in(lda, A+i+k*lda, AA);
    do_copy_square_in(lda, B+k+j*lda, BB);
    do_copy_square_in(lda, C+i+j*lda, CC);
    //memset(CC, 0, sizeof(double)*M*M);

    //printf("Aij, %f\n", A[i+k*lda]);
    //printf("AA , %f\n", AA[0]);
    //printf("Bij, %f\n", B[k+j*lda]);
    //printf("BB , %f\n", BB[0]);

    basic_dgemm_square(AA, BB, CC);

    do_copy_square_out(lda, C+i+j*lda, CC);
    //printf("Cij, %f\n", C[i+j*lda]);
    //printf("CC , %f\n", CC[0]);
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE;
    int bi, bj, bk;
    
    const int leftover = M % BLOCK_SIZE ? 1:0;
    double  AA[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  BB[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));
    double  CC[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (__BIGGEST_ALIGNMENT__)));

    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block_square(M, A, B, C, AA, BB, CC, i, j, k);
                //do_block(M, A, B, C, i, j, k);
            }
            if (leftover){
                const int k = n_blocks * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
        if (leftover){
            const int j = n_blocks * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks + leftover; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
    if (leftover){
        const int i = n_blocks * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks + leftover; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks + leftover; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
