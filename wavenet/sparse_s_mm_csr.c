/*******************************************************************************
* Copyright 2013-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*   Content : Intel (R) MKL IE Sparse BLAS C example for mkl_sparse_spmm
*
********************************************************************************
*
* Consider the matrix A
*
*                 |  10     11      0     0     0   |
*                 |   0      0     12    13     0   |
*   A    =        |  15      0      0     0    14   |,
*                 |   0     16     17     0     0   |
*                 |   0      0      0    18    19   |
*
* and diagonal matrix B
*
*                 |   5      0      0     0     0   |
*                 |   0      6      0     0     0   |
*   B    =        |   0      0      7     0     0   |.
*                 |   0      0      0     8     0   |
*                 |   0      0      0     0     9   |
*
*  Both matrices A and B are stored in a zero-based compressed sparse row (CSR) storage
*  scheme with three arrays (see 'Sparse Matrix Storage Schemes' in the
*  Intel (R) Math Kernel Library Developer Reference) as follows:
*
*           values_A = ( 10  11  12  13  15  14  16  17  18  19 )
*          columns_A = (  0   1   2   3   0   4   1   2   3   4 )
*         rowIndex_A = (  0       2       4       6       8      10 )
*
*           values_B = ( 5  6  7  8  9  )
*          columns_B = ( 0  1  2  3  4  )
*         rowIndex_B = ( 0  1  2  3  4  5 )
*
*  The example computes two scalar products :
*
*         < (A*B)*x ,       y > = left,   using MKL_SPARSE_SPMM and CBLAS_DDOT.
*         <     B*x , (A^t)*y > = right,  using MKL_SPARSE_D_MV and CBLAS_DDOT.
*
*         These products should result in the same value. To obtain matrix C,
*         use MKL_SPARSE_D_EXPORT_CSR and print the result.
*
******************************************************************************/

/*
clang -m64  -w -DMKL_ILP64 -I"/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/include" \
        ./sparse_s_mm_csr.c -O3\
        "/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/lib/libmkl_intel_ilp64.dylib" \
        "/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/lib/libmkl_sequential.dylib" \
        "/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/lib/libmkl_core.dylib" \
        -Wl,-rpath,/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/lib -Wl,-rpath,/opt/intel/compilers_and_libraries_2019.3.199/mac/mkl/../compiler/lib \
         -lpthread -lm -o sparse_s_mm_csr.out
*/

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "mkl.h"

int main() {

/*
#define M 128
#define K 1024
#define N 3072
#define NNZ 125385
*/
#define M 128
#define K 1024
#define N 3072
#define NNZ 125385
#define ALIGN 512
#define DENSE_ALIGN 512

/* To avoid constantly repeating the part of code that checks inbound SparseBLAS functions' status,
   use macro CALL_AND_CHECK_STATUS */
#define CALL_AND_CHECK_STATUS(function, error_message) do { \
          if(function != SPARSE_STATUS_SUCCESS)             \
          {                                                 \
          printf(error_message); fflush(0);                 \
          status = 1;                                       \
          goto memory_free;                                 \
          }                                                 \
} while(0)

/* Declaration of values */
    float  *values_A = NULL, *values_B = NULL, *values_C = NULL;
    MKL_INT *columns_A = NULL, *columns_B = NULL, *columns_C = NULL;
    MKL_INT *rowIndex_A = NULL, *rowIndex_B = NULL, *pointerB_C = NULL, *pointerE_C = NULL;

    float  *x = NULL, *y = NULL;

    float   alpha = 1.0, beta = 1.0;
    MKL_INT  rows, cols, i, j, ii, status;

    sparse_index_base_t    indexing;
    struct matrix_descr    descr_type_gen;
    sparse_matrix_t        csrA = NULL;
    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
/* Allocation of memory */
    values_A = (float *)mkl_malloc(sizeof(float) * NNZ, ALIGN);
    columns_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * NNZ, ALIGN);
    rowIndex_A = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (N + 1), ALIGN);

    x = (float *)mkl_calloc(K * M, sizeof( float ), DENSE_ALIGN);


    y = (float *)mkl_calloc(M * N, sizeof(float), DENSE_ALIGN);

/* Set values of the variables*/
    status = 0, ii = 0;
 //Matrix A
    for( i = 0; i < NNZ; i++ )
          values_A[i] = i;
    for( i = 0; i < NNZ; i++ )
          columns_A[i] = i % K;
    rowIndex_A[0] = 0;
    for( i = 1; i < N + 1; i++ ) {
      int next = rowIndex_A[i - 1] + (int)(NNZ / N + 1);
      next = next > NNZ-2 ? NNZ-2 : next;
      rowIndex_A[i] = next;
    }
    rowIndex_A[N] = NNZ;
    /*
    for (i = 0; i < N+1; i++) {
      printf("%d\n", rowIndex_A[i]);
    }
    */

 //Matrix B

/* Prepare arrays, which are related to matrices.
   Create handles for matrices A and B stored in CSR format */
    CALL_AND_CHECK_STATUS(mkl_sparse_s_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, N, K, rowIndex_A, rowIndex_A+1, columns_A, values_A ),
                          "Error after MKL_SPARSE_S_CREATE_CSR, csrA \n");
    mkl_sparse_optimize ( csrA );
/* Compute C = A * B  */
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < M * K; j++) {
        x[j] = i + j;
      }
      clock_t start = clock();
      CALL_AND_CHECK_STATUS(mkl_sparse_s_mm( SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA,
                                             descr_type_gen, SPARSE_LAYOUT_ROW_MAJOR, x,
                                             M, K, beta, y, N),
                            "Error after MKL_SPARSE_SPMM \n");
      clock_t end = clock();
      double cpu_time_used = ((double) (end - start)) * 1000 * 1000 / CLOCKS_PER_SEC;
      printf("time: %f us\n", cpu_time_used);
    }
    clock_t start = clock();
    clock_t end = clock();
    double overhead_time = ((double) (end - start)) * 1000 * 1000 / CLOCKS_PER_SEC;
    printf("overhead: %f us\n", overhead_time);

    double res = 1;
    for (int i = 0; i < M * N; i++) {
      res = res + y[i];
    }

/* Deallocate memory */
memory_free:
 //Release matrix handle. Not necessary to deallocate arrays for which we don't allocate memory: values_C, columns_C, pointerB_C, and pointerE_C.
 //These arrays will be deallocated together with csrC structure.
 //Deallocate arrays for which we allocate memory ourselves.
    mkl_free(x); mkl_free(y);

 //Release matrix handle and deallocate arrays for which we allocate memory ourselves.
    if( mkl_sparse_destroy( csrA ) != SPARSE_STATUS_SUCCESS)
    { printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(0); status = 1; }
    mkl_free(values_A); mkl_free(columns_A); mkl_free(rowIndex_A);

    printf("\n ret = %d\n", res);
    return res;
}
