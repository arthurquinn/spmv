#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(
    const int nnz,
    const int * coord_row,
    const int * coord_col,
    const float * A,
    const float * x,
          float * y)
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = nnz % thread_num ? nnz / thread_num + 1 : nnz / thread_num;

    for (int i = 0; i < iter; i++)
    {
        int dataid = thread_id + i * thread_num;
        if (dataid < nnz)
        {
            float data = A[dataid];
            int row = coord_row[dataid];
            int col = coord_col[dataid];
            float temp = data * x[col];
            atomicAdd(&y[row], temp);
        }
    }
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    
    int nnz = mat->nz;
    int vdim = mat->M;

    // Allocate device memory
    int * d_coord_row;
    int * d_coord_col;
    float * d_A;
    float * d_x;
    float * d_y;

    cudaMalloc(&d_coord_row, nnz * sizeof(int));
    cudaMalloc(&d_coord_col, nnz * sizeof(int));
    cudaMalloc(&d_A, nnz * sizeof(float));
    cudaMalloc(&d_x, vdim * sizeof(float));
    cudaMalloc(&d_y, vdim * sizeof(float));

    // Copy memory from device to host
    cudaMemcpy(d_coord_row, mat->rIndex, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord_col, mat->cIndex, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, mat->val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, vec->val, vdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, res->val, vdim * sizeof(float), cudaMemcpyHostToDevice);

    // Timing code
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Invoke kernel
    getMulAtomic_kernel<<<blockNum, blockSize>>>(nnz, d_coord_row, d_coord_col, d_A, d_x, d_y);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    // Retrieve result vector
    cudaMemcpy(res->val, d_y, vdim * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory
    cudaFree(d_coord_row);
    cudaFree(d_coord_col);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}
