#include "genresult.cuh"
#include <sys/time.h>

#define WARP_SIZE 32

// One iteration of segment scan
__device__ void segment_scan(const int lane, const int * rows, float * vals)
{

	if (lane >= 1 && rows[threadIdx.x] == rows[threadIdx.x - 1])
		vals[threadIdx.x] += vals[threadIdx.x - 1];

	if (lane >= 2 && rows[threadIdx.x] == rows[threadIdx.x - 2])
		vals[threadIdx.x] += vals[threadIdx.x - 2];

	if (lane >= 4 && rows[threadIdx.x] == rows[threadIdx.x - 4])
		vals[threadIdx.x] += vals[threadIdx.x - 4];

	if (lane >= 8 && rows[threadIdx.x] == rows[threadIdx.x - 8])
		vals[threadIdx.x] += vals[threadIdx.x - 8];

	if (lane >= 16 && rows[threadIdx.x] == rows[threadIdx.x - 16])
		vals[threadIdx.x] += vals[threadIdx.x - 16];
}

__global__ void spmv_segmented_scan(
	const int nnz,
	const int * coord_row,
	const int * coord_col,
	const float * A,
	const float * x,
		  float * y)
{

	extern __shared__ int a[];

	int * s_rows = a;
	float * s_vals = (float *)(a + blockDim.x);

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = nnz % thread_num ? nnz / thread_num + 1 : nnz / thread_num;
    int warp_iter = blockDim.x / WARP_SIZE ? blockDim.x / WARP_SIZE + 1 : blockDim.x / WARP_SIZE;

    for (int i = 0; i < iter; i++)
    {
    	// Assign data into shared memory arrays
        int dataid = thread_id + i * thread_num;
        if (dataid < nnz)
        {
        	s_rows[threadIdx.x] = coord_row[dataid];
        	s_vals[threadIdx.x] = A[dataid] * x[coord_col[dataid]];

        	__syncthreads();

        	for (int i = 0; i < warp_iter; i++)
        	{

        		if (threadIdx.x / WARP_SIZE == i)
        		{
        			int lane = threadIdx.x % WARP_SIZE;
        			segment_scan(lane, s_rows, s_vals);

        			if (threadIdx.x != blockDim.x - 1 &&
        				lane == WARP_SIZE - 1 &&
        				s_rows[threadIdx.x] == s_rows[threadIdx.x + 1])
        			{
        				int val = s_vals[threadIdx.x + 1] + s_vals[threadIdx.x];
        				atomicAdd(&s_vals[threadIdx.x + 1], val);
        			}
        		}
        	}

        	if (threadIdx.x == blockDim.x - 1)
        		atomicAdd(&y[s_rows[threadIdx.x]], s_vals[threadIdx.x]);

        	if (threadIdx.x != blockDim.x -1 && s_rows[threadIdx.x] != s_rows[threadIdx.x + 1])
        		atomicAdd(&y[s_rows[threadIdx.x]], s_vals[threadIdx.x]);
        }
    }
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum)
{
    int nnz = mat->nz;
    int vdim = mat->M;

    // Allocate device memory
	int * d_coord_row;
	int * d_coord_col;
	float * d_A;
	float * d_x;
	float * d_y;

	cudaMalloc((void **)&d_coord_row, nnz * sizeof(int));
	cudaMalloc((void **)&d_coord_col, nnz * sizeof(int));
	cudaMalloc((void **)&d_A, nnz * sizeof(float));
	cudaMalloc((void **)&d_x, vdim * sizeof(float));
	cudaMalloc((void **)&d_y, vdim * sizeof(float));

	// Copy memory from host to device
    cudaMemcpy(d_coord_row, mat->rIndex, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coord_col, mat->cIndex, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, mat->val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, vec->val, vdim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, res->val, vdim * sizeof(float), cudaMemcpyHostToDevice);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Invoke kernel
    int smem_size = (blockSize * sizeof(float)) + (blockSize * sizeof(int));
    spmv_segmented_scan<<<blockNum, blockSize, smem_size>>>(nnz, d_coord_row, d_coord_col, d_A, d_x, d_y);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    // Get return value
    cudaMemcpy(res->val, d_y, vdim * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_coord_row);
    cudaFree(d_coord_col);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
}
