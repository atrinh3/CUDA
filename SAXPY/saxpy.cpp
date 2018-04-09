// This function generates random data and stores them into an array.
void getData(float* input, int input_size)
{
    srand(100)
    for(i = 0, i < input_size; i++){
        input[i] = rand();
    }
}


// SAXPY kernel - Multiplies an array "x" with a scalar constant "a"
// and performs element-wise addition with another array "y".
__global__
void saxpy(int n, float a, float* x, float* y, float* z){
    // Get a global ID number for the current kernel run.  Equivalent
    // to the i variable in a for loop.  Obtained by taking the
    // "width" or "blockIdx.x" and multiplying it by it's "height"
    // or "blockDim.x". Then add "threadIdx.x" which obtains the
    // location of the current kernel of the global data.
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This upcoming if loop is a little redundant.  But, it would
    // help protect from someone purposefully trying to break the
    // program.
    if(id < n){
        z = a * x[id] + y[id];
    }
}


int main(void)
{
    // Initialize variables and variable names to distinguish
    // memory allocations for both host and device side.
    int N = 10;
    float a = 2.0f
    float *x, *y, *z, *d_x, *d_y *d_z;    // initiate variables
    
    // Allocate host-side memory.
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    z = (float*)malloc(N*sizeof(float));

    // Allocate device-side memory.    
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(float));
    
    // Generate input data to perform SAXPY with.
    x = getData(&x, N);
    y = getData(&y, N);

    // Copy input data into device memory to allow kernel run.
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel execution.
    saxpy<<<(N + 255 / 256, 256)>>>(N, a, d_x, d_y, d_z);
    
    // Copy the output data (z) back to the host device for
    // additional operations.
    cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    // Once the results are copied back onto the host device and the
    // kernel runs have all finished, all the allocated memory on
    // the host and device must be released.
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);
}
