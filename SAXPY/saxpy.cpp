std::vector<std::int> getData(filename)
{
    index = 0;   
    return output;
}

int main(void)
{
    int N = 10;
    float *x, *y, *d_x, *d_y;             // initiate variables
    x = (float*)malloc(N*sizeof(float));  // Allocate host memory for x
    y = (float*)malloc(N*sizeof(float));  // Allocate host memory for y
    
    cudaMalloc(&d_x, N*sizeof(float));    // Allocate device memory for x
    cudaMalloc(&d_y, N*sizeof(float));    // Allocate device memory for y
    
    x = getData(filename);
    y = getData(filename);
    
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    
    saxpy<<<(N + 255 / 256, 256)>>>(N, 2.0, d_x, d_y);
    
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
