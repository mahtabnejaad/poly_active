/*__device__ void warp_Reduce(volatile double *ssdata, int tid) {
    ssdata[tid] += ssdata[tid + 32];
    ssdata[tid] += ssdata[tid + 16];
    ssdata[tid] += ssdata[tid + 8];
    ssdata[tid] += ssdata[tid + 4];
    ssdata[tid] += ssdata[tid + 2];
    ssdata[tid] += ssdata[tid + 1];
}

__device__ void warp_Reduce_int(volatile int *ssdata, int tid) {
    ssdata[tid] += ssdata[tid + 32];
    ssdata[tid] += ssdata[tid + 16];
    ssdata[tid] += ssdata[tid + 8];
    ssdata[tid] += ssdata[tid + 4];
    ssdata[tid] += ssdata[tid + 2];
    ssdata[tid] += ssdata[tid + 1];
}*/

//this kernel is used to sum array components on block level in a parallel way
/*__global__ void reduce_kernel(double *FF1 ,double *FF2 , double *FF3,
 double *AA1 ,double *AA2 , double *AA3,
  int size)
{
    //size= Nmd (or N )
    //we want to add all the tangential vectors' components in one axis and calculate the total fa in one axis.
    //(OR generally we want to add all the components of a 1D array to each other) 
    int tid = threadIdx.x; //tid represents the index of the thread within the block.
    int index = blockIdx.x * blockDim.x + threadIdx.x ;//index represents the global index of the element in the input (F1,F2 or F3) array that the thread is responsible for.
    extern __shared__ double ssssdata1[];  // This declares a shared memory array sdata, which will be used for the reduction within the block
    extern __shared__ double ssssdata2[];
    extern __shared__ double ssssdata3[];
    


 
    if(index<size){
       
        // Load the value into shared memory
    //Each thread loads the corresponding element from the F1,F2 or F3 array into the shared memory array sdata. If the thread's index is greater than or equal to size, it loads a zero.
        ssssdata1[tid] = (index < size) ? FF1[index] : 0.0; 
        __syncthreads();  // Synchronize threads within the block to ensure all threads have loaded their data into shared memory before proceeding.

        //printf("iiikkk\n");
        ssssdata2[tid] = (index < size) ? FF2[index] : 0.0;
        __syncthreads();  // Synchronize threads within the block

        ssssdata3[tid] = (index < size) ? FF3[index] : 0.0;
        __syncthreads();  // Synchronize threads within the block

        //printf("hihihi\n");

        // Reduction in shared memory
        //This loop performs a binary reduction on the sdata array in shared memory.
        //The loop iteratively adds elements from sdata[tid + s] to sdata[tid], where s is halved in each iteration.
        //The threads cooperate to perform the reduction in parallel.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                ssssdata1[tid] += ssssdata1[tid + s];
                ssssdata2[tid] += ssssdata2[tid + s];
                ssssdata3[tid] += ssssdata3[tid + s];
                //printf("***");
            }
            __syncthreads();
        }
    
        // Store the block result in the result array
        //Only the first thread in the block performs this operation.
        //It stores the final reduced value of the block into A1, A2 or A3 array at the corresponding block index
        if (tid == 0)
        {
            AA1[blockIdx.x] = ssssdata1[0];
            AA2[blockIdx.x] = ssssdata2[0];
            AA3[blockIdx.x] = ssssdata3[0];
  
            //printf("A1[blockIdx.x]=%f",AA1[blockIdx.x]);
            //printf("\nA2[blockIdx.x]=%f",AA2[blockIdx.x]);
            //printf("\nA3[blockIdx.x]=%f\n",AA3[blockIdx.x]);


        }
        __syncthreads();
        //printf("BLOCKSUM1[0]=%f\n",A1[0]);
        //printf("BLOCKSUM1[1]=%f\n",A1[1]);
    }
   
}*/

//this kernel is used to sum array components on block level in a parallel way
__global__ void reduce_kernel_var(double *FF1 ,double *FF2 , double *FF3,
 double *AA1 ,double *AA2 , double *AA3,
  int size)
{
    //size= Nmd (or N )
    //we want to add all the tangential vectors' components in one axis and calculate the total fa in one axis.
    //(OR generally we want to add all the components of a 1D array to each other) 
    int tid = threadIdx.x; //tid represents the index of the thread within the block.
    int index = blockIdx.x * blockDim.x + threadIdx.x ;//index represents the global index of the element in the input (F1,F2 or F3) array that the thread is responsible for.
    const int gridSize = blockSize*gridDim.x;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;


    for (int i = index; i < size; i += gridSize){
        sum1 += FF1[i];
        sum2 += FF2[i];
        sum3 += FF3[i];
      }

    
    __shared__ double ssssdata1[blockSize];  // This declares a shared memory array sdata, which will be used for the reduction within the block
    __shared__ double ssssdata2[blockSize];
    __shared__ double ssssdata3[blockSize];
    
    ssssdata1[tid] = sum1;
    ssssdata2[tid] = sum2;
    ssssdata3[tid] = sum3;


      

        // Reduction in shared memory
        //This loop performs a binary reduction on the sdata array in shared memory.
        //The loop iteratively adds elements from sdata[tid + s] to sdata[tid], where s is halved in each iteration.
        //The threads cooperate to perform the reduction in parallel.
        for (int s = blockSize/2; s>0; s/=2)
        {
            if (tid < s)
            {
                ssssdata1[tid] += ssssdata1[tid + s];
                ssssdata2[tid] += ssssdata2[tid + s];
                ssssdata3[tid] += ssssdata3[tid + s];
                //printf("***");
            }
            __syncthreads();
        }
    
        // Store the block result in the result array
        //Only the first thread in the block performs this operation.
        //It stores the final reduced value of the block into A1, A2 or A3 array at the corresponding block index
        if (tid == 0)
        {
            AA1[blockIdx.x] = ssssdata1[0];
            AA2[blockIdx.x] = ssssdata2[0];
            AA3[blockIdx.x] = ssssdata3[0];
  
            //printf("A1[blockIdx.x]=%f",AA1[blockIdx.x]);
            //printf("\nA2[blockIdx.x]=%f",AA2[blockIdx.x]);
            //printf("\nA3[blockIdx.x]=%f\n",AA3[blockIdx.x]);


        }
        __syncthreads();
        //printf("BLOCKSUM1[0]=%f\n",A1[0]);
        //printf("BLOCKSUM1[1]=%f\n",A1[1]);
}
   


/*__global__ void reduceKernel_(double *input, double *output, int N) {
    extern __shared__ double sssdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sssdata[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sssdata[tid] += sssdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_Reduce(sssdata, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sssdata[0];
    }
}*/

__global__ void reduceKernel_var(double *input_X, double *output_X, int N){
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockSize*gridDim.x;
    double sum1 = 0.0;
    
    
    
    for (int j = i; j < N; j += gridSize)
    {
       
        sum1 += input_X[j];
        
        __syncthreads();
    }

    __shared__ double sssdata_x[blockSize];
  

    sssdata_x[tid] = sum1;
 

    for (int s = blockSize/2; s>0; s/=2)
    {
        if (tid<s)
            sssdata_x[tid] += sssdata_x[tid+s];
           

        __syncthreads();
    }

    if (tid == 0) {
        output_X[blockIdx.x] = sssdata_x[0];
       
    }
}

/*__global__ void intreduceKernel_(int *input, int *output, int N) {
    extern __shared__ int sssdata_int[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sssdata_int[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sssdata_int[tid] += sssdata_int[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_Reduce_int(sssdata_int, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sssdata_int[0];
    }
}*/

__global__ void intreduceKernel_var(int *input_X, int *output_X, int N){
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockSize*gridDim.x;
    double sum1 = 0;
    
    
    
    for (int j = i; j < N; j += gridSize)
    {
       
        sum1 += input_X[j];
       
        
    }

    __shared__ double sssdata_x[blockSize];
    __syncthreads();
    

    sssdata_x[tid] = sum1;

    for (int s = blockSize/2; s>0; s/=2)
    {
        if (tid<s)      sssdata_x[tid] += sssdata_x[tid+s];

        __syncthreads();
    }

    if (tid == 0) {
        output_X[blockIdx.x] = sssdata_x[0];
    }
}

__global__ void print_kernel(double *X, double *Y, double *Z, int N){
    
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N){
      printf("X[%i]=%f, Y[%i]=%f, Z[%i]=%f\n", tid, X[tid], tid, Y[tid], tid, Z[tid]);
    }
}


__host__ void CM_system(double *mdX, double *mdY, double *mdZ,  double *dX, double *dY, double *dZ, double *mdVx, double *mdVy, double *mdVz,  double *dVx, double *dVy, double *dVz, int Nmd, int N, 
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *dX_tot, double *dY_tot, double *dZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *dVx_tot, double *dVy_tot, double *dVz_tot, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double mass, double mass_fluid,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, 
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, int topology){
 
    if(topology == 4)
    {
        //MD particle part
        double *mdXtot, *mdYtot, *mdZtot;
        //mdXtot=0.0; mdYtot=0.0; mdZtot=0.0;
        double *mdVxtot, *mdVytot, *mdVztot;
        //mdVxtot=0.0; mdVytot=0.0; mdVztot=0.0;

        mdXtot = (double *)malloc(sizeof(double));  mdYtot = (double *)malloc(sizeof(double));  mdZtot = (double *)malloc(sizeof(double));
        mdVxtot = (double *)malloc(sizeof(double));  mdVytot = (double *)malloc(sizeof(double));  mdVztot = (double *)malloc(sizeof(double));
        
        cudaMemcpy(mdXtot, mdX, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdYtot, mdY, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdZtot, mdZ, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(mdVxtot, mdVx, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdVytot, mdVy, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdVztot, mdVz, sizeof(double), cudaMemcpyDeviceToHost);

        cudaError_t err = cudaGetLastError();        // Get error code

        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        *mdX_tot = *mdXtot;
        *mdY_tot = *mdYtot;
        *mdZ_tot = *mdZtot;

        *mdVx_tot = *mdVxtot;
        *mdVy_tot = *mdVytot;
        *mdVz_tot = *mdVztot;

        //print_kernel<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Nmd);

        ///////MPCD particles part:
       
        grid_size_=grid_size;

        double *block_sum_dX, *block_sum_dY, *block_sum_dZ, *block_sum_dVx, *block_sum_dVy, *block_sum_dVz;
        //host allocation:
        block_sum_dX = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dY = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dZ = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_dVx = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVy = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVz = (double*)malloc(sizeof(double) * grid_size_);
       
        //print_kernel<<<grid_size,blockSize>>>(dX, dY, dZ, N);

        reduce_kernel_var<<<grid_size,blockSize>>>(dX, dY, dZ, CMsumblock_x, CMsumblock_y, CMsumblock_z,  N);

        //reduce_kernel<<<grid_size_,blockSize_,shared_mem_size_>>>(dX, dY, dZ, CMsumblock_x, CMsumblock_y, CMsumblock_z,  N);

        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dX, CMsumblock_x, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dY, CMsumblock_y, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dZ, CMsumblock_z, N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        reduce_kernel_var<<<grid_size,blockSize>>>(dVx, dVy, dVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz,  N);

        //reduce_kernel<<<grid_size_,blockSize_,shared_mem_size_>>>(dVx, dVy, dVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz,  N);

        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVx, CMsumblock_Vx, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVy, CMsumblock_Vy, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVz, CMsumblock_Vz, N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        cudaMemcpy(block_sum_dX, CMsumblock_x, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dY, CMsumblock_y, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dZ, CMsumblock_z, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_dVx, CMsumblock_Vx, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVy, CMsumblock_Vy, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVz, CMsumblock_Vz, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);




        *dX_tot=0.0; *dY_tot=0.0; *dZ_tot=0.0;
        *dVx_tot=0.0; *dVy_tot=0.0; *dVz_tot=0.0;


        for (int j = 0; j < grid_size_; j++)
        {
            *dX_tot += block_sum_dX[j];
            *dY_tot += block_sum_dY[j];
            *dZ_tot += block_sum_dZ[j];

            *dVx_tot += block_sum_dVx[j];
            *dVy_tot += block_sum_dVy[j];
            *dVz_tot += block_sum_dVz[j];


        }

        cudaDeviceSynchronize();
        printf("Xtot = %f, Ytot = %f, Ztot = %f\n", *dX_tot, *dY_tot, *dZ_tot); 

        double XCM , YCM, ZCM;
        XCM=0.0; YCM=0.0; ZCM=0.0;
 
        double VXCM , VYCM, VZCM;
        VXCM=0.0; VYCM=0.0; VZCM=0.0;

    
        int M_tot;
        M_tot = (mass*Nmd)+(mass_fluid*N);
        //int M_tot = 1 ;

        XCM = ( (mass*Nmd* *mdX_tot) + (mass_fluid*N* *dX_tot) )/M_tot;
        YCM = ( (mass*Nmd* *mdY_tot) + (mass_fluid*N* *dY_tot) )/M_tot;
        ZCM = ( (mass*Nmd* *mdZ_tot) + (mass_fluid*N* *dZ_tot) )/M_tot;

        cudaMemcpy(Xcm, &XCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm, &YCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm, &ZCM, sizeof(double), cudaMemcpyHostToDevice);
    
        //printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM); 
    
        VXCM = ( (mass*Nmd* *mdVx_tot) + (mass_fluid*N* *dVx_tot) )/M_tot;
        VYCM = ( (mass*Nmd* *mdVy_tot) + (mass_fluid*N* *dVy_tot) )/M_tot;
        VZCM = ( (mass*Nmd* *mdVz_tot) + (mass_fluid*N* *dVz_tot) )/M_tot;

        cudaMemcpy(Vxcm, &VXCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VYCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VZCM, sizeof(double), cudaMemcpyHostToDevice);
    
        printf("Xcm = %f, Ycm = %f, Zcm = %f\n", XCM, YCM, ZCM); 

        free(mdXtot); free(mdYtot); free(mdZtot); free(mdVxtot); free(mdVytot); free(mdVztot); 


        free(block_sum_dX); free(block_sum_dY); free(block_sum_dZ); free(block_sum_dVx); free(block_sum_dVy); free(block_sum_dVz); 
 

        
    }
    else
    {

        grid_size_=grid_size;

        double *block_sum_mdX, *block_sum_mdY, *block_sum_mdZ, *block_sum_mdVx, *block_sum_mdVy, *block_sum_mdVz;
        //host allocation:
        block_sum_mdX = (double*)malloc(sizeof(double) * grid_size);  block_sum_mdY = (double*)malloc(sizeof(double) * grid_size);  block_sum_mdZ = (double*)malloc(sizeof(double) * grid_size);
        block_sum_mdVx = (double*)malloc(sizeof(double) * grid_size); block_sum_mdVy = (double*)malloc(sizeof(double) * grid_size); block_sum_mdVz = (double*)malloc(sizeof(double) * grid_size);
       
        print_kernel<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Nmd);

        reduce_kernel_var<<<grid_size,blockSize>>>(mdX, mdY, mdZ, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        
        reduce_kernel_var<<<grid_size,blockSize>>>(mdVx, mdVy, mdVz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        cudaMemcpy(block_sum_mdX, CMsumblock_mdx, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdY, CMsumblock_mdy, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdZ, CMsumblock_mdz, grid_size*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_mdVx, CMsumblock_mdVx, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdVy, CMsumblock_mdVy, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdVz, CMsumblock_mdVz, grid_size*sizeof(double), cudaMemcpyDeviceToHost);


        *mdX_tot = 0.0; *mdY_tot = 0.0; *mdZ_tot = 0.0;
        *mdVx_tot = 0.0; *mdVy_tot = 0.0; *mdVz_tot = 0.0;



        for (int i = 0; i < grid_size; i++)
        {
            *mdX_tot +=block_sum_mdX[i];
            *mdY_tot +=block_sum_mdY[i];
            *mdZ_tot +=block_sum_mdZ[i];

            *mdVx_tot +=block_sum_mdVx[i];
            *mdVy_tot +=block_sum_mdVy[i];
            *mdVz_tot +=block_sum_mdVz[i];
        }

        cudaDeviceSynchronize();
        /*cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
        }*/
  
    
        /////// mpcd part:

        double *block_sum_dX, *block_sum_dY, *block_sum_dZ, *block_sum_dVx, *block_sum_dVy, *block_sum_dVz;
        //host allocation:
        block_sum_dX = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dY = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dZ = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_dVx = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVy = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVz = (double*)malloc(sizeof(double) * grid_size_);
       
        //print_kernel<<<grid_size,blockSize>>>(dX, dY, dZ, N);
        
        reduce_kernel_var<<<grid_size,blockSize>>>(dX, dY, dZ, CMsumblock_x, CMsumblock_y, CMsumblock_z,  N);

        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dX, CMsumblock_x, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dY, CMsumblock_y, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dZ, CMsumblock_z, N);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        reduce_kernel_var<<<grid_size,blockSize>>>(dVx, dVy, dVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz,  N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVx, CMsumblock_Vx, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVy, CMsumblock_Vy, N);
        //reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVz, CMsumblock_Vz, N);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        cudaMemcpy(block_sum_dX, CMsumblock_x, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dY, CMsumblock_y, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dZ, CMsumblock_z, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_dVx, CMsumblock_Vx, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVy, CMsumblock_Vy, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVz, CMsumblock_Vz, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);


        *dX_tot=0.0; *dY_tot=0.0; *dZ_tot=0.0;
        *dVx_tot=0.0; *dVy_tot=0.0; *dVz_tot=0.0;


        for (int j = 0; j < grid_size_; j++)
        {
            *dX_tot += block_sum_dX[j];
            *dY_tot += block_sum_dY[j];
            *dZ_tot += block_sum_dZ[j];

            *dVx_tot += block_sum_dVx[j];
            *dVy_tot += block_sum_dVy[j];
            *dVz_tot += block_sum_dVz[j];
        }

        cudaDeviceSynchronize();
        //printf("Xtot = %lf, Ytot = %lf, Ztot = %lf\n", *dX_tot, *dY_tot, *dZ_tot); 

        double XCM , YCM, ZCM;
        XCM=0.0; YCM=0.0; ZCM=0.0;

        double VXCM , VYCM, VZCM;
        VXCM=0.0; VYCM=0.0; VZCM=0.0;


    
        int M_tot = mass*Nmd+mass_fluid*N;
        

   
        XCM = ( (mass*Nmd* *mdX_tot) + (mass_fluid*N* *dX_tot) )/M_tot;
        YCM = ( (mass*Nmd* *mdY_tot) + (mass_fluid*N* *dY_tot) )/M_tot;
        ZCM = ( (mass*Nmd* *mdZ_tot) + (mass_fluid*N* *dZ_tot) )/M_tot;

        VXCM = ( (mass*Nmd* *mdVx_tot) + (mass_fluid*N* *dVx_tot) )/M_tot;
        VYCM = ( (mass*Nmd* *mdVy_tot) + (mass_fluid*N* *dVy_tot) )/M_tot;
        VZCM = ( (mass*Nmd* *mdVz_tot) + (mass_fluid*N* *dVz_tot) )/M_tot;


        cudaMemcpy(Xcm, &XCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm, &YCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm, &ZCM, sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(Vxcm, &VXCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VYCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vzcm, &VZCM, sizeof(double), cudaMemcpyHostToDevice);
    
    
        //printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM);
        //printf("Vxcm = %lf, Vycm = %lf, Vzcm = %lf\n", VXCM, VYCM, VZCM); 


        free(block_sum_mdX); free(block_sum_mdY); free(block_sum_mdZ); free(block_sum_mdVx); free(block_sum_mdVy); free(block_sum_mdVz);

        free(block_sum_dX); free(block_sum_dY); free(block_sum_dZ); free(block_sum_dVx); free(block_sum_dVy); free(block_sum_dVz); 
 
    }

    
}

////////////////////////////////// outer particles part:

/*__global__ void reduceKernel_outbox(double *input_X, double *output_X, double *input_V, double *output_V, double *x, double *y, double *z, int N, double *L, double *Xcm, double *Ycm, double *Zcm){
    extern __shared__ double sssdata_x[];
    extern __shared__ double sssdata_v[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    


    if (i < N){
        if((x[i]+ *Xcm) > L[0]/2 || (x[i]+ *Xcm) < -L[0]/2 || (y[i] + *Ycm) > L[1]/2 || (y[i] + *Ycm) < -L[1]/2 || (z[i] + *Zcm) > L[2]/2 || (z[i] + *Zcm) < -L[2]/2){
            sssdata_x[tid] = input_X[i];
            sssdata_v[tid] = input_V[i];

            __syncthreads();
        }
        else {
            sssdata_x[tid] = 0.0;
            sssdata_v[tid] = 0.0;
            __syncthreads();
        }
    }

    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sssdata_x[tid] += sssdata_x[tid + s];
            sssdata_v[tid] += sssdata_v[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_Reduce(sssdata_x, tid);
        warp_Reduce(sssdata_v, tid);
    }

    if (tid == 0) {
        output_X[blockIdx.x] = sssdata_x[0];
        output_V[blockIdx.x] = sssdata_v[0];
    }
}

__global__ void reduceKernel_outbox_var(double *input_X, double *output_X, double *input_V, double *output_V, double *x, double *y, double *z, int N, double *L, double *Xcm, double *Ycm, double *Zcm){
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int gridSize = blockSize*gridDim.x;
    double sum1 = 0.0;
    double sum2 = 0.0;
    


    
    for (int j = i; j < N; j += gridSize)
    {
        if((x[j]+ *Xcm) > L[0]/2 || (x[j]+ *Xcm) < -L[0]/2 || (y[j] + *Ycm) > L[1]/2 || (y[j] + *Ycm) < -L[1]/2 || (z[j] + *Zcm) > L[2]/2 || (z[j] + *Zcm) < -L[2]/2)
        {
            sum1 += input_X[j];
            sum2 += input_V[j];
        }
        
    }

    __shared__ double ssdata_x[blockSize];
    __shared__ double ssdata_v[blockSize];

    ssdata_x[tid] = sum1;
    ssdata_v[tid] = sum2;
    __syncthreads();

    for (int s = blockSize/2; s>0; s/=2)
    {
        if (tid<s){
            ssdata_x[tid] += ssdata_x[tid+s];
            ssdata_v[tid] += ssdata_v[tid+s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output_X[blockIdx.x] = ssdata_x[0];
        output_V[blockIdx.x] = ssdata_v[0];
    }
}




__global__ void particles_outbox_counter(double *x, double *y, double *z, int N, double *L, double *Xcm, double *Ycm, double *Zcm, int *n_outbox){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){

        n_outbox[i] =0; 
        
        if((x[i]+ *Xcm) > L[0]/2 || (x[i]+ *Xcm) < -L[0]/2 || (y[i] + *Ycm) > L[1]/2 || (y[i] + *Ycm) < -L[1]/2 || (z[i] + *Zcm) > L[2]/2 || (z[i] + *Zcm) < -L[2]/2){
        
            n_outbox[i] = 1;
            printf("the %i th particle goes out\n", i);
            printf("***** X[%i]=%f, Y[%i]=%f, Z[%i]=%f\n", i, x[i]+ *Xcm, i, y[i] + *Ycm, i, z[i] + *Zcm);

        }
        else n_outbox[i] = 0;


    }

}


__host__ void outerParticles_reduceKernels_(double *dX, double *dY, double *dZ, double *dVx, double *dVy, double *dVz, double *Xcm, double *Ycm, double *Zcm, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz,
 int shared_mem_size_, int blockSize_, int grid_size_, int N, double *L, int *n_outbox, int *CMsumblock_n_outbox){

    reduceKernel_outbox_var<<<grid_size_,blockSize_>>>(dX, CMsumblock_x, dVx, CMsumblock_Vx, dX, dY , dZ, N, L, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    reduceKernel_outbox_var<<<grid_size_,blockSize_>>>(dY, CMsumblock_y, dVy, CMsumblock_Vy, dX, dY , dZ, N, L, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    reduceKernel_outbox_var<<<grid_size_,blockSize_>>>(dZ, CMsumblock_z, dVz, CMsumblock_Vz, dX, dY , dZ, N, L, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    particles_outbox_counter<<<grid_size_,blockSize_>>>(dX, dY, dZ, N, L, Xcm, Ycm, Zcm, n_outbox);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    intreduceKernel_var<<<grid_size_,blockSize_>>>(n_outbox, CMsumblock_n_outbox, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );





}



//a function in which center of mass of the particles that go outside the box is measured
__host__ void outerParticles_CM_system(double *mdX, double *mdY, double *mdZ,  double *dX, double *dY, double *dZ,  double *mdVx, double *mdVy, double *mdVz, double *dVx, double *dVy, double *dVz, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *dX_tot, double *dY_tot, double *dZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *dVx_tot, double *dVy_tot, double *dVz_tot, int *dn_mpcd_tot, int *dn_md_tot,
int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double mass, double mass_fluid, double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, 
double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, int *CMsumblock_n_outbox_mpcd, int *CMsumblock_n_outbox_md, int topology, double *L){
 
    if(topology == 4)
    {

        //MD particle part
        double *mdXtot, *mdYtot, *mdZtot;
        
        double *mdVxtot, *mdVytot, *mdVztot;
       
        int *dnMDtot;

        double dL[3];

        double *XCM, *YCM, *ZCM;
        

        mdXtot = (double *)malloc(sizeof(double));  mdYtot = (double *)malloc(sizeof(double));  mdZtot = (double *)malloc(sizeof(double));
        mdVxtot = (double *)malloc(sizeof(double));  mdVytot = (double *)malloc(sizeof(double));  mdVztot = (double *)malloc(sizeof(double));
        dnMDtot = (int *)malloc(sizeof(int));
        XCM = (double *)malloc(sizeof(double));  YCM = (double *)malloc(sizeof(double)); ZCM = (double *)malloc(sizeof(double));
        
        
        cudaMemcpy(mdXtot, mdX, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdYtot, mdY, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdZtot, mdZ, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(mdVxtot, mdVx, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdVytot, mdVy, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdVztot, mdVz, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(dL, L, sizeof(double) * 3, cudaMemcpyDeviceToHost);

        cudaMemcpy(XCM, Xcm, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(YCM, Ycm, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(ZCM, Zcm, sizeof(double), cudaMemcpyDeviceToHost);

 
        //printf("dL[0]=%f\n", dL[0]);
        //printf("*mdXtot=%f\n", *mdXtot);
        //printf("*XCM=%f\n", *XCM);


        if((*mdXtot+ *XCM) > dL[0]/2 || (*mdXtot+ *XCM) < -dL[0]/2 || (*mdYtot + *YCM) > dL[1]/2 || (*mdYtot + *YCM) < -dL[1]/2 || (*mdZtot + *ZCM) > dL[2]/2 || (*mdZtot + *ZCM) < -dL[2]/2)
        {
                    printf("oooo\n");
                    *mdX_tot = *mdXtot;
                    *mdY_tot = *mdYtot;
                    *mdZ_tot = *mdZtot;

                    *mdVx_tot = *mdVxtot;
                    *mdVy_tot = *mdVytot;
                    *mdVz_tot = *mdVztot;

                    *dn_md_tot = 1;
        }
        else
        {
                    *mdX_tot = 0.0;
                    *mdY_tot = 0.0;
                    *mdZ_tot = 0.0;

                    *mdVx_tot = 0.0;
                    *mdVy_tot = 0.0;
                    *mdVz_tot = 0.0;

                    *dn_md_tot = 0;


        }


        ////////////////// MPCD particles part:
   
        grid_size_=grid_size;

        double *block_sum_dX, *block_sum_dY, *block_sum_dZ, *block_sum_dVx, *block_sum_dVy, *block_sum_dVz;
        int *block_sum_n_mpcd;
        //host allocation:
        block_sum_dX = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dY = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dZ = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_dVx = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVy = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVz = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_n_mpcd = (int*)malloc(sizeof(int) * grid_size_);





        outerParticles_reduceKernels_(dX, dY, dZ, dVx, dVy, dVz, Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz,
                                    shared_mem_size_, blockSize_, grid_size_, N, L, n_outbox_mpcd, CMsumblock_n_outbox_mpcd);

        


        cudaMemcpy(block_sum_dX, CMsumblock_x, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dY, CMsumblock_y, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dZ, CMsumblock_z, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_dVx, CMsumblock_Vx, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVy, CMsumblock_Vy, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVz, CMsumblock_Vz, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_n_mpcd, CMsumblock_n_outbox_mpcd, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
        

        *dX_tot=0.0; *dY_tot=0.0; *dZ_tot=0.0;
        *dVx_tot=0.0; *dVy_tot=0.0; *dVz_tot=0.0;
        *dn_mpcd_tot=0;


        for (int j = 0; j < grid_size_; j++)
        {
            *dX_tot += block_sum_dX[j];
            *dY_tot += block_sum_dY[j];
            *dZ_tot += block_sum_dZ[j];

            *dVx_tot += block_sum_dVx[j];
            *dVy_tot += block_sum_dVy[j];
            *dVz_tot += block_sum_dVz[j];

            *dn_mpcd_tot += block_sum_n_mpcd[j];


        }

        cudaDeviceSynchronize();
        //printf("Xtot = %lf, Ytot = %lf, Ztot = %lf\n", *dX_tot, *dY_tot, *dZ_tot); 

        double XCM_out, YCM_out, ZCM_out;
        XCM_out=0.0; YCM_out=0.0; ZCM_out=0.0;
 
        double VXCM_out , VYCM_out, VZCM_out;
        VXCM_out=0.0; VYCM_out=0.0; VZCM_out=0.0;

    
        int M_tot;
        M_tot = mass * *dn_md_tot+mass_fluid * *dn_mpcd_tot;
        printf("outerparticles M_tot=%i\n", M_tot);
        printf("*dn_mpcd_tot=%i\n", *dn_mpcd_tot);



        if(M_tot != 0){

            XCM_out = ( (mass * *dn_md_tot * *mdX_tot) + (mass_fluid * *dn_mpcd_tot * *dX_tot) )/M_tot;
            YCM_out = ( (mass * *dn_md_tot * *mdY_tot) + (mass_fluid * *dn_mpcd_tot * *dY_tot) )/M_tot;
            ZCM_out = ( (mass * *dn_md_tot * *mdZ_tot) + (mass_fluid * *dn_mpcd_tot * *dZ_tot) )/M_tot;
            VXCM_out = ( (mass* *dn_md_tot * *mdVx_tot) + (mass_fluid* *dn_mpcd_tot * *dVx_tot) )/M_tot;
            VYCM_out = ( (mass* *dn_md_tot * *mdVy_tot) + (mass_fluid* *dn_mpcd_tot * *dVy_tot) )/M_tot;
            VZCM_out = ( (mass* *dn_md_tot * *mdVz_tot) + (mass_fluid* *dn_mpcd_tot * *dVz_tot) )/M_tot;
          }
        else{

            XCM_out = 0.0;
            YCM_out = 0.0;
            ZCM_out = 0.0;
            VXCM_out = 0.0;
            VYCM_out = 0.0;
            VZCM_out = 0.0;
          }

        printf("Xcm_out = %f, Ycm_out = %f, Zcm_out = %f\n", XCM_out, YCM_out, ZCM_out);
        
        cudaMemcpy(Xcm_out, &XCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm_out, &YCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm_out, &ZCM_out, sizeof(double), cudaMemcpyHostToDevice);  

        cudaMemcpy(Vxcm_out, &VXCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm_out, &VYCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm_out, &VZCM_out, sizeof(double), cudaMemcpyHostToDevice);
    
        //printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM); 

        free(mdXtot); free(mdYtot); free(mdZtot); free(mdVxtot); free(mdVytot); free(mdVztot); free(dnMDtot);

        free(block_sum_dX); free(block_sum_dY); free(block_sum_dZ); free(block_sum_dVx); free(block_sum_dVy); free(block_sum_dVz); free(block_sum_n_mpcd);
 

        
    }
    else
    {

        //MD part:
        
        outerParticles_reduceKernels_(mdX, mdY, mdZ, mdVx, mdVy, mdVz, Xcm, Ycm, Zcm, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, shared_mem_size_, blockSize_, grid_size_, N, L, n_outbox_md, CMsumblock_n_outbox_md);

        grid_size_ = grid_size;

        double *block_sum_mdX, *block_sum_mdY, *block_sum_mdZ, *block_sum_mdVx, *block_sum_mdVy, *block_sum_mdVz;
        int *block_sum_n_md;
        //host allocation:
        block_sum_mdX = (double*)malloc(sizeof(double) * grid_size);  block_sum_mdY = (double*)malloc(sizeof(double) * grid_size);  block_sum_mdZ = (double*)malloc(sizeof(double) * grid_size);
        block_sum_mdVx = (double*)malloc(sizeof(double) * grid_size); block_sum_mdVy = (double*)malloc(sizeof(double) * grid_size); block_sum_mdVz = (double*)malloc(sizeof(double) * grid_size);
        block_sum_n_md = (int*)malloc(sizeof(int) * grid_size);
 

        cudaMemcpy(block_sum_mdX, CMsumblock_mdx, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdY, CMsumblock_mdy, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdZ, CMsumblock_mdz, grid_size*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_mdVx, CMsumblock_mdVx, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdVy, CMsumblock_mdVy, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_mdVz, CMsumblock_mdVz, grid_size*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_n_md, CMsumblock_n_outbox_md, grid_size*sizeof(int), cudaMemcpyDeviceToHost);


        *mdX_tot = 0.0; *mdY_tot = 0.0; *mdZ_tot = 0.0;
        *mdVx_tot = 0.0; *mdVy_tot = 0.0; *mdVz_tot = 0.0;
        *dn_md_tot=0;

        

        for (int i = 0; i < grid_size; i++)
        {
            *mdX_tot +=block_sum_mdX[i];
            *mdY_tot +=block_sum_mdY[i];
            *mdZ_tot +=block_sum_mdZ[i];

            *mdVx_tot +=block_sum_mdVx[i];
            *mdVy_tot +=block_sum_mdVy[i];
            *mdVz_tot +=block_sum_mdVz[i];

            *dn_md_tot +=block_sum_n_md[i];
        }

        cudaDeviceSynchronize();
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
        }
  
    
  

        //MPCD part:

        outerParticles_reduceKernels_(dX, dY, dZ, dVx, dVy, dVz, Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, shared_mem_size_, blockSize_, grid_size_, N, L, n_outbox_mpcd, CMsumblock_n_outbox_mpcd);

        grid_size_=grid_size;

        double *block_sum_dX, *block_sum_dY, *block_sum_dZ, *block_sum_dVx, *block_sum_dVy, *block_sum_dVz;
        int *block_sum_n_mpcd;
        //host allocation:
        block_sum_dX = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dY = (double*)malloc(sizeof(double) * grid_size_);  block_sum_dZ = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_dVx = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVy = (double*)malloc(sizeof(double) * grid_size_); block_sum_dVz = (double*)malloc(sizeof(double) * grid_size_);
        block_sum_n_mpcd = (int*)malloc(sizeof(int) * grid_size_);

        cudaMemcpy(block_sum_dX, CMsumblock_x, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dY, CMsumblock_y, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dZ, CMsumblock_z, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_dVx, CMsumblock_Vx, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVy, CMsumblock_Vy, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_sum_dVz, CMsumblock_Vz, grid_size_*sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(block_sum_n_mpcd, CMsumblock_n_outbox_mpcd, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);


        *dX_tot=0.0; *dY_tot=0.0; *dZ_tot=0.0;
        *dVx_tot=0.0; *dVy_tot=0.0; *dVz_tot=0.0;
        *dn_mpcd_tot=0;


        for (int j = 0; j < grid_size; j++)
        {
            *dX_tot += block_sum_dX[j];
            *dY_tot += block_sum_dY[j];
            *dZ_tot += block_sum_dZ[j];

            *dVx_tot += block_sum_dVx[j];
            *dVy_tot += block_sum_dVy[j];
            *dVz_tot += block_sum_dVz[j];

            *dn_mpcd_tot += block_sum_n_mpcd[j];
        }

        cudaDeviceSynchronize();
        //printf("Xtot = %lf, Ytot = %lf, Ztot = %lf\n", *dX_tot, *dY_tot, *dZ_tot); 

        double XCM_out , YCM_out, ZCM_out;
        XCM_out=0.0; YCM_out=0.0; ZCM_out=0.0;

        double VXCM_out , VYCM_out, VZCM_out;
        VXCM_out=0.0; VYCM_out=0.0; VZCM_out=0.0;


    
        int M_tot = mass * *dn_md_tot + mass_fluid * *dn_mpcd_tot;
        

        if( M_tot !=0 ){
            XCM_out = ( (mass * *dn_md_tot * *mdX_tot) + (mass_fluid * *dn_mpcd_tot * *dX_tot) )/M_tot;
            YCM_out = ( (mass * *dn_md_tot * *mdY_tot) + (mass_fluid * *dn_mpcd_tot * *dY_tot) )/M_tot;
            ZCM_out = ( (mass * *dn_md_tot * *mdZ_tot) + (mass_fluid * *dn_mpcd_tot * *dZ_tot) )/M_tot;

            VXCM_out = ( (mass * *dn_md_tot * *mdX_tot) + (mass_fluid * *dn_mpcd_tot * *dVx_tot) )/M_tot;
            VYCM_out = ( (mass * *dn_md_tot * *mdY_tot) + (mass_fluid * *dn_mpcd_tot * *dVy_tot) )/M_tot;
            VZCM_out = ( (mass * *dn_md_tot * *mdZ_tot) + (mass_fluid * *dn_mpcd_tot * *dVz_tot) )/M_tot;
        }
        else if ( M_tot == 0){
            XCM_out = 0.0;
            YCM_out = 0.0;
            ZCM_out = 0.0;

            VXCM_out = 0.0;
            VYCM_out = 0.0;
            VZCM_out = 0.0;

        }


        cudaMemcpy(Xcm_out, &XCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm_out, &YCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm_out, &ZCM_out, sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(Vxcm_out, &VXCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm_out, &VYCM_out, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vzcm_out, &VZCM_out, sizeof(double), cudaMemcpyHostToDevice);
    
    
        printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM_out, YCM_out, ZCM_out);
        printf("Vxcm = %lf, Vycm = %lf, Vzcm = %lf\n", VXCM_out, VYCM_out, VZCM_out);


    

        free(block_sum_mdX); free(block_sum_mdY); free(block_sum_mdZ); free(block_sum_mdVx); free(block_sum_mdVy); free(block_sum_mdVz); free(block_sum_n_md);

        free(block_sum_dX); free(block_sum_dY); free(block_sum_dZ); free(block_sum_dVx); free(block_sum_dVy); free(block_sum_dVz); free(block_sum_n_mpcd);
 
    }



}*/