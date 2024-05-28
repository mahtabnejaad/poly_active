__device__ void warp_Reduce(volatile double *ssdata, int tid) {
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
}

__global__ void reduceKernel_(double *input, double *output, int N) {
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
}

__global__ void intreduceKernel_(int *input, int *output, int N) {
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
}

__host__ void CM_system(double *mdX, double *mdY, double *mdZ, double *dVx, double *dVy, double *dVz, double *mdVx, double *mdVy, double *mdVz, double *dX, double *dY, double *dZ, int Nmd, int N, 
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *dX_tot, double *dY_tot, double *dZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *dVx_tot, double *dVy_tot, double *dVz_tot, int grid_size, int shared_mem_size, int blockSize_, int grid_size_, int mass, int mass_fluid, double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, 
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, int topology){
 
    if(topology == 4)
    {
        //MD particle part
        double mdXtot, mdYtot, mdZtot;
        mdXtot=0.0; mdYtot=0.0; mdZtot=0.0;
        double mdVxtot, mdVytot, mdVztot;
        mdVxtot=0.0; mdVytot=0.0; mdVztot=0.0;
        
        cudaMemcpy(&mdXtot, mdX, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&mdYtot, mdY, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&mdZtot, mdZ, sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(&mdVxtot, mdVx, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&mdVytot, mdVy, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&mdVztot, mdVz, sizeof(double), cudaMemcpyDeviceToHost);

        *mdX_tot = mdXtot;
        *mdY_tot = mdYtot;
        *mdZ_tot = mdZtot;

        *mdVx_tot = mdVxtot;
        *mdVy_tot = mdVytot;
        *mdVz_tot = mdVztot;

        //MPCD particles part
        int shared_mem_size_ = 3 * blockSize_ * sizeof(double);

        double block_sum_dX[grid_size_]; double block_sum_dY[grid_size_]; double block_sum_dZ[grid_size_];
        double block_sum_dVx[grid_size_]; double block_sum_dVy[grid_size_]; double block_sum_dVz[grid_size_];


        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dX, CMsumblock_x, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dY, CMsumblock_y, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dZ, CMsumblock_z, N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVx, CMsumblock_Vx, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVy, CMsumblock_Vy, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVz, CMsumblock_Vz, N);
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
        //int M_tot = 1 ;

        XCM = ( (mass*Nmd* *mdX_tot) + (mass_fluid*N* *dX_tot) )/M_tot;
        YCM = ( (mass*Nmd* *mdY_tot) + (mass_fluid*N* *dY_tot) )/M_tot;
        ZCM = ( (mass*Nmd* *mdZ_tot) + (mass_fluid*N* *dZ_tot) )/M_tot;

        cudaMemcpy(Xcm, &XCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm, &YCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm, &ZCM, sizeof(double), cudaMemcpyHostToDevice);
    
        //printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM); 
    
        VXCM = ( (mass*Nmd* *mdX_tot) + (mass_fluid*N* *dVx_tot) )/M_tot;
        VYCM = ( (mass*Nmd* *mdY_tot) + (mass_fluid*N* *dVy_tot) )/M_tot;
        VZCM = ( (mass*Nmd* *mdZ_tot) + (mass_fluid*N* *dVz_tot) )/M_tot;

        cudaMemcpy(Vxcm, &VXCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VYCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VZCM, sizeof(double), cudaMemcpyHostToDevice);
    
        //printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM); 

        
    }
    else
    {
        double block_sum_mdX[grid_size]; double block_sum_mdY[grid_size]; double block_sum_mdZ[grid_size];
        reduce_kernel<<<grid_size,blockSize, shared_mem_size>>>(mdX, mdY, mdZ, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        double block_sum_mdVx[grid_size]; double block_sum_mdVy[grid_size]; double block_sum_mdVz[grid_size];
        reduce_kernel<<<grid_size,blockSize, shared_mem_size>>>(mdVx, mdVy, mdVz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, Nmd);
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
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
        }
  
    
  


        int shared_mem_size_ = 3 * blockSize_ * sizeof(double);

        double block_sum_dX[grid_size_]; double block_sum_dY[grid_size_]; double block_sum_dZ[grid_size_];

        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dX, CMsumblock_x, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dY, CMsumblock_y, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dZ, CMsumblock_z, N);

        double block_sum_dVx[grid_size_]; double block_sum_dVy[grid_size_]; double block_sum_dVz[grid_size_];

        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVx, CMsumblock_Vx, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVy, CMsumblock_Vy, N);
        reduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(dVz, CMsumblock_Vz, N);

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


        for (int j = 0; j < grid_size; j++)
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

        VXCM = ( (mass*Nmd* *mdX_tot) + (mass_fluid*N* *dVx_tot) )/M_tot;
        VYCM = ( (mass*Nmd* *mdY_tot) + (mass_fluid*N* *dVy_tot) )/M_tot;
        VZCM = ( (mass*Nmd* *mdZ_tot) + (mass_fluid*N* *dVz_tot) )/M_tot;


        cudaMemcpy(Xcm, &XCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Ycm, &YCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Zcm, &ZCM, sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(Vxcm, &VXCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vycm, &VYCM, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(Vzcm, &VZCM, sizeof(double), cudaMemcpyHostToDevice);
    
    
        printf("Xcm = %lf, Ycm = %lf, Zcm = %lf\n", XCM, YCM, ZCM);
        printf("Vxcm = %lf, Vycm = %lf, Vzcm = %lf\n", VXCM, VYCM, VZCM); 
    }


}