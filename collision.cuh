__global__ void checkfunction1(int *n, double *m, int Nc)
{
   int idxx = blockIdx.x * blockDim.x + threadIdx.x; 
   if (idxx<Nc){
      //printf("check 1 m[%i]=%f\n", idxx, m[idxx]);
      //printf("check 1 n[%i]=%i\n", idxx, n[idxx]);

    }
}
__global__ void checkfunction2(int *n, double *m, int *index, int N)
{
   int idxx = blockIdx.x * blockDim.x + threadIdx.x; 
   
   if (idxx<N){
   
      int indxx = index[idxx];
      //printf("check 2 m[%i]=%f\n, tid=%i", indxx, m[indxx], idxx);
      //printf("check 2 n[%i]=%i, tid =%i\n", indxx, n[indxx], idxx);

   }
}




//cell sort, calculating the index of each particle there is a unique ID for each cell  based on their position
//Purpose: This kernel calculates the unique ID for each particle's cell based on their position.
//x,y,z are positions of the particles //L=dimentions of the cells//N=number of particles
//index= Array to store the calculated unique ID (index) for each particle's cell
__global__ void cellSort(double* x,double* y,double* z, double *L, int* index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        //index[tid] = int(x[tid] + L[0] / 2 + 2) + (L[0] + 4) * int(y[tid] + L[1] / 2 + 2) + (L[0] + 4) * (L[1] + 4) * int(z[tid] + L[2] / 2 + 2) ;
        index[tid] = int(x[tid] + L[0] / 2 + 1) + (L[0] + 2) * int(y[tid] + L[1] / 2 + 1) + (L[0] + 2) * (L[1] + 2) * int(z[tid] + L[2] / 2 + 1) ;
        //if (index[tid]>307918 || index[tid]<0) printf("index[%i]=%i , x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, index[tid], tid, x[tid], tid, y[tid], tid, z[tid]);
        //if(x[tid] == 0.00000)  printf("index[tid]=%i, y[tid]=%f, z[tid]=%f", index[tid], y[tid], z[tid]);
        //printf("index[%i]=%f\n", tid, index[tid]);
        //printf("index[%i]=%i, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, index[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
    }

} //Output: The index array will be updated with the computed unique IDs.


//Purpose: This kernel initializes cell-related arrays to prepare for calculations.
//ux,uy,uz are cell velocities.//n is th number of particles in each cell//e is cell energy//Nc is number of cells.
__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n, double *m, int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0.0;
        uy[tid] = 0.0;
        uz[tid] = 0.0;
        n[tid] = 0;
        m[tid] = 0.0;
        e[tid]=0.0;
        //printf("inital m[%i]=%f\n", tid, m[tid]);
    } 
} //Output: The arrays ux, uy, uz, e, and n will be set to 0.

//Purpose: This kernel calculates the mean velocity and mass of particles in each cell.
//vx, vy, vz: Arrays containing the particle velocities 
__global__ void MeanVelCell(double* ux, double* vx,double* uy, double* vy,double* uz, double* vz,int* index, int *n, double *m, double mass, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];//Retrieves the index idxx of the cell to which the particle belongs.
        //Multiplies the velocity components (vx, vy, vz) of the particle by its mass, storing the result in a temporary variable tmp.
        double tmp =vx[tid] *mass; 
        //Atomically adds the value of tmp to the corresponding velocity component sum (ux, uy, uz) of the cell idxx.
        atomicAdd(&ux[idxx] , tmp );
        tmp =vy[tid] *mass;
        atomicAdd(&uy[idxx] , tmp );
        tmp =vz[tid] *mass;
        atomicAdd(&uz[idxx] , tmp );
        //Atomically increments the counters n[idxx] and m[idxx] by 1 and mass, respectively. 
        //These counters are used to keep track of the number of particles and the total mass within each cell.
        atomicAdd(&n[idxx] , 1 );
        atomicAdd(&m[idxx], mass);
    }
}  //Output: The ux, uy, uz, n, and m arrays will be updated with the calculated mean velocities and masses for each cell.

__global__ void RotationStep1(double *ux , double *uy ,double *uz,double *rot, double *m ,double *phi , double *theta, int Nc)
//This kernel performs a rotation transformation on cell velocities and calculates rotation matrices for each cell.
//ux, uy, uz: Arrays containing the cell velocities //rot: Array to store the rotation matrices for each cell.
//m: Array containing the mass of particles in each cell. //phi, theta: Arrays containing rotation angles (phi, theta) for each cell.
//Nc: Number of cells.
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha = 13.0 / 18.0 * M_PI;
    double co = cos(alpha), si = sin(alpha);
    if (tid<Nc)
    {
        //if(m[tid] == 0.0)  printf("m is zero and ux[%i]=%f, uy[%i]=%f, uz[%i]=%f\n", tid, ux[tid], tid, uy[tid], tid, uz[tid]);
        theta[tid] = theta[tid]* 2 -1; //This line modifies the value of theta for the current particle or cell. 
                                       //It scales the value by 2 and subtracts 1, 
                                       //effectively mapping the value from the range [0, 1] to the range [-1, 1].
        phi[tid] = phi[tid]* M_PI*2;   // It scales the value by 2 * pi (where M_PI is the constant for pi) to map it from the range [0, 1] to the range [0, 2*pi].
        if(m[tid] != 0.0){
            ux[tid] = ux[tid]/m[tid];
            uy[tid] = uy[tid]/m[tid];
            uz[tid] = uz[tid]/m[tid];
        }

        (isnan(ux[tid])|| isnan(uy[tid]) || isnan(uz[tid])) ? printf("RRux[%i]=%f, uy[%i]=%f, uz[%i]=%f \n", tid, ux[tid], tid, uy[tid], tid, uz[tid])
                                                         : printf("");

        //The next three lines calculate three components (n1, n2, and n3) of a unit vector n based on theta and phi.
        //This unit vector n will be used to construct the rotation matrix in the subsequent lines.
        double n1 = std::sqrt(1 - theta[tid] * theta[tid]) * cos(phi[tid]);
        double n2 = std::sqrt(1 - theta[tid] * theta[tid]) * sin(phi[tid]);
        double n3 = theta[tid];
        
        //The following nine lines calculate the elements of the 3x3 rotation matrix rot for the current particle or cell using the unit vector n and the constants co and si. 
        //The rotation matrix will be stored in the rot array at the appropriate index (tid*9 + i)
        rot[tid*9+0] =n1 * n1 + (1 - n1 * n1) * co ;
        rot[tid*9+1] =n1 * n2 * (1 - co) - n3 * si;
        rot[tid*9+2] =n1 * n3 * (1 - co) + n2 * si;
        rot[tid*9+3] =n1 * n2 * (1 - co) + n3 * si;
        rot[tid*9+4] =n2 * n2 + (1 - n2 * n2) * co;
        rot[tid*9+5] =n2 * n3 * (1 - co) - n1 * si;
        rot[tid*9+6] =n1 * n3 * (1 - co) - n2 * si;
        rot[tid*9+7] =n2 * n3 * (1 - co) + n1 * si;
        rot[tid*9+8] =n3 * n3 + (1 - n3 * n3) * co;
        
    }
} //Output: The rot array will be updated with the calculated rotation matrices.

__global__ void RotationStep2(double *rvx , double *rvy, double *rvz , double *rot , int *index,int N)
//This kernel applies the rotation matrices calculated in the previous step to the particle velocities
//rvx, rvy, rvz: Arrays containing the relative velocities of particles (with respect to their cells).
//rot: Array containing the rotation matrices for each cell
//index: Array containing the unique IDs of each particle's cell.
//N: Number of particles.
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(tid<N)
    {
        const unsigned int idxx = index[tid];//This line retrieves the unique ID (idxx) of the cell associated with the current particle. The index array maps each particle to its corresponding cell using unique IDs.

        double RV[3] = {rvx[tid] , rvy[tid] , rvz[tid]}; //This line creates a 3-element array RV to store the current relative velocity components (rvx[tid], rvy[tid], rvz[tid]) of the current particle.
        double rv[3] = {0.};//This line creates a 3-element array rv initialized to all zeros. This array will be used to store the updated relative velocity components after applying the rotation.
        
        //The following two nested loops are used to calculate the updated relative velocity components (rv) after applying the rotation:
        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
                rv[i] += rot[idxx*9+3*j+i] * RV[j];//This line updates the i-th component of rv by adding the product of the corresponding element from the rotation matrix (rot) and the j-th component of the current relative velocity (RV[j]).
                                                   //The rotation matrix element to be used is rot[idxx*9+3*j+i]. Since rot is stored as a 1D array representing a 3x3 matrix for each cell,
                                                   //the index calculation idxx*9+3*j+i accesses the appropriate element of the rotation matrix for the current cell. 
                                                   //The i index iterates over rows, and the j index iterates over columns, effectively performing matrix multiplication between the rotation matrix and the relative velocity vector.
        }
    
        // This line updates the relative velocity components (rvx[tid], rvy[tid], rvz[tid]) of the current particle with the values stored in the rv array.
        // These updated relative velocity components now reflect the rotation applied to the particle's motion.
        rvx[tid] = rv[0];
        rvy[tid] = rv[1];
        rvz[tid] = rv[2];   
    }
} //Output: The rvx, rvy, and rvz arrays will be updated with the transformed velocities.

/*__global__ void MakeCellReady(double* ux , double* uy , double* uz,double* e, int* n,double* m,int Nc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nc)
    {
        ux[tid] = 0;
        uy[tid] = 0;
        uz[tid] = 0;
        n[tid] = 0;
        e[tid]=0;
        m[tid] = 0;
    }

}*/

__global__ void UpdateVelocity(double* vx, double *vy, double *vz , double* ux, double *uy , double *uz ,double *factor,int *index, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;// This index identifies which particle's velocity components (vx, vy, vz) the current thread will handle.
    if (tid<N)
    {   
        //idxx represents the unique ID of the cell associated with the current particle. It is used to access the corresponding mean velocities of the cell (ux[idxx], uy[idxx], uz[idxx]) 
        //in order to update the particle velocities (vx[tid], vy[tid], vz[tid]) based on the mean velocities and the factor array.
        const unsigned int idxx = index[tid];
        vx[tid] = ux[idxx] + vx[tid]*factor[idxx]; 
        vy[tid] = uy[idxx] + vy[tid]*factor[idxx];
        vz[tid] = uz[idxx] + vz[tid]*factor[idxx];
    }

}

__global__ void relativeVelocity(double* ux , double* uy , double* uz, int* n, double* vx, double* vy, double* vz, int* index,int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];
        vx[tid] = vx[tid] - ux[idxx] ;
        vy[tid] = vy[tid] - uy[idxx] ;
        vz[tid] = vz[tid] - uz[idxx] ;

        (isnan(vx[tid])|| isnan(vy[tid]) || isnan(vz[tid])) ? printf("BBvx[%i]=%f, vy[%i]=%f, vz[%i]=%f \n", tid, vx[tid], tid, vy[tid], tid, vz[tid])
                                                            : printf("");

        (isnan(ux[idxx])|| isnan(uy[idxx]) || isnan(uz[idxx])) ? printf("00ux[%i]=%f, uy[%i]=%f, uz[%i]=%f \n", idxx, ux[idxx], idxx, uy[idxx], idxx, uz[idxx])
                                                         : printf("");
    }

}

__host__ void MPCD_MD_collision(double *d_vx ,double *d_vy ,double *d_vz , int *d_index,
double *d_mdVx ,double *d_mdVy,double *d_mdVz , int *d_mdIndex,
double  *d_ux ,double *d_uy ,double *d_uz ,
double *d_e ,double *d_scalefactor, int *d_n , double *d_m,
double *d_rot, double *d_theta, double *d_phi ,
int N , int Nmd, int Nc,
curandState *devStates, int grid_size)
{
            //This launches the MakeCellReady kernel with the specified grid size and block size.
            //The kernel resets cell properties such as mean velocity (d_ux, d_uy, d_uz), energy (d_e), and count (d_n, d_m) to zero for all cells (Nc).
            MakeCellReady<<<grid_size,blockSize>>>(d_ux , d_uy, d_uz ,d_e, d_n, d_m, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );            

            //This launches the MeanVelCell kernel with the specified grid size and block size.
            //The kernel calculates the mean velocities (d_ux, d_uy, d_uz) of particles within each cell based on their individual velocities (d_vx, d_vy, d_vz). 
            //The d_index array maps each particle to its corresponding cell. 
            //The result is updated in the d_ux, d_uy, and d_uz arrays, and the particle count (d_n) and mass (d_m) arrays are updated for each cell (N is the total number of particles).
            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_vx , d_uy, d_vy, d_uz, d_vz, d_index, d_n , d_m, 1 ,N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //This launches the MeanVelCell kernel again, but this time it calculates the mean velocities of MD particles within each cell.
            // The MD particle velocities are provided in the d_mdVx, d_mdVy, and d_mdVz arrays, and the d_mdIndex array maps each MD particle to its corresponding cell.
            // The result is updated in the d_ux, d_uy, and d_uz arrays, and the particle count (d_n) and mass (d_m) arrays are updated for each MD cell (Nmd is the total number of MD particles).
            MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_mdVx , d_uy, d_mdVy, d_uz, d_mdVz, d_mdIndex, d_n ,d_m , density ,Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() ); 

            checkfunction1<<<grid_size,blockSize>>>(d_n, d_m , Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            checkfunction2<<<grid_size,blockSize>>>(d_n, d_m , d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );



            //This launches the RotationStep1 kernel with the specified grid size and block size.
            // The kernel calculates the rotation matrices (d_rot) for each cell based on the angle values (d_phi, d_theta) and the mass (d_m) of particles in each cell.
            // The number of cells is given by Nc.
            RotationStep1<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_rot, d_m, d_phi, d_theta, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // The kernel calculates the relative velocities between particles and their corresponding cell mean velocities.
            // It uses the previously computed mean velocities (d_ux, d_uy, d_uz) and particle velocities (d_vx, d_vy, d_vz). The d_index array maps each particle to its corresponding cell. 
            //The total number of particles is given by N.
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_vx, d_vy, d_vz, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //This launches the relativeVelocity kernel again, but this time it calculates the relative velocities between MD particles and their corresponding cell mean velocities
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //The kernel is responsible for updating the velocities of regular particles (d_vx, d_vy, d_vz) based on the calculated rotation matrices (d_rot).
            //The d_index array maps each particle to its corresponding cell, and the total number of particles is given by N.
            RotationStep2<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_rot, d_index, N);
            //This line checks for any errors that might have occurred during the kernel launch using the cudaPeekAtLastError() function. If there are any errors, they will be recorded, and the error status will be reset for the next kernel launch.
            gpuErrchk( cudaPeekAtLastError() );
            //This line synchronizes the device and the host. It ensures that all previously issued CUDA calls are completed before continuing with the program execution. This synchronization is needed because the subsequent operations may depend on the results of the previous kernel execution.
            gpuErrchk( cudaDeviceSynchronize() );


            //Similar to the previous line, this one launches the RotationStep2 kernel again. 
            //However, this time it updates the velocities of MD particles (d_mdVx, d_mdVy, d_mdVz) based on the calculated rotation matrices (d_rot). The d_mdIndex array maps each MD particle to its corresponding cell, and the total number of MD particles is given by Nmd.
            RotationStep2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_rot, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //The kernel is responsible for updating the cell energy (d_e) due to the velocity changes of regular particles. It uses the updated particle velocities (d_vx, d_vy, d_vz) and the d_index array that maps each particle to its corresponding cell. The total number of particles is given by N, 
            //and the last argument 1 is likely a parameter specifying the mass of particles.
            E_cell<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_e, d_index, N, 1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            // Similar to the previous line, this one launches the E_cell kernel again. However, this time it updates the cell energy (d_e) due to the velocity changes of MD particles
            //The total number of MD particles is given by Nmd
            E_cell<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_e, d_mdIndex, Nmd , density);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // This kernel calculates the scalefactor fo each cell to use it in UpdateVelocity kernel.
            MBS<<<grid_size,blockSize>>>(d_scalefactor,d_n,d_e,devStates, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_ux, d_uy, d_uz, d_scalefactor, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_ux, d_uy, d_uz, d_scalefactor, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
}


////no slip collision part , must contain virtual particles:




__device__ void warp_Reduce_intt(volatile int *ssdata, int tid) {
    ssdata[tid] += ssdata[tid + 32];
    ssdata[tid] += ssdata[tid + 16];
    ssdata[tid] += ssdata[tid + 8];
    ssdata[tid] += ssdata[tid + 4];
    ssdata[tid] += ssdata[tid + 2];
    ssdata[tid] += ssdata[tid + 1];
}
__device__ void warp_Reduce_double(volatile double *ssdata, int tid) {
    ssdata[tid] += ssdata[tid + 32];
    ssdata[tid] += ssdata[tid + 16];
    ssdata[tid] += ssdata[tid + 8];
    ssdata[tid] += ssdata[tid + 4];
    ssdata[tid] += ssdata[tid + 2];
    ssdata[tid] += ssdata[tid + 1];
}

__global__ void reduceKernel_int(int *input, int *output, int N) {
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
        warp_Reduce_intt(sssdata_int, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sssdata_int[0];
    }
}

__global__ void reduceKernel_double(double *input, double *output, int N) {
    extern __shared__ double sssdata_double[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sssdata_double[tid] = (i < N) ? input[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sssdata_double[tid] += sssdata_double[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_Reduce_double(sssdata_double, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sssdata_double[0];
    }
}

__global__ void MeanNumCell(int *index, int *n, double *m, double mass, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];//Retrieves the index idxx of the cell to which the particle belongs.
        //Atomically increments the counters n[idxx] and m[idxx] by 1 and mass, respectively. 
        //These counters are used to keep track of the number of particles and the total mass within each cell.
        //if(index[tid]>307918)  printf(" index[%i]=%f an error\n", tid, index[tid]);
        atomicAdd(&n[idxx] , 1 );
        atomicAdd(&m[idxx], mass);
    }
}



__global__ void noslip_MeanVelCell(double* ux, double* vx,double* uy, double* vy,double* uz, double* vz,int* index, double mass, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        const unsigned int idxx = index[tid];//Retrieves the index idxx of the cell to which the particle belongs.
        //Multiplies the velocity components (vx, vy, vz) of the particle by its mass, storing the result in a temporary variable tmp.
        double tmp =vx[tid] *mass; 
        //Atomically adds the value of tmp to the corresponding velocity component sum (ux, uy, uz) of the cell idxx.
        atomicAdd(&ux[idxx] , tmp );
        tmp =vy[tid] *mass;
        atomicAdd(&uy[idxx] , tmp );
        tmp =vz[tid] *mass;
        atomicAdd(&uz[idxx] , tmp );
        //Atomically increments the counters n[idxx] and m[idxx] by 1 and mass, respectively. 
        //These counters are used to keep track of the number of particles and the total mass within each cell.
       
    }
}  //Output: The ux, uy, uz, n, and m arrays will be updated with the calculated mean velocities and masses for each cell.

// Kernel to initialize cuRAND states
__global__ void initializeCurandStates(curandState *states, unsigned long long seed, int Nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<Nc){
        curand_init(seed, tid, 0, &states[tid]);
    }
}


__device__ double boxMullerTransform(curandState *state) {
    double u1 = curand_uniform(state);
    double u2 = curand_uniform(state);
    return sqrt(abs(-2.0 * log(u1))) * cos(2.0 * M_PI * u2);
}

__global__ void createNormalDistributions(double *d_ux, double *d_uy, double *d_uz, double *N_avg, double mass, int *d_n, double *variance, int Nc, double *a_x, double *a_y, double *a_z, curandState *state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Nc) {
        double mean_x = d_ux[tid]; // Use the mean from d_ux[tid]
        double mean_y = d_uy[tid];
        double mean_z = d_uz[tid];
        variance[tid] = abs((*N_avg - d_n[tid])/mass);//*(kT = 1)
        double std_dev = sqrt(variance[tid]); // Calculate standard deviation from variance
        curandState localState = state[tid];
        a_x[tid] = mean_x + std_dev * boxMullerTransform(&localState); // Generate normally distributed random number
        a_y[tid] = mean_y + std_dev * boxMullerTransform(&localState); // Generate normally distributed random number
        a_z[tid] = mean_z + std_dev * boxMullerTransform(&localState); // Generate normally distributed random number
        state[tid] = localState; // Update state
    }
}


/*__global__ void createNormalDistributions(double *d_ux, double *d_uy, double *d_uz, double *N_avg, double mass, int *d_n, double *variance, std::normal_distribution<double> *normalDistributions_x, std::normal_distribution<double> *normalDistributions_y,std::normal_distribution<double> *normalDistributions_z, int Nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Nc) {
        variance[tid] = (*N_avg - d_n[tid])/mass;//*(kT = 1)
        normalDistributions_x[tid] = std::normal_distribution<double>(d_ux[tid], variance[tid]);
        normalDistributions_y[tid] = std::normal_distribution<double>(d_uy[tid], variance[tid]);
        normalDistributions_z[tid] = std::normal_distribution<double>(d_uz[tid], variance[tid]);
    }
}*/




__global__ void virtualMassiveParticle(double *d_ux, double *d_uy, double *d_uz, double *M_avg, double *N_avg, double *a_x, double *a_y, double *a_z, double mass , int density, int *d_n, int Nc){

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
       
       
        
        if (tid<Nc){
            if (*N_avg-d_n[tid] > 0){

                d_ux[tid] += (*N_avg-d_n[tid])*mass*a_x[tid];
                d_uy[tid] += (*N_avg-d_n[tid])*mass*a_y[tid];
                d_uz[tid] += (*N_avg-d_n[tid])*mass*a_z[tid];
            }
        }


}




__host__ void noslip_MPCD_MD_collision(double *d_vx ,double *d_vy ,double *d_vz , int *d_index,
double *d_mdVx ,double *d_mdVy,double *d_mdVz , int *d_mdIndex,
double  *d_ux ,double *d_uy ,double *d_uz ,
double *d_e ,double *d_scalefactor, int *d_n , double *d_m,
double *d_rot, double *d_theta, double *d_phi ,
int N , int Nmd, int Nc,
curandState *devStates, int grid_size, int *dn_tot, double *N_avg, int *sumblock_n, double *dm_tot, double *M_avg, double *sumblock_m,
double *a_x, double *a_y, double *a_z, double *variance, curandState *States)
{

            int shared_mem_size = 3 * blockSize * sizeof(double);
            //This launches the MakeCellReady kernel with the specified grid size and block size.
            //The kernel resets cell properties such as mean velocity (d_ux, d_uy, d_uz), energy (d_e), and count (d_n, d_m) to zero for all cells (Nc).
            MakeCellReady<<<grid_size,blockSize>>>(d_ux , d_uy, d_uz ,d_e, d_n, d_m, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() ); 

            //The particle count (d_n) and mass (d_m) arrays are updated for each cell (N is the total number of particles).
            //a kernel to calculate the mean number of mpcd particles in each cell
            MeanNumCell<<<grid_size,blockSize>>>(d_index, d_n , d_m, 1 ,N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            MeanNumCell<<<grid_size,blockSize>>>(d_mdIndex, d_n ,d_m , density ,Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //checkfunction1<<<grid_size,blockSize>>>(d_n, d_m , Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //checkfunction2<<<grid_size,blockSize>>>(d_n, d_m , d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            reduceKernel_int<<<grid_size,blockSize,shared_mem_size>>>(d_n, sumblock_n, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );  

            reduceKernel_double<<<grid_size,blockSize,shared_mem_size>>>(d_m, sumblock_m, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );  

            double block_sum_dn[grid_size];
            double block_sum_dm[grid_size];  

            cudaMemcpy(block_sum_dn , sumblock_n, grid_size*sizeof(int), cudaMemcpyDeviceToHost);  
            cudaMemcpy(block_sum_dm , sumblock_m, grid_size*sizeof(double), cudaMemcpyDeviceToHost); 

            int *hn_tot = (int*)malloc(sizeof(int));
            double *hm_tot = (double*)malloc(sizeof(double));
            double *h_N_avg = (double*)malloc(sizeof(double));
            double *h_M_avg= (double*)malloc(sizeof(double));

            *hn_tot=0;
            *hm_tot = 0.0;
            *h_N_avg=0;
            *h_M_avg=0.0;
            



            for (int j = 0; j < grid_size; j++)
        {
            *hn_tot += block_sum_dn[j];
            *hm_tot += block_sum_dm[j];
            
        }
            *h_N_avg = *hn_tot / Nc ; //calculate the average number of particles in cells.
            *h_M_avg = *hm_tot / Nc ; //calculate the average number of particles in cells.


            cudaMemcpy(dn_tot , hn_tot, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dm_tot , hm_tot, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(N_avg , h_N_avg, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(M_avg , h_M_avg, sizeof(double), cudaMemcpyHostToDevice);

            //This launches the MeanVelCell kernel with the specified grid size and block size.
            //The kernel adds the velocities of MPCD particles multipied by their mass within each cell based on their individual velocities (d_vx, d_vy, d_vz) to finally calculate (d_ux, d_uy, d_uz). 
            //The d_index array maps each particle to its corresponding cell. 
            //The result is updated in the d_ux, d_uy, and d_uz arrays.
            noslip_MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_vx , d_uy, d_vy, d_uz, d_vz, d_index, 1 ,N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //This launches the MeanVelCell kernel again, but this time it adds the velocities of MD particles within each cell multipled by their mass to d_ux, d_uy and d_uz.
            // The MD particle velocities are provided in the d_mdVx, d_mdVy, and d_mdVz arrays, and the d_mdIndex array maps each MD particle to its corresponding cell.
            // The result is updated in the d_ux, d_uy, and d_uz arrays (Nmd is the total number of MD particles).
            noslip_MeanVelCell<<<grid_size,blockSize>>>(d_ux , d_mdVx , d_uy, d_mdVy, d_uz, d_mdVz, d_mdIndex, density ,Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            ///////////////////////////////////virtual particle part:
            unsigned long long seed = 1234; // Choose a seed
            initializeCurandStates<<<grid_size, blockSize>>>(States, seed, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            createNormalDistributions<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, N_avg, 1, d_n, variance, Nc, a_x, a_y, a_z, States);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            virtualMassiveParticle<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, M_avg, N_avg, a_x, a_y, a_z, 1, density, d_n, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //////////////////////////////////////////////////////////

            //This launches the RotationStep1 kernel with the specified grid size and block size.
            // The kernel calculates the rotation matrices (d_rot) for each cell based on the angle values (d_phi, d_theta) and the mass (d_m) of particles in each cell.
            // The number of cells is given by Nc.
            RotationStep1<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_rot, d_m, d_phi, d_theta, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // The kernel calculates the relative velocities between particles and their corresponding cell mean velocities.
            // It uses the previously computed mean velocities (d_ux, d_uy, d_uz) and particle velocities (d_vx, d_vy, d_vz). The d_index array maps each particle to its corresponding cell. 
            //The total number of particles is given by N.
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_vx, d_vy, d_vz, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //This launches the relativeVelocity kernel again, but this time it calculates the relative velocities between MD particles and their corresponding cell mean velocities
            relativeVelocity<<<grid_size,blockSize>>>(d_ux, d_uy, d_uz, d_n, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //The kernel is responsible for updating the velocities of regular particles (d_vx, d_vy, d_vz) based on the calculated rotation matrices (d_rot).
            //The d_index array maps each particle to its corresponding cell, and the total number of particles is given by N.
            RotationStep2<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_rot, d_index, N);
            //This line checks for any errors that might have occurred during the kernel launch using the cudaPeekAtLastError() function. If there are any errors, they will be recorded, and the error status will be reset for the next kernel launch.
            gpuErrchk( cudaPeekAtLastError() );
            //This line synchronizes the device and the host. It ensures that all previously issued CUDA calls are completed before continuing with the program execution. This synchronization is needed because the subsequent operations may depend on the results of the previous kernel execution.
            gpuErrchk( cudaDeviceSynchronize() );


            //Similar to the previous line, this one launches the RotationStep2 kernel again. 
            //However, this time it updates the velocities of MD particles (d_mdVx, d_mdVy, d_mdVz) based on the calculated rotation matrices (d_rot). The d_mdIndex array maps each MD particle to its corresponding cell, and the total number of MD particles is given by Nmd.
            RotationStep2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_rot, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            //The kernel is responsible for updating the cell energy (d_e) due to the velocity changes of regular particles. It uses the updated particle velocities (d_vx, d_vy, d_vz) and the d_index array that maps each particle to its corresponding cell. The total number of particles is given by N, 
            //and the last argument 1 is likely a parameter specifying the mass of particles.
            E_cell<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_e, d_index, N, 1);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            // Similar to the previous line, this one launches the E_cell kernel again. However, this time it updates the cell energy (d_e) due to the velocity changes of MD particles
            //The total number of MD particles is given by Nmd
            E_cell<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_e, d_mdIndex, Nmd , density);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // This kernel calculates the scalefactor fo each cell to use it in UpdateVelocity kernel.
            MBS<<<grid_size,blockSize>>>(d_scalefactor,d_n,d_e,devStates, Nc);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, d_ux, d_uy, d_uz, d_scalefactor, d_index, N);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            UpdateVelocity<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_ux, d_uy, d_uz, d_scalefactor, d_mdIndex, Nmd);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
}
