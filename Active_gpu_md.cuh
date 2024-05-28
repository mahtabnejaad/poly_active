//#include "center_of_mass.cuh"

__global__ void tangential_vectors(double *mdX_v, double *mdY_v , double *mdZ_v ,
double *ex_v , double *ey_v , double *ez_v, 
double *L_v,int size_v , double ux_v, int mass_v, double real_time_v, int m_v , int topology_v) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    //int ID=0;

    if (tid<size_v)
    {
      
        int loop = int(tid/m_v);
        //if (tid == m-1)   printf("loop%i",loop);
        int ID = tid % (m_v);
        //printf("*%i",ID);
        //printf("tid%i",tid);
        double a[3];
        if (ID == (m_v-1))
        {
           
            LeeEdwNearestImage(mdX_v[tid],mdY_v[tid],mdZ_v[tid],mdX_v[m_v*loop],mdY_v[m_v*loop],mdZ_v[m_v*loop],a,L_v,ux_v,real_time_v);
            
        }
        else if (ID < (m_v-1))
        {
           
            LeeEdwNearestImage(mdX_v[tid],mdY_v[tid],mdZ_v[tid],mdX_v[tid+1],mdY_v[tid+1],mdZ_v[tid+1],a,L_v,ux_v,real_time_v);
        }
        else 
        {
            //printf("errrooooor");
        }
        double a_sqr=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
        double a_root=sqrt(a_sqr);//length of the vector between two adjacent monomers. 

        //tangential unit vector components :
        ex_v[tid] = a[0]/a_root;
        ey_v[tid] = a[1]/a_root;
        ez_v[tid] = a[2]/a_root;
       
        //printf("ex_v=%f\n",ex_v[tid]);
       // printf("ey_v=%f\n",ey_v[tid]);
        //printf("ez_v=%f\n",ez_v[tid]);
    


    }
}
// a kernel to put active forces on the polymer in an specific way that can be changes as you wish
__global__ void SpecificOrientedForce(double *mdX, double *mdY, double *mdZ, double real_time,double u0, int size, double *fa_kx, double *fa_ky, double *fa_kz,double *fb_kx, double *fb_ky, double *fb_kz, double *gama_T, double Q, double u_scale)
{
 
    int tid = blockIdx.x*blockDim.x+threadIdx.x;//index of the particle in the system
    if (tid < size)
    {
        //printf("gama-T=%f\n", *gama_T);
        fa_kx[tid] = 200;
        fa_ky[tid] = 0.0;  //u_scale * sin(real_time) * *gama_T;
        fa_kz[tid] = 0.0;
        fb_kx[tid] = fa_kx[tid] * Q;
        fb_ky[tid] = fa_ky[tid] * Q;
        fb_kz[tid] = fa_kz[tid] * Q;

    }

    

}

//this kernel is used to sum array components on block level in a parallel way
__global__ void reduce_kernel(double *FF1 ,double *FF2 , double *FF3,
 double *AA1 ,double *AA2 , double *AA3,
  int size)
{
    //size= Nmd (or N )
    //we want to add all the tangential vectors' components in one axis and calculate the total fa in one axis.
    //(OR generally we want to add all the components of a 1D array to each other) 
    int tid = threadIdx.x; //tid represents the index of the thread within the block.
    int index = blockIdx.x * blockDim.x + threadIdx.x ;//index represents the global index of the element in the input (F1,F2 or F3) array that the thread is responsible for.
    extern __shared__ double ssssdata[];  // This declares a shared memory array sdata, which will be used for the reduction within the block
   

 
    if(index<size){
       
        // Load the value into shared memory
    //Each thread loads the corresponding element from the F1,F2 or F3 array into the shared memory array sdata. If the thread's index is greater than or equal to size, it loads a zero.
        ssssdata[tid] = (index < size) ? FF1[index] : 0.0; 
        __syncthreads();  // Synchronize threads within the block to ensure all threads have loaded their data into shared memory before proceeding.

        ssssdata[tid+size] = (index < size) ? FF2[index] : 0.0;
        __syncthreads();  // Synchronize threads within the block

        ssssdata[tid+2*size] = (index < size) ? FF3[index] : 0.0;
        __syncthreads();  // Synchronize threads within the block

        // Reduction in shared memory
        //This loop performs a binary reduction on the sdata array in shared memory.
        //The loop iteratively adds elements from sdata[tid + s] to sdata[tid], where s is halved in each iteration.
        //The threads cooperate to perform the reduction in parallel.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                ssssdata[tid] += ssssdata[tid + s];
                ssssdata[tid + size] += ssssdata[tid + size + s];
                ssssdata[tid + 2 * size] += ssssdata[tid + 2 * size + s];

            }
            __syncthreads();
        }
    
        // Store the block result in the result array
        //Only the first thread in the block performs this operation.
        //It stores the final reduced value of the block into A1, A2 or A3 array at the corresponding block index
        if (tid == 0)
        {
            AA1[blockIdx.x] = ssssdata[0];
            AA2[blockIdx.x] = ssssdata[size];
            AA3[blockIdx.x] = ssssdata[2*size];
  
            //printf("A1[blockIdx.x]=%f",AA1[blockIdx.x]);
            //printf("\nA2[blockIdx.x]=%f",AA2[blockIdx.x]);
            //printf("\nA3[blockIdx.x]=%f\n",AA3[blockIdx.x]);


        }
        __syncthreads();
        //printf("BLOCKSUM1[0]=%f\n",A1[0]);
        //printf("BLOCKSUM1[1]=%f\n",A1[1]);
    }
   
}


//a kernel to build a random 0 or 1 array of size Nmd   

__global__ void randomArray(int *random , int size, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size)
    {
        curandState state;
        curand_init(seed, tid, 0, &state);

        // Generate a random float between 0 and 1
        float random_float = curand_uniform(&state);

        // Convert the float to an integer (0 or 1)
        random[tid] = (random_float < 0.5f) ? 0 : 1;
    }
}


__global__ void choiceArray(int *flag, int size)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x ;
   if (tid<size)
   {
        if (tid%2 ==0) flag[tid] = 1;
        else flag[tid] = 0;



   } 



}   
__global__ void Active_calc_forces(double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass,double mass_fluid, int size, int N, double *gama_T,double u_scale){

    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    //calculating (-M/mN+MN(m))
    //***
    double Q= -mass/(size*mass+mass_fluid*N);
    //double Q = 1.0;
    if(tid<size){
        //printf("gama_T=%f\n",*fgama_T);
        //calculating active forces in each axis for each particle:
        fa_kx[tid]=ex[tid]*u_scale* *gama_T;
        fa_ky[tid]=ey[tid]*u_scale* *gama_T;
        fa_kz[tid]=ez[tid]*u_scale* *gama_T;
        Aa_kx[tid]=fa_kx[tid]/mass;
        Aa_ky[tid]=fa_ky[tid]/mass;
        Aa_kz[tid]=fa_kz[tid]/mass;

        //calculating backflow forces in each axis for each particle: k is the index for each particle. 
        fb_kx[tid]=fa_kx[tid]*Q;
        fb_ky[tid]=fa_ky[tid]*Q;
        fb_kz[tid]=fa_kz[tid]*Q;
        Ab_kx[tid]=fb_kx[tid]/mass;
        Ab_ky[tid]=fb_ky[tid]/mass;
        Ab_kz[tid]=fb_kz[tid]/mass;
    }

    //printf("gama_T=%f\n",*gama_T);

}



__global__ void totalActive_calc_acceleration(double *Ax, double *Ay, double *Az, double *Aa_kx, double *Aa_ky, double *Aa_kz, double *Ab_kx, double *Ab_ky, double *Ab_kz, int *random_array, double *Ax_tot, double *Ay_tot, double *Az_tot, int size){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    //here I added a randomness to the active and backflow forces exerting on the monomers. 
    //we can change this manually or we can replace any other function instead of random_array as we prefer.
    
    if(tid< size){

    
        Ax_tot[tid]=Ax[tid]+(Aa_kx[tid]+Ab_kx[tid])*random_array[tid]; 
        Ay_tot[tid]=Ay[tid]+(Aa_ky[tid]+Ab_ky[tid])*random_array[tid];
        Az_tot[tid]=Az[tid]+(Aa_kz[tid]+Ab_kz[tid])*random_array[tid];
    }
   





}

__global__ void random_tangential(double *ex, double *ey, double *ez, int *random_array, int size){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    if(tid<size){

        ex[tid]=ex[tid]*random_array[tid];
        ey[tid]=ey[tid]*random_array[tid];
        ez[tid]=ez[tid]*random_array[tid];


    }
}

__global__ void choice_tangential(double *ex, double *ey, double *ez, int *flag_array, int size){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<size) {
        ex[tid]=ex[tid]*flag_array[tid];
        ey[tid]=ey[tid]*flag_array[tid];
        ez[tid]=ez[tid]*flag_array[tid];
    }

}

__host__ void monomer_active_backward_forces(double *mdX, double *mdY , double *mdZ ,
double *Ax, double *Ay, double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, int mass, double *gama_T,
double *L, int size, int mass_fluid, double real_time, int m, int topology, int grid_size, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array,double u_scale)
{
    double Q = -mass/(size*mass+mass_fluid*N);
    //shared_mem_size: The amount of shared memory allocated per block for the reduction operation.
    int shared_mem_size = 3 * blockSize * sizeof(double);
    

    if (topology == 4) //size= 1 (Nmd = 1) only one particle exists.
    {
        double *gamaTT;
        cudaMalloc((void**)&gamaTT, sizeof(double));
        cudaMemcpy(gamaTT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);


        SpecificOrientedForce<<<grid_size,blockSize>>>(mdX, mdY, mdZ, real_time, u_scale, size, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, gamaTT, Q, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        double fax, fay, faz;
        cudaMemcpy(&fax ,fa_kx, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&fay ,fa_ky, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&faz ,fa_kz, sizeof(double), cudaMemcpyDeviceToHost);

        *fa_x= fax;
        *fa_y= fay;
        *fa_z= faz;
        *fb_x= fax * Q;
        *fb_y= fax * Q;
        *fb_z= fax * Q;

     
    cudaFree(gamaTT);
    }

    else
    {
        
        if (random_flag == 1)
        {

            //int shared_mem_size = 3 * blockSize * sizeof(double); // allocate shared memory for the intermediate reduction results.
            //printf("ex[0]%f\n",ex[0]);
            //calculating tangential vectors:
            tangential_vectors<<<grid_size,blockSize>>>(mdX, mdY, mdZ, ex, ey, ez, L, size, u_scale, mass, real_time, m, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

           
            double *gamaT;
            cudaMalloc((void**)&gamaT, sizeof(double));
            cudaMemcpy(gamaT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);
            //printf("gama_T=%f\n",*gama_T);
        
            //printf("88gama_T=%f\n",*gama_T);
            //forces calculations in a seperate kernel:
            Active_calc_forces<<<grid_size,blockSize>>>(fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz,
                    ex, ey, ez, u_scale, mass, mass_fluid, size, N, gamaT, u_scale);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

       
            //calling the random_array kernel:
            // **** I think I should define 3 different random arrays for each axis so I'm gonna apply this later
            randomArray<<<grid_size, blockSize>>>(random_array, size, seed);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            //calling the totalActive_calc_acceleration kernel:
            totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, random_array, Ax_tot, Ay_tot, Az_tot, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
    

            //calculating the sum of tangential vectors in each axis:
            //grid_size: The number of blocks launched in the grid.
            //block_size: The number of threads per block.

        
            random_tangential<<<grid_size,blockSize>>>(ex, ey, ez, random_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
   

            reduce_kernel<<<grid_size, blockSize, shared_mem_size>>>(ex, ey, ez, block_sum_ex, block_sum_ey, block_sum_ez, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
            }
            double sumx[grid_size];
            double sumy[grid_size];
            double sumz[grid_size];
            cudaMemcpy(sumx ,block_sum_ex, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumy ,block_sum_ey, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumz ,block_sum_ez, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            //printf("%lf",sumx[0]);

            //Perform the reduction on the host side to obtain the final sum.
            *fa_x = 0.0; 
            *fa_y = 0.0;
            *fa_z = 0.0;
            
            for (int i = 0; i < grid_size; i++)
            {
               
                *fa_x += sumx[i]* u_scale* *gama_T;
                *fa_y += sumy[i]* u_scale* *gama_T;
                *fa_z += sumz[i]* u_scale* *gama_T;

            }
            //printf("fa_x=%lf", *fa_x);
           
    
            //*fa_x=*fa_x* *gama_T*u_scale;
            //*fa_y=*fa_y* *gama_T*u_scale;
            //*fa_z=*fa_z* *gama_T*u_scale;
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            
            cudaFree(gamaT);
        }
        if(random_flag == 0)
        { //if(random_flag == 0){
            
            //int shared_mem_size = 3 * blockSize * sizeof(double); // allocate shared memory for the intermediate reduction results.
            //printf("ex[0]%f\n",ex[0]);
            //calculating tangential vectors:
            tangential_vectors<<<grid_size,blockSize>>>(mdX, mdY, mdZ, ex, ey, ez, L, size, ux, mass, real_time, m, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

           
            double *gamaT;
            cudaMalloc((void**)&gamaT, sizeof(double));
            cudaMemcpy(gamaT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);
            //printf("gama_T=%f\n",*gama_T);
        
            //printf("88gama_T=%f\n",*gama_T);
            //forces calculations in a seperate kernel:
            Active_calc_forces<<<grid_size,blockSize>>>(fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz,
                    ex, ey, ez, ux, mass, mass_fluid, size, N, gamaT, u_scale);

          
    
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

       

            choiceArray<<<grid_size,blockSize>>>(flag_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
   

            totalActive_calc_acceleration<<<grid_size,blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            choice_tangential<<<grid_size, blockSize>>>(ex, ey, ez, flag_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            reduce_kernel<<<grid_size,blockSize>>>(ex, ey, ez, block_sum_ex, block_sum_ey, block_sum_ez, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            cudaDeviceSynchronize();


            double sumx[grid_size];
            double sumy[grid_size];
            double sumz[grid_size];
            cudaMemcpy(sumx ,block_sum_ex, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumy ,block_sum_ey, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumz ,block_sum_ez, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            //printf("%lf",sumx[0]);

            //Perform the reduction on the host side to obtain the final sum.
            for (int i = 0; i < grid_size; i++)
            {
             
                *fa_x += sumx[i]* u_scale* *gama_T;
                *fa_y += sumy[i]* u_scale* *gama_T;
                *fa_z += sumz[i]* u_scale* *gama_T;

            }
            //printf("fa_x=%lf", *fa_x);
           
    
            //*fa_x=*fa_x* *gama_T*u_scale;
            //*fa_y=*fa_y* *gama_T*u_scale;
            //*fa_z=*fa_z* *gama_T*u_scale;
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            cudaFree(gamaT);
     
        }
  
    }
}

__global__ void Active_nb_b_interaction( 
double *NmdX, double *NmdY , double *NmdZ ,
double *Nfx , double *Nfy , double *Nfz, 
double *NL,int Nsize , double Nux, int Nmass, double Nreal_time, int Nm , int Ntopology)
{
    int size2 = Nsize*(Nsize); //size2 calculates the total number of particle pairs for the interaction.


    //In the context of the nb_b_interaction kernel, each thread is responsible for calculating the interaction between a pair of particles. The goal is to calculate the interaction forces between all possible pairs of particles in the simulation. To achieve this, the thread ID is mapped to particle indices.
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size2)
    {
        //ID1 and ID2 are calculated from tid to determine the indices of the interacting particles.
        //The combination of these calculations ensures that each thread ID is mapped to a unique pair of particle indices. This way, all possible pairs of particles are covered, and the interactions between particles can be calculated in parallel.
        int ID1 = int(tid /Nsize);//tid / size calculates how many "rows" of particles the thread ID represents. In other words, it determines the index of the first particle in the pair (ID1).
        int ID2 = tid%(Nsize);//tid % size calculates the remainder of the division of tid by size. This remainder corresponds to the index of the second particle in the pair (ID2)
        if(ID1 != ID2) //This condition ensures that the particle does not interact with itself. Interactions between a particle and itself are not considered
        {
        double r[3];
        //This line calculates the nearest image of particle positions in the periodic boundary conditions using the LeeEdwNearestImage function
        //The resulting displacement is stored in the r array.
        LeeEdwNearestImage(NmdX[ID1], NmdY[ID1], NmdZ[ID1] , NmdX[ID2] , NmdY[ID2] , NmdZ[ID2] , r,NL, Nm , Nreal_time);
        double r_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];//r_sqr calculates the squared distance between the particles.
        double f =0;//initialize the force to zero.

 
        //lennard jones:
       
        if (r_sqr < 1.258884)
        {
                double r8 = 1/r_sqr* 1/r_sqr; //r^{-4}
                r8 *= r8; //r^{-8}
                double r14 = r8 *r8; //r^{-16}
                r14 *= r_sqr; //r^{-14}
                f = 24 * (2 * r14 - r8);
        }
        
        //FENE:
        //This part of the code is responsible for calculating the interaction forces between particles based on the FENE (Finitely Extensible Nonlinear Elastic) potential. The FENE potential is often used to model polymer chains where bonds between particles cannot be stretched beyond a certain limit
        
        if (Ntopology == 1)
        {
            if (int(ID1/Nm) == int(ID2/Nm)) //checks if the interacting particles belong to the same chain (monomer). This is achieved by dividing the particle indices by m (monomer size) and checking if they are in the same division.
            {
                //check if the interacting particles are next to each other in the same chain. If they are, it calculates the FENE interaction contribution,
                if( ID2 - ID1 == 1 || ID2 - ID1 == Nm-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == Nm-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }   
        }
        
        //FENE:
        if (Ntopology == 2 || Ntopology == 3)
        {
            if (int(ID1/Nm) == int(ID2/Nm)) //similar conditions are checked for particles within the same chain
            {
                if( ID2 - ID1 == 1 || ID2 - ID1 == Nm-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == Nm-1 ) 
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }
            
            if (ID1==int(Nm/4) && ID2 ==Nm+int(3*Nm/4))
            {
                
                f -= 30/(1 - r_sqr/2.25);
            }
                
            if (ID2==int(Nm/4) && ID1 ==Nm+int(3*Nm/4))
            {
                f -= 30/(1 - r_sqr/2.25);
            }
        }
        f/=Nmass; //After the interaction forces are calculated (f), they are divided by the mass of the particles to obtain the correct acceleration.

        Nfx[tid] = f * r[0] ;
        Nfy[tid] = f * r[1] ;
        Nfz[tid] = f * r[2] ;
        }
    
        else
        {
            Nfx[tid] = 0;
            Nfy[tid] = 0;
            Nfz[tid] = 0;
        }
      

    }

}




__host__ void Active_calc_acceleration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz, 
double *Ax , double *Ay , double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T, 
double *L,int size ,int m ,int topology, double real_time, int grid_size, int mass_fluid, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot, double *fa_x, double *fa_y, double *fa_z,double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array, double u_scale)

{
  

    Active_nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux, mass, real_time , m , topology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sum_kernel<<<grid_size,blockSize>>>(Fx , Fy, Fz, Ax , Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //printf("**GAMA=%f\n",*agama_T);
    

    monomer_active_backward_forces(x, y ,z ,
    Ax , Ay, Az, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, ex, ey, ez, ux, mass, gama_T, 
    L, size , mass_fluid, real_time, m, topology, grid_size, N , random_array, seed , Ax_tot, Ay_tot, Az_tot, fa_x, fa_y, fa_z, fb_x, fb_y, fb_z, block_sum_ex, block_sum_ey, block_sum_ez, flag_array, u_scale);
    

    
}


//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void ActivevelocityVerletKernel2(double *mdVx , double *mdVy , double *mdVz,
double *mdAx , double *mdAy , double *mdAz,
 double h, int size)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < size)
    {
        mdVx[particleID] += 0.5 * h * mdAx[particleID];
        mdVy[particleID] += 0.5 * h * mdAy[particleID];
        mdVz[particleID] += 0.5 * h * mdAz[particleID];
    }
}

//first kernel: x+= hv(half time) + 0.5hha(new) ,v += 0.5ha(new)

__global__ void ActivevelocityVerletKernel1(double *mdX, double *mdY , double *mdZ , 
double *mdVx , double *mdVy , double *mdVz,
double *mdAx , double *mdAy , double *mdAz,
 double h, int size)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < size)
    {
        // Particle velocities are updated by half a time step, and particle positions are updated based on the new velocities.

        mdVx[particleID] += 0.5 * h * mdAx[particleID];
        mdVy[particleID] += 0.5 * h * mdAy[particleID];
        mdVz[particleID] += 0.5 * h * mdAz[particleID];

        mdX[particleID] = mdX[particleID] + h * mdVx[particleID] ;
        mdY[particleID] = mdY[particleID] + h * mdVy[particleID] ;
        mdZ[particleID] = mdZ[particleID] + h * mdVz[particleID] ;


    }
}
__global__ void gotoCMframe(double *X, double *Y, double *Z, double *Xcm,double *Ycm, double *Zcm, double *Vx, double *Vy, double *Vz, double *Vxcm,double *Vycm, double *Vzcm, int size){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid < size)
    {
        
        X[tid] = X[tid] - *Xcm;
        Y[tid] = Y[tid] - *Ycm;
        Z[tid] = Z[tid] - *Zcm;

        Vx[tid] = Vx[tid] - *Vxcm;
        Vy[tid] = Vy[tid] - *Vycm;
        Vz[tid] = Vz[tid] - *Vzcm;



    }
}

__global__ void backtoLabframe(double *X, double *Y, double *Z, double *Xcm,double *Ycm, double *Zcm, double *Vx, double *Vy, double *Vz, double *Vxcm,double *Vycm, double *Vzcm, int size){
    
        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        if (tid < size)
        {
            
            X[tid] = X[tid] + *Xcm;
            Y[tid] = Y[tid] + *Ycm;
            Z[tid] = Z[tid] + *Zcm;

            Vx[tid] = Vx[tid] - *Vxcm;
            Vy[tid] = Vy[tid] - *Vycm;
            Vz[tid] = Vz[tid] - *Vzcm;

        }
}

__host__ void Active_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_x, double *d_y, double *d_z,
    double *d_mdVx, double *d_mdVy, double *d_mdVz,
    double *d_vx, double *d_vy, double *d_vz,
    double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *dX_tot, double *dY_tot, double *dZ_tot,
    double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *dVx_tot, double *dVy_tot, double *dVz_tot,
    double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
    double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz,
    double *d_Fx, double *d_Fy, double *d_Fz,
    double *d_fa_kx, double *d_fa_ky, double *d_fa_kz,
    double *d_fb_kx, double *d_fb_ky, double *d_fb_kz,
    double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz,
    double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz,
    double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot,
    double *d_ex, double *d_ey, double *d_ez,
    double *h_fa_x, double *h_fa_y, double *h_fa_z,
    double *h_fb_x, double *h_fb_y, double *h_fb_z,
    double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md , int Nmd, int density, double *d_L , double ux, int grid_size, int shared_mem_size, int blockSize_, int grid_size_, int delta, 
    double real_time, int m, int N, double mass, double mass_fluid, double *gama_T, int *random_array, unsigned int seed, int topology, 
    double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, int *flag_array, double u_scale)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {

        CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, mdVx_tot, mdVy_tot, mdVz_tot, dVx_tot, dVy_tot, dVz_tot, grid_size, shared_mem_size, blockSize_, grid_size_, density, 1,
                Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

        //with this function call particles go to box's center of mass frame. 
        gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //firt velocity verlet step, in which particles' positions and velocities are updated.
        ActivevelocityVerletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_Ax_tot, d_Ay_tot, d_Az_tot , h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        //After updating particles' positions, a kernel named LEBC is called to apply boundary conditions to ensure that particles stay within the simulation box.
        LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //one can choose to have another kind of boundary condition , in this case it is nonslip in y z planes and (lees edwards) periodic in x plane. 
        //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx ,d_mdVy, d_mdVz, ux , d_L, real_time , Nmd);
        //gpuErrchk( cudaPeekAtLastError() );
        //gpuErrchk( cudaDeviceSynchronize() );

        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        //***
        Active_calc_acceleration( d_mdX ,d_mdY , d_mdZ , 
        d_Fx , d_Fy , d_Fz,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fa_ky, d_fa_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz,d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd , m , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);


        /*Active_nb_b_interaction<<<grid_size,blockSize>>>(d_mdX , d_mdY , d_mdZ, d_Fx , d_Fy , d_Fz ,d_L , Nmd , ux,density, real_time , m , topology);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/

        sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        /*monomer_active_backward_forces( d_mdX , d_mdY , d_mdZ,
        d_mdAx ,d_mdAy, d_mdAz ,d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez, ux, mass,gama_T, 
        d_L, Nmd , mass_fluid, real_time, m, topology,  grid_size, N , random_array, seed , d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );*/
      

        
        
        //velocityVerletKernel2 is called to complete the velocity Verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.

        //***
        ActivevelocityVerletKernel2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;


        
    }
}

