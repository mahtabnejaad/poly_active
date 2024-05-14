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

__global__ void randomArray(int *Arandom , int Asize, unsigned int Aseed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<Asize)
    {
        curandState state;
        curand_init(Aseed, tid, 0, &state);

        // Generate a random float between 0 and 1
        float random_float = curand_uniform(&state);

        // Convert the float to an integer (0 or 1)
        Arandom[tid] = (random_float < 0.5f) ? 0 : 1;
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
__global__ void Active_calc_forces(double *ffa_kx, double *ffa_ky, double *ffa_kz, double *ffb_kx, double *ffb_ky, double *ffb_kz,
double *fAa_kx, double *fAa_ky, double *fAa_kz,double *fAb_kx, double *fAb_ky, double *fAb_kz, double *fex, double *fey, double *fez, double fux, double fmass,double fmass_fluid, int fsize, int fN, double *fgama_T,double u_scale){

    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    //calculating (-M/mN+MN(m))
    //***
    double Q= -fmass/(fsize*fmass+fmass_fluid*fN);
    //double Q = 1.0;
    if(tid<fsize){
        //printf("gama_T=%f\n",*fgama_T);
        //calculating active forces in each axis for each particle:
        ffa_kx[tid]=fex[tid]*u_scale* *fgama_T;
        ffa_ky[tid]=fey[tid]*u_scale* *fgama_T;
        ffa_kz[tid]=fez[tid]*u_scale* *fgama_T;
        fAa_kx[tid]=ffa_kx[tid]/fmass;
        fAa_ky[tid]=ffa_ky[tid]/fmass;
        fAa_kz[tid]=ffa_kz[tid]/fmass;

        //calculating backflow forces in each axis for each particle: k is the index for each particle. 
        ffb_kx[tid]=ffa_kx[tid]*Q;
        ffb_ky[tid]=ffa_ky[tid]*Q;
        ffb_kz[tid]=ffa_kz[tid]*Q;
        fAb_kx[tid]=ffb_kx[tid]/fmass;
        fAb_ky[tid]=ffb_ky[tid]/fmass;
        fAb_kz[tid]=ffb_kz[tid]/fmass;
    }

    //printf("gama_T=%f\n",*fgama_T);

}



__global__ void totalActive_calc_acceleration(double *tAx, double *tAy, double *tAz, double *tAa_kx, double *tAa_ky, double *tAa_kz, double *tAb_kx, double *tAb_ky, double *tAb_kz, int *trandom_array, double *tAx_tot, double *tAy_tot, double *tAz_tot, int tsize){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    //here I added a randomness to the active and backflow forces exerting on the monomers. 
    //we can change this manually or we can replace any other function instead of random_array as we prefer.
    
    if(tid< tsize){

    
        tAx_tot[tid]=tAx[tid]+(tAa_kx[tid]+tAb_kx[tid])*trandom_array[tid]; 
        tAy_tot[tid]=tAy[tid]+(tAa_ky[tid]+tAb_ky[tid])*trandom_array[tid];
        tAz_tot[tid]=tAz[tid]+(tAa_kz[tid]+tAb_kz[tid])*trandom_array[tid];
    }
   





}

__global__ void random_tangential(double *rex, double *rey, double *rez, int *rrandom_array, int rsize){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    if(tid<rsize){

        rex[tid]=rex[tid]*rrandom_array[tid];
        rey[tid]=rey[tid]*rrandom_array[tid];
        rez[tid]=rez[tid]*rrandom_array[tid];


    }
}

__global__ void choice_tangential(double *cex, double *cey, double *cez, int *cflag_array, int size){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<size) {
        cex[tid]=cex[tid]*cflag_array[tid];
        cey[tid]=cey[tid]*cflag_array[tid];
        cez[tid]=cez[tid]*cflag_array[tid];
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




__host__ void Active_calc_accelaration( double *ax ,double *ay , double *az , 
double *aFx , double *aFy , double *aFz, 
double *aAx , double *aAy , double *aAz,double *afa_kx, double *afa_ky, double *afa_kz, double *afb_kx, double *afb_ky, double *afb_kz,
double *aAa_kx, double *aAa_ky, double *aAa_kz,double *aAb_kx, double *aAb_ky, double *aAb_kz, double *aex, double *aey, double *aez, double aux, double amass, double *agama_T, 
double *aL,int asize ,int am ,int atopology, double areal_time, int agrid_size, int amass_fluid, int aN, int *arandom_array, unsigned int aseed, double *aAx_tot, double *aAy_tot, double *aAz_tot, double *afa_x, double *afa_y, double *afa_z,double *afb_x, double *afb_y, double *afb_z, double *ablock_sum_ex, double *ablock_sum_ey, double *ablock_sum_ez, int *aflag_array, double u_scale)

{
  

    Active_nb_b_interaction<<<agrid_size,blockSize>>>(ax , ay , az, aFx , aFy , aFz ,aL , asize , aux, amass, areal_time , am , atopology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sum_kernel<<<agrid_size,blockSize>>>(aFx ,aFy,aFz, aAx ,aAy, aAz, asize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //printf("**GAMA=%f\n",*agama_T);
    

    monomer_active_backward_forces(ax, ay ,az ,
    aAx , aAy, aAz,afa_kx, afa_ky, afa_kz,afb_kx, afb_ky, afb_kz,aAa_kx, aAa_ky, aAa_kz, aAb_kx, aAb_ky, aAb_kz, aex, aey, aez, aux, amass, agama_T, 
    aL, asize , amass_fluid, areal_time, am, atopology,  agrid_size, aN , arandom_array, aseed , aAx_tot, aAy_tot, aAz_tot, afa_x, afa_y, afa_z, afb_x, afb_y, afb_z, ablock_sum_ex, ablock_sum_ey, ablock_sum_ez, aflag_array, u_scale);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
}


//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void ActivevelocityVerletKernel2(double *VmdVx , double *VmdVy , double *VmdVz,
double *VmdAx , double *VmdAy , double *VmdAz,
 double Vh, int Vsize)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Vsize)
    {
        VmdVx[particleID] += 0.5 * Vh * VmdAx[particleID];
        VmdVy[particleID] += 0.5 * Vh * VmdAy[particleID];
        VmdVz[particleID] += 0.5 * Vh * VmdAz[particleID];
    }
}

//first kernel: x+= hv(half time) + 0.5hha(new) ,v += 0.5ha(new)

__global__ void ActivevelocityVerletKernel1(double *WmdX, double *WmdY , double *WmdZ , 
double *WmdVx , double *WmdVy , double *WmdVz,
double *WmdAx , double *WmdAy , double *WmdAz,
 double Wh, int Wsize)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Wsize)
    {
        // Particle velocities are updated by half a time step, and particle positions are updated based on the new velocities.

        WmdVx[particleID] += 0.5 * Wh * WmdAx[particleID];
        WmdVy[particleID] += 0.5 * Wh * WmdAy[particleID];
        WmdVz[particleID] += 0.5 * Wh * WmdAz[particleID];

        WmdX[particleID] = WmdX[particleID] + Wh * WmdVx[particleID] ;
        WmdY[particleID] = WmdY[particleID] + Wh * WmdVy[particleID] ;
        WmdZ[particleID] = WmdZ[particleID] + Wh * WmdVz[particleID] ;


    }
}
__global__ void gotoCMframe(double *g_X, double *g_Y, double *g_Z, double *gXcm,double *gYcm, double *gZcm , int size_g){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid < size_g)
    {
        
        g_X[tid] = g_X[tid] - *gXcm;
        g_Y[tid] = g_Y[tid] - *gYcm;
        g_Z[tid] = g_Z[tid] - *gZcm;



    }
}

__global__ void backtoLabframe(double *b_X, double *b_Y, double *b_Z, double *bXcm,double *bYcm, double *bZcm , int size_b){
    
        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        if (tid < size_b)
        {
            
            b_X[tid] = b_X[tid] + *bXcm;
            b_Y[tid] = b_Y[tid] + *bYcm;
            b_Z[tid] = b_Z[tid] + *bZcm;
        }
}

__host__ void Active_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdVx, double *d_mdVy, double *d_mdVz,
    double *d_mdAx, double *d_mdAy, double *d_mdAz,
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
    double h_md ,int Nmd, int density, double *d_L ,double ux,int grid_size ,int delta, double real_time, int m, int N, double mass, double mass_fluid, double *gama_T, int *random_array, unsigned int seed, int topology, double *Xcm, double *Ycm, double *Zcm, int *flag_array, double u_scale)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {


        gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        ActivevelocityVerletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_Ax_tot, d_Ay_tot, d_Az_tot , h_md,Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        //After updating particle positions, a kernel named LEBC is called to apply boundary conditions to ensure that particles stay within the simulation box.
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
        Active_calc_accelaration( d_mdX ,d_mdY , d_mdZ , 
        d_Fx , d_Fy , d_Fz,
        d_Ax_tot , d_Ay_tot , d_Az_tot, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fa_ky, d_fa_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz,d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd , m , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        Active_nb_b_interaction<<<grid_size,blockSize>>>(d_mdX , d_mdY , d_mdZ, d_Fx , d_Fy , d_Fz ,d_L , Nmd , ux,density, real_time , m , topology);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        monomer_active_backward_forces( d_mdX , d_mdY , d_mdZ,
        d_mdAx ,d_mdAy, d_mdAz ,d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez, ux, mass,gama_T, 
        d_L, Nmd , mass_fluid, real_time, m, topology,  grid_size, N , random_array, seed , d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
      

        
        
        //velocityVerletKernel2 is called to complete the velocity Verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.

        //***
        ActivevelocityVerletKernel2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;


        
    }
}

