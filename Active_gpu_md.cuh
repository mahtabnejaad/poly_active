

__global__ void tangential_vectors(double *mdX, double *mdY , double *mdZ ,
double *ex , double *ey , double *ez, 
double *L, int size, double ux, double mass, double real_time, int m, int topology) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    //int ID=0;

    if (tid<size)
    {
      
        int loop = int(tid/m);
        //if (tid == m-1)   printf("loop%i",loop);
        int ID = tid % (m);
        //printf("*%i",ID);
        //printf("tid%i",tid);
        double a[3];
        if (ID == (m-1))
        {
           
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid], mdX[m*loop], mdY[m*loop], mdZ[m*loop], a, L, ux, real_time);
            
        }
        else if (ID < (m-1))
        {
           
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid], mdX[tid+1], mdY[tid+1], mdZ[tid+1], a, L, ux, real_time);
        }
        else 
        {
            //printf("errrooooor");
        }
        double a_sqr=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
        double a_root=sqrt(a_sqr);//length of the vector between two adjacent monomers. 

         //tangential unit vector components :
        if (a_root != 0.0){
            ex[tid] = a[0]/a_root;
            ey[tid] = a[1]/a_root;
            ez[tid] = a[2]/a_root;
        }
        else{
            ex[tid] = a[0];
            ey[tid] = a[1];
            ez[tid] = a[2];
        }
    


    }
}
// a kernel to put active forces on the polymer in an specific way that can be changes as you wish
__global__ void SpecificOrientedForce(double *mdX, double *mdY, double *mdZ, double real_time,double u0, int size, double *fa_kx, double *fa_ky, double *fa_kz,double *fb_kx, double *fb_ky, double *fb_kz, double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *gama_T, double *Q, double mass, double u_scale)
{
 
    int tid = blockIdx.x*blockDim.x+threadIdx.x;//index of the particle in the system
    if (tid < size)
    {
        //printf("gama-T=%f\n", *gama_T);
        fa_kx[tid] = 1.0;
        fa_ky[tid] = 0.0;  //u_scale * sin(real_time) * *gama_T;
        fa_kz[tid] = 0.0;
        fb_kx[tid] = fa_kx[tid] * *Q;
        fb_ky[tid] = fa_ky[tid] * *Q;
        fb_kz[tid] = fa_kz[tid] * *Q;

        Aa_kx[tid]=fa_kx[tid]/mass;
        Aa_ky[tid]=fa_ky[tid]/mass;
        Aa_kz[tid]=fa_kz[tid]/mass;
        Ab_kx[tid]=fb_kx[tid]/mass;
        Ab_ky[tid]=fb_ky[tid]/mass;
        Ab_kz[tid]=fb_kz[tid]/mass;

        printf("\n *******Aa_kx[%i]=%f\n",tid, Aa_kx[tid]);

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
        //if (tid%2 ==0) flag[tid] = 1;
        //else flag[tid] = 0;
        flag[tid] = 1;


   } 



}   
__global__ void Active_calc_forces(double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double mass_fluid, int size, int N, double *gama_T,double u_scale){

    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    //calculating (-M/mN+MN(m))
    //***
   double Q;
    Q = -mass/(size*mass+mass_fluid*N);

    
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

        //printf("Q=%f\n", Q);
        //printf("Aa_kx[%i]=%f, Aa_ky[%i]=%f, Aa_kz[%i]=%f\n", tid, Aa_kx[tid], tid, Aa_ky[tid], tid, Aa_kz[tid]);
        //printf("Ab_kx[%i]=%f, Ab_ky[%i]=%f, Ab_kz[%i]=%f\n", tid, Ab_kx[tid], tid, Ab_ky[tid], tid, Ab_kz[tid]);
        //printf("fa_kx[%i]=%f, fa_ky[%i]=%f, fa_kz[%i]=%f\n", tid, fa_kx[tid], tid, fa_ky[tid], tid, fa_kz[tid]);
        //printf("fb_kx[%i]=%f, fb_ky[%i]=%f, fb_kz[%i]=%f\n", tid, fb_kx[tid], tid, fb_ky[tid], tid, fb_kz[tid]);
        
    }

    //printf("gama_T=%f\n",*gama_T);

}



__global__ void totalActive_calc_acceleration(double *Ax, double *Ay, double *Az, double *Aa_kx, double *Aa_ky, double *Aa_kz, double *Ab_kx, double *Ab_ky, double *Ab_kz, int *random_array, double *Ax_tot, double *Ay_tot, double *Az_tot, int size, int topology){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    //here I added a randomness to the active and backflow forces exerting on the monomers. 
    //we can change this manually or we can replace any other function instead of random_array as we prefer.
    
    if(tid< size){

        if(topology == 4){
            
            Ax_tot[tid]=Ax[tid] + Aa_kx[tid] + Ab_kx[tid]; 
            Ay_tot[tid]=Ay[tid] + Aa_ky[tid] + Ab_ky[tid];
            Az_tot[tid]=Az[tid] + Aa_kz[tid] + Ab_kz[tid];
            //printf("Aa_kx[%i]=%f, Aa_ky[%i]=%f, Aa_kz[%i]=%f\n", tid, Aa_kx[tid], tid, Aa_ky[tid], tid, Aa_kz[tid]);

        }
        else{

            Ax_tot[tid]=Ax[tid]+(Aa_kx[tid]+Ab_kx[tid])*random_array[tid]; 
            Ay_tot[tid]=Ay[tid]+(Aa_ky[tid]+Ab_ky[tid])*random_array[tid];
            Az_tot[tid]=Az[tid]+(Aa_kz[tid]+Ab_kz[tid])*random_array[tid];
            //printf("Aa_kx[%i]=%f, Aa_ky[%i]=%f, Aa_kz[%i]=%f\n", tid, Aa_kx[tid], tid, Aa_ky[tid], tid, Aa_kz[tid]);
            //printf("Ax_tot[%i]=%f, Ay_tot[%i]=%f, Az_tot[%i]=%f\n", tid, Ax_tot[tid], tid, Ay_tot[tid], tid, Az_tot[tid]);
        }
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
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T,
double *L, int size, double mass_fluid, double real_time, int m, int topology, int grid_size, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array,double u_scale)
{
    double Q;
    Q = -mass/(size*mass+mass_fluid*N);
    printf("Q=%f\n", Q);

    double *d_Q;
    cudaMalloc((void**)&d_Q, sizeof(double));
    cudaMemcpy(d_Q, &Q, sizeof(double), cudaMemcpyHostToDevice);
    //shared_mem_size: The amount of shared memory allocated per block for the reduction operation.
    int shared_mem_size = 3 * blockSize * sizeof(double);
    

    if (topology == 4) //size= 1 (Nmd = 1) only one particle exists.
    {
        double *gamaTT;
        cudaMalloc((void**)&gamaTT, sizeof(double));
        cudaMemcpy(gamaTT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);


        SpecificOrientedForce<<<grid_size,blockSize>>>(mdX, mdY, mdZ, real_time, u_scale, size, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, gamaTT, d_Q, mass, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //calling the totalActive_calc_acceleration kernel:
        totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, 1, topology);
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
            totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, random_array, Ax_tot, Ay_tot, Az_tot, size, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
    

            //calculating the sum of tangential vectors in each axis:
            //grid_size: The number of blocks launched in the grid.
            //block_size: The number of threads per block.

        
            random_tangential<<<grid_size,blockSize>>>(ex, ey, ez, random_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
   

            reduce_kernel_var<<<grid_size, blockSize, shared_mem_size>>>(ex, ey, ez, block_sum_ex, block_sum_ey, block_sum_ez, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            cudaDeviceSynchronize();
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                //fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
            }
            double *sumx;
            double *sumy;
            double *sumz;
            sumx = (double *)malloc(sizeof(double) * grid_size);
            sumy = (double *)malloc(sizeof(double) * grid_size);
            sumz = (double *)malloc(sizeof(double) * grid_size);

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
           
    
        
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            
            cudaFree(gamaT);
            free(sumx);  free(sumy);  free(sumz);

        }
        if(random_flag == 0)
        { 
            
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
            //printf("mmmm = %i\n", mass);
            //forces calculations in a seperate kernel:
            Active_calc_forces<<<grid_size,blockSize>>>(fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz,
                    ex, ey, ez, ux, mass, mass_fluid, size, N, gamaT, u_scale);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

       

            choiceArray<<<grid_size,blockSize>>>(flag_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
   

            totalActive_calc_acceleration<<<grid_size,blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, size, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            choice_tangential<<<grid_size, blockSize>>>(ex, ey, ez, flag_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            reduce_kernel_var<<<grid_size,blockSize>>>(ex, ey, ez, block_sum_ex, block_sum_ey, block_sum_ez, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );


            cudaDeviceSynchronize();


            double *sumx;
            double *sumy;
            double *sumz;
            sumx = (double *)malloc(sizeof(double) * grid_size);
            sumy = (double *)malloc(sizeof(double) * grid_size);
            sumz = (double *)malloc(sizeof(double) * grid_size);

            cudaMemcpy(sumx ,block_sum_ex, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumy ,block_sum_ey, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sumz ,block_sum_ez, grid_size*sizeof(double), cudaMemcpyDeviceToHost);
            //printf("%lf",sumx[0]);

            *fa_x = 0.0; 
            *fa_y = 0.0;
            *fa_z = 0.0;

            //Perform the reduction on the host side to obtain the final sum.
            for (int i = 0; i < grid_size; i++)
            {
             
                *fa_x += sumx[i]* u_scale* *gama_T;
                *fa_y += sumy[i]* u_scale* *gama_T;
                *fa_z += sumz[i]* u_scale* *gama_T;

            }
            //printf("fa_x=%lf", *fa_x);
           
    
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            cudaFree(gamaT);
            free(sumx);  free(sumy);  free(sumz);
     
        }
  
    }
}

__global__ void Active_nb_b_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *L,int size , double ux, double mass, double real_time, int m , int topology, double K_FENE, double K_bend)
{
    int size2 = size*(size); //size2 calculates the total number of particle pairs for the interaction.

    
    //In the context of the nb_b_interaction kernel, each thread is responsible for calculating the interaction between a pair of particles. The goal is to calculate the interaction forces between all possible pairs of particles in the simulation. To achieve this, the thread ID is mapped to particle indices.
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size2)
    {
        //ID1 and ID2 are calculated from tid to determine the indices of the interacting particles.
        //The combination of these calculations ensures that each thread ID is mapped to a unique pair of particle indices. This way, all possible pairs of particles are covered, and the interactions between particles can be calculated in parallel.
        int ID1 = int(tid /size);//tid / size calculates how many "rows" of particles the thread ID represents. In other words, it determines the index of the first particle in the pair (ID1).
        int ID2 = tid%(size);//tid % size calculates the remainder of the division of tid by size. This remainder corresponds to the index of the second particle in the pair (ID2)
        if(ID1 != ID2) //This condition ensures that the particle does not interact with itself. Interactions between a particle and itself are not considered
        {
        double r[3];
        //This line calculates the nearest image of particle positions in the periodic boundary conditions using the LeeEdwNearestImage function
        //The resulting displacement is stored in the r array.
        LeeEdwNearestImage(mdX[ID1], mdY[ID1], mdZ[ID1], mdX[ID2], mdY[ID2], mdZ[ID2], r, L, m, real_time);
        double r_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];//r_sqr calculates the squared distance between the particles.
        double f =0;//initialize the force to zero.
        double sigma = 0.8;
        double limit = 1.122462 * sigma;
 
        //lennard jones:
       
        //if (r_sqr < 1.258884)
        if(r_sqr < limit)
        {
                double r8 = 1/r_sqr* 1/r_sqr; //r^{-4}
                r8 *= r8; //r^{-8}
                double r14 = r8 *r8; //r^{-16}
                r14 *= r_sqr; //r^{-14}
                double sigma6 = sigma * sigma * sigma* sigma * sigma * sigma;
                double sigma12 = sigma6 * sigma6;
                //f = 24 * (2 * r14 - r8);
                f = 24 * (2 * sigma12* r14 - sigma6 * r8);
        }
        
        //FENE:
        //This part of the code is responsible for calculating the interaction forces between particles based on the FENE (Finitely Extensible Nonlinear Elastic) potential. The FENE potential is often used to model polymer chains where bonds between particles cannot be stretched beyond a certain limit
        
        if (topology == 1)
        {
            if (int(ID1/m) == int(ID2/m)) //checks if the interacting particles belong to the same chain (monomer). This is achieved by dividing the particle indices by m (monomer size) and checking if they are in the same division.
            {
                //check if the interacting particles are next to each other in the same chain. If they are, it calculates the FENE interaction contribution,
                if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                {
                    f -= K_FENE/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                {
                    f -= K_FENE/(1 - r_sqr/2.25);
                }
            }   
        }
        
        //FENE:
        if (topology == 2 || topology == 3)
        {
            if (int(ID1/m) == int(ID2/m)) //similar conditions are checked for particles within the same chain
            {
                if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                {
                    f -= K_FENE/(1 - r_sqr/2.25);
                }

                if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                {
                    f -= K_FENE/(1 - r_sqr/2.25);
                }
            }
            
            if (ID1==int(m/4) && ID2 == m + int(3*m/4))
            {
                
                f -= K_FENE/(1 - r_sqr/2.25);
            }
                
            if (ID2==int(m/4) && ID1 == m + int(3*m/4))
            {
                f -= K_FENE/(1 - r_sqr/2.25);
            }
        }
        f/=mass; //After the interaction forces are calculated (f), they are divided by the mass of the particles to obtain the correct acceleration.

        fx[tid] = f * r[0] ;
        fy[tid] = f * r[1] ;
        fz[tid] = f * r[2] ;
        }
    
        else
        {
            fx[tid] = 0;
            fy[tid] = 0;
            fz[tid] = 0;
        }
      

    }

}


__global__ void Active_bending_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *fx_bend , double *fy_bend , double *fz_bend,
double *L,int size , double ux, double mass, double real_time, int m , int topology, double K_FENE, double K_bend)
{
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    

        if (tid<size){

        int loop;
        
        int ID = tid%m;
        double Ri_2[3];
        double Ri_1[3];
        double Ri[3];
        double Ri1[3];
        double ri[3];
        double r2;
        double dot_product;
        

        if(ID == 0){

            loop= int(tid/m) + 1;

            LeeEdwNearestImage(mdX[tid+1], mdY[tid+1], mdZ[tid+1] , mdX[tid+2] , mdY[tid+2] , mdZ[tid+2] , Ri1, L, ux, real_time);
            
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid] , mdX[tid+1] , mdY[tid+1] , mdZ[tid+1] , Ri, L, ux, real_time);

            LeeEdwNearestImage(mdX[m*loop-1], mdY[m*loop-1], mdZ[m*loop-1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

            LeeEdwNearestImage(mdX[m*loop-2], mdY[m*loop-2], mdZ[m*loop-2] , mdX[m*loop-1] , mdY[m*loop-1] , mdZ[m*loop-1] , Ri_2, L, ux, real_time);


             
        }
        else if (ID == 1){

            loop= int(tid/m) + 1;

            LeeEdwNearestImage(mdX[tid+1], mdY[tid+1], mdZ[tid+1] , mdX[tid+2] , mdY[tid+2] , mdZ[tid+2] , Ri1, L, ux, real_time);
            
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid] , mdX[tid+1] , mdY[tid+1] , mdZ[tid+1] , Ri, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-1], mdY[tid-1], mdZ[tid-1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

            LeeEdwNearestImage(mdX[m*loop-1], mdY[m*loop-1], mdZ[m*loop-1] , mdX[tid-1] , mdY[tid-1] , mdZ[tid-1] , Ri_2, L, ux, real_time);

            
        }
        else if(ID == (m-1)){

            loop= int(tid/m);

            LeeEdwNearestImage(mdX[m*loop], mdY[m*loop], mdZ[m*loop] , mdX[m*loop+1] , mdY[m*loop+1] , mdZ[m*loop+1] , Ri1, L, ux, real_time);
            
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid] , mdX[m*loop] , mdY[m*loop] , mdZ[m*loop] , Ri, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-1], mdY[tid-1], mdZ[tid-1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-2], mdY[tid-2], mdZ[tid-2] , mdX[tid-1] , mdY[tid-1] , mdZ[tid-1] , Ri_2, L, ux, real_time);
        }

        else if(ID == (m-2)){

            loop= int(tid/m);

            LeeEdwNearestImage(mdX[tid+1], mdY[tid+1], mdZ[tid+1] , mdX[m*loop] , mdY[m*loop] , mdZ[m*loop] , Ri1, L, ux, real_time);
            
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid] , mdX[tid+1] , mdY[tid+1] , mdZ[tid+1] , Ri, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-1], mdY[tid-1], mdZ[tid-1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-2], mdY[tid-2], mdZ[tid-2] , mdX[tid-1] , mdY[tid-1] , mdZ[tid-1] , Ri_2, L, ux, real_time);



        }


        
        else if(1 < ID < m-2){

            LeeEdwNearestImage(mdX[tid+1], mdY[tid+1], mdZ[tid+1] , mdX[tid+2] , mdY[tid+2] , mdZ[tid+2] , Ri1, L, ux, real_time);
            
            LeeEdwNearestImage(mdX[tid], mdY[tid], mdZ[tid] , mdX[tid+1] , mdY[tid+1] , mdZ[tid+1] , Ri, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-1], mdY[tid-1], mdZ[tid-1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

            LeeEdwNearestImage(mdX[tid-2], mdY[tid-2], mdZ[tid-2] , mdX[tid-1] , mdY[tid-1] , mdZ[tid-1] , Ri_2, L, ux, real_time);





        }

        ri[0]=mdX[tid];
        ri[1]=mdY[tid];
        ri[2]=mdZ[tid];

        r2 = ri[0]*ri[0]+ri[1]*ri[1]+ri[2]*ri[2];
        dot_product = (3*Ri_1[0] - 3*Ri[0] + Ri1[0] - Ri_2[0])*ri[0] + (3*Ri_1[1] - 3*Ri[1] + Ri1[1] - Ri_2[1])*ri[1] + (3*Ri_1[2] - 3*Ri[2] + Ri1[2] - Ri_2[2])*ri[2];

        fx_bend[tid] = -K_bend*ri[0]*dot_product/r2;
        fy_bend[tid] = -K_bend*ri[1]*dot_product/r2;
        fz_bend[tid] = -K_bend*ri[2]*dot_product/r2;

        fx[tid] = fx[tid] + fx_bend[tid];
        fy[tid] = fy[tid] + fy_bend[tid];
        fz[tid] = fz[tid] + fz_bend[tid];

    }


}

__host__ void Active_calc_acceleration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz, double *Fx_bend, double *Fy_bend, double *Fz_bend,
double *Ax , double *Ay , double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T, 
double *L, int size, int m, int topology, double real_time, int grid_size, double mass_fluid, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot, double *fa_x, double *fa_y, double *fa_z,double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array, double u_scale, double K_FENE, double K_bend)

{
  

    Active_nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux, mass, real_time , m , topology, K_FENE, K_bend);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_bending_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz , Fx_bend, Fy_bend, Fz_bend, L , size , ux,density, real_time , m , topology, K_FENE, K_bend);
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
    double *d_Fx_bend, double *d_Fy_bend, double *d_Fz_bend,
    double *d_fa_kx, double *d_fa_ky, double *d_fa_kz,
    double *d_fb_kx, double *d_fb_ky, double *d_fb_kz,
    double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz,
    double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz,
    double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot,
    double *d_ex, double *d_ey, double *d_ez,
    double *h_fa_x, double *h_fa_y, double *h_fa_z,
    double *h_fb_x, double *h_fb_y, double *h_fb_z,
    double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md , int Nmd, int density, double *d_L , double ux, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, int delta, 
    double real_time, int m, int N, double mass, double mass_fluid, double *gama_T, int *random_array, unsigned int seed, int topology, 
    double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, int *flag_array, double u_scale, double K_FENE, double K_bend)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {

        CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, mdVx_tot, mdVy_tot, mdVz_tot, dVx_tot, dVy_tot, dVz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

        //with this function call particles go to box's center of mass frame. 
        gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //firt velocity verlet step, in which particles' positions and velocities are updated.
        ActivevelocityVerletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_Ax_tot, d_Ay_tot, d_Az_tot , h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        //After updating particles' positions, a kernel named LEBC is called to apply boundary conditions to ensure that particles stay within the simulation box.
        CM_LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, Vxcm, ux, d_L, real_time, Nmd);
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
        d_Fx , d_Fy , d_Fz, d_Fx_bend , d_Fy_bend , d_Fz_bend,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd , m , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale, K_FENE, K_bend);



        sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


      
        //call CM_system again after Active_calc_acceleration because the CM has changed now.
        CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, mdVx_tot, mdVy_tot, mdVz_tot, dVx_tot, dVy_tot, dVz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

        //now we go to this another CM reference frame:
        gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        //velocityVerletKernel2 is called to complete the velocity Verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        ActivevelocityVerletKernel2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //call CM_system again after velocity verlet second step because the CM has changed again now.
        CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, mdVx_tot, mdVy_tot, mdVz_tot, dVx_tot, dVy_tot, dVz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


        backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;


        
    }
}

