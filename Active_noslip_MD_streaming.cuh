

__global__ void noslip_tangential_vectors(double *mdX, double *mdY , double *mdZ ,
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
           
            regular_distance(mdX[tid], mdY[tid], mdZ[tid], mdX[m*loop], mdY[m*loop], mdZ[m*loop], a, L, ux, real_time);
            
        }
        else if (ID < (m-1))
        {
           
            regular_distance(mdX[tid], mdY[tid], mdZ[tid], mdX[tid+1], mdY[tid+1], mdZ[tid+1], a, L, ux, real_time);
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
       
        //printf("ex[%i]=%f, ey[%i]=%f, ez[%i]=%f\n", tid, ex[tid], tid, ey[tid], tid, ez[tid]);


    }
}

__global__ void CM_totalActive_calc_acceleration(double *Ax, double *Ay, double *Az, double *Aa_kx, double *Aa_ky, double *Aa_kz, double *Ab_kx, double *Ab_ky, double *Ab_kz, int *random_array, double *Ax_tot, double *Ay_tot, double *Az_tot, int size, int topology){

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


__global__ void Lab_totalActive_calc_acceleration(double *Ax, double *Ay, double *Az, double *Aa_kx, double *Aa_ky, double *Aa_kz, int *random_array, double *Ax_tot_lab, double *Ay_tot_lab, double *Az_tot_lab, int size, int topology){

    int tid=blockIdx.x * blockDim.x + threadIdx.x;

    //here I added a randomness to the active and backflow forces exerting on the monomers. 
    //we can change this manually or we can replace any other function instead of random_array as we prefer.
    
    if(tid< size){

        if(topology == 4){
            
            Ax_tot_lab[tid]=Ax[tid] + Aa_kx[tid]; 
            Ay_tot_lab[tid]=Ay[tid] + Aa_ky[tid];
            Az_tot_lab[tid]=Az[tid] + Aa_kz[tid];
            //printf("Aa_kx[%i]=%f, Aa_ky[%i]=%f, Aa_kz[%i]=%f\n", tid, Aa_kx[tid], tid, Aa_ky[tid], tid, Aa_kz[tid]);

        }
        else{

            Ax_tot_lab[tid]=Ax[tid]+(Aa_kx[tid])*random_array[tid]; 
            Ay_tot_lab[tid]=Ay[tid]+(Aa_ky[tid])*random_array[tid];
            Az_tot_lab[tid]=Az[tid]+(Aa_kz[tid])*random_array[tid];
            //printf("Aa_kx[%i]=%f, Aa_ky[%i]=%f, Aa_kz[%i]=%f\n", tid, Aa_kx[tid], tid, Aa_ky[tid], tid, Aa_kz[tid]);
            //printf("Ax_tot_lab[%i]=%f, Ay_tot_lab[%i]=%f, Az_tot_lab[%i]=%f\n", tid, Ax_tot_lab[tid], tid, Ay_tot_lab[tid], tid, Az_tot_lab[tid]);
        }
    }
   


}





__host__ void noslip_monomer_active_backward_forces(double *mdX, double *mdY , double *mdZ ,
double *Ax, double *Ay, double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T,
double *L, int size, double mass_fluid, double real_time, int m, int topology, int grid_size, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot,
double *Ax_tot_lab, double *Ay_tot_lab, double *Az_tot_lab, double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array,double u_scale)
{
    double Q = -mass/(size*mass+mass_fluid*N);
    double Mtot = (size*mass+mass_fluid*N);
    //shared_mem_size: The amount of shared memory allocated per block for the reduction operation.
    int shared_mem_size = 3 * blockSize * sizeof(double);
    
    double *d_Q;
    cudaMalloc((void**)&d_Q, sizeof(double));
    cudaMemcpy(d_Q, &Q, sizeof(double), cudaMemcpyHostToDevice); 

    if (topology == 4) //size= 1 (Nmd = 1) only one particle exists.
    {
        double *gamaTT;
        cudaMalloc((void**)&gamaTT, sizeof(double));
        cudaMemcpy(gamaTT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);


        SpecificOrientedForce<<<grid_size,blockSize>>>(mdX, mdY, mdZ, real_time, u_scale, size, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, gamaTT, d_Q, mass, u_scale);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        CM_totalActive_calc_acceleration<<<grid_size,blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, 1, topology);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        Lab_totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, random_array, Ax_tot_lab, Ay_tot_lab, Az_tot_lab, size, topology);
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

        *Ax_cm = fax/Mtot;
        *Ay_cm = fay/Mtot;
        *Az_cm = faz/Mtot;

     
    cudaFree(gamaTT);
    }

    else
    {
        
        if (random_flag == 1)
        {

            //int shared_mem_size = 3 * blockSize * sizeof(double); // allocate shared memory for the intermediate reduction results.
            //printf("ex[0]%f\n",ex[0]);
            //calculating tangential vectors:
            noslip_tangential_vectors<<<grid_size,blockSize>>>(mdX, mdY, mdZ, ex, ey, ez, L, size, u_scale, mass, real_time, m, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

           
            double *gamaT;
            cudaMalloc((void**)&gamaT, sizeof(double));
            cudaMemcpy(gamaT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);
            //printf("gama_T=%f\n",*gama_T);
        
            //printf("88gama_T=%f\n",*gama_T);
            //printf("mmm=%i\n", mass);
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
            CM_totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, random_array, Ax_tot, Ay_tot, Az_tot, size, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            Lab_totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, random_array, Ax_tot_lab, Ay_tot_lab, Az_tot_lab, size, topology);
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
                fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(cudaStatus));
    
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
           

            *fb_x = *fa_x * Q;
            *fb_y = *fa_y * Q;
            *fb_z = *fa_z * Q;

            *Ax_cm = *fa_x/Mtot;
            *Ay_cm = *fa_y/Mtot;
            *Az_cm = *fa_z/Mtot;
            

            
            cudaFree(gamaT);
            free(sumx);  free(sumy);  free(sumz);

        }
        else if(random_flag == 0)
        { 
            
            //int shared_mem_size = 3 * blockSize * sizeof(double); // allocate shared memory for the intermediate reduction results.
            //printf("ex[0]%f\n",ex[0]);
            //calculating tangential vectors:
            noslip_tangential_vectors<<<grid_size,blockSize>>>(mdX, mdY, mdZ, ex, ey, ez, L, size, ux, mass, real_time, m, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

           
            double *gamaT;
            cudaMalloc((void**)&gamaT, sizeof(double));
            cudaMemcpy(gamaT, gama_T, sizeof(double) , cudaMemcpyHostToDevice);
            //printf("gama_T=%f\n",*gama_T);
        
            //printf("88gama_T=%f\n",*gama_T);
            //printf("mmm=%i\n", mass);
            //forces calculations in a seperate kernel:
            Active_calc_forces<<<grid_size,blockSize>>>(fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz,
                    ex, ey, ez, ux, mass, mass_fluid, size, N, gamaT, u_scale);

          
    
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

       

            choiceArray<<<grid_size,blockSize>>>(flag_array, size);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
   

            CM_totalActive_calc_acceleration<<<grid_size,blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, size, topology);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            Lab_totalActive_calc_acceleration<<<grid_size, blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, flag_array, Ax_tot_lab, Ay_tot_lab, Az_tot_lab, size, topology);
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
           
    
            //*fa_x=*fa_x* *gama_T*u_scale;
            //*fa_y=*fa_y* *gama_T*u_scale;
            //*fa_z=*fa_z* *gama_T*u_scale;
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            *Ax_cm = *fa_x/Mtot;
            *Ay_cm = *fa_y/Mtot;
            *Az_cm = *fa_z/Mtot;

            cudaFree(gamaT);
            free(sumx);  free(sumy);  free(sumz);
     
        }
  
    }
}






//calculating interaction matrix of the system in the given time when BC is periodic
__global__ void Active_noslip_nb_b_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *L,int size , double ux, double mass, double real_time, int m , int topology, double K_FENE, double K_bend)
{
    int size2 = size*(size); //size2 calculates the total number of particle pairs for the interaction.

    //printf("noslip BC\n");

    //In the context of the nb_b_interaction kernel, each thread is responsible for calculating the interaction between a pair of particles. The goal is to calculate the interaction forces between all possible pairs of particles in the simulation. To achieve this, the thread ID is mapped to particle indices.
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if(topology == 4){

        fx[tid] = 0;
        fy[tid] = 0;
        fz[tid] = 0;

    }
    else{
        if (tid<size2)
        {
            //ID1 and ID2 are calculated from tid to determine the indices of the interacting particles.
            //The combination of these calculations ensures that each thread ID is mapped to a unique pair of particle indices. This way, all possible pairs of particles are covered, and the interactions between particles can be calculated in parallel.
            int ID1 = int(tid /size);//tid / size calculates how many "rows" of particles the thread ID represents. In other words, it determines the index of the first particle in the pair (ID1).
            int ID2 = tid%(size);//tid % size calculates the remainder of the division of tid by size. This remainder corresponds to the index of the second particle in the pair (ID2)
            if(ID1 != ID2) //This condition ensures that the particle does not interact with itself. Interactions between a particle and itself are not considered
            {
            double r[3];
            //This line calculates the distance of particle positions in the noslip regular conditions using the regular_distance function
            //The resulting displacement is stored in the r array.
            regular_distance(mdX[ID1], mdY[ID1], mdZ[ID1] , mdX[ID2] , mdY[ID2] , mdZ[ID2] , r,L, ux, real_time);
            double r_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];//r_sqr calculates the squared distance between the particles.
            double f =0;//initialize the force to zero.
            double sigma = 0.8;
            double limit =  1.122462 * sigma;

 
            //lennard jones:
       
            //if (r_sqr < 1.258884)
            if (r_sqr < limit)
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
            //printf("f=%f\n", f); 
            //printf("r_sqr=%f\n", r_sqr);   
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
            
                if (ID1==int(m/4) && ID2 ==m+int(3*m/4))
                {
                
                    f -= K_FENE/(1 - r_sqr/2.25);
                }
                
                if (ID2==int(m/4) && ID1 ==m+int(3*m/4))
                {
                    f -= K_FENE/(1 - r_sqr/2.25);
                }
            } 
            f/=mass; //After the interaction forces are calculated (f), they are divided by the mass of the particles to obtain the correct acceleration.

            fx[tid] = f * r[0] ;
            fy[tid] = f * r[1] ;
            fz[tid] = f * r[2] ;

            //printf("fx[%i]=%f, fy[%i]=%f, fz[%i]=%f\n", tid, fx[tid], tid, fy[tid], tid, fz[tid]);
            }
    
            else
            {
                fx[tid] = 0;
                fy[tid] = 0;
                fz[tid] = 0;
            }
      

        }

    }
}



__global__ void Active_noslip_bending_interaction( 
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
        
        int TID1;
        int TID2;
        int TID_1;
        int TID_2;
        
        loop = int(tid/m);
        TID1 = loop*m + (tid+1)%m;
        TID2 = loop*m + (tid+2)%m;
        TID_1 = loop*m + (tid-1)%m ;
        TID_2 = loop*m + (tid-2)%m ;
        
        if(tid == 0){
            TID_1 = TID_1 + m;
            TID_2 = TID_2 + m;
        }
        if(tid == 1) TID_2 = TID_2 + m;


        regular_distance(mdX[TID1], mdY[TID1], mdZ[TID1] , mdX[TID2] , mdY[TID2] , mdZ[TID2] , Ri1, L, ux, real_time);
            
        regular_distance(mdX[tid], mdY[tid], mdZ[tid] , mdX[TID1] , mdY[TID1] , mdZ[TID1] , Ri, L, ux, real_time);

        regular_distance(mdX[TID_1], mdY[TID_1], mdZ[TID_1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

        regular_distance(mdX[TID_2], mdY[TID_2], mdZ[TID_2] , mdX[TID_1] , mdY[TID_1] , mdZ[TID_1] , Ri_2, L, ux, real_time);



        ri[0]=mdX[tid];
        ri[1]=mdY[tid];
        ri[2]=mdZ[tid];

        r2 = ri[0]*ri[0]+ri[1]*ri[1]+ri[2]*ri[2];
        dot_product = (3*Ri_1[0] - 3*Ri[0] + Ri1[0] - Ri_2[0])*ri[0] + (3*Ri_1[1] - 3*Ri[1] + Ri1[1] - Ri_2[1])*ri[1] + (3*Ri_1[2] - 3*Ri[2] + Ri1[2] - Ri_2[2])*ri[2];


        if(r2 != 0 ){
            fx_bend[tid] = -K_bend*ri[0]*dot_product/r2;
            fy_bend[tid] = -K_bend*ri[1]*dot_product/r2;
            fz_bend[tid] = -K_bend*ri[2]*dot_product/r2;
        }

        //if(ID != 0 ){
            fx[tid] = fx[tid] + fx_bend[tid];
            fy[tid] = fy[tid] + fy_bend[tid];
            fz[tid] = fz[tid] + fz_bend[tid];
        //}
        

    }

}


__global__ void Active_noslip_stretching_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *fx_stretch , double *fy_stretch , double *fz_stretch, 
double *L,int size , double ux, double mass, double real_time, int m , int topology, double K_FENE, double K_l)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size){
        int loop;
        
      
        int ID = tid%m;
        double Ri_1[3];
        double Ri[3]; 
        double Ri2;
        double Ri_12;
        double li;
        double li_1;
        
        
        int TID1;
        int TID_1;
        
        
        loop = int(tid/m);
        TID1 = loop*m + (tid+1)%m;
        TID_1 = loop*m + (tid-1)%m ;

        if(tid == 0){
            TID_1 = TID_1 + m;
        }
        

        regular_distance(mdX[tid], mdY[tid], mdZ[tid] , mdX[TID1] , mdY[TID1] , mdZ[TID1] , Ri, L, ux, real_time);

        regular_distance(mdX[TID_1], mdY[TID_1], mdZ[TID_1] , mdX[tid] , mdY[tid] , mdZ[tid] , Ri_1, L, ux, real_time);

        Ri2 = Ri[0] * Ri[0] + Ri[1] * Ri[1] + Ri[2] * Ri[2];
        Ri_12 = Ri_1[0] * Ri_1[0] + Ri_1[1] * Ri_1[1] + Ri_1[2] * Ri_1[2];
        li = sqrt(abs(Ri2));
        li_1 = sqrt(abs(Ri_12));

        if(Ri_12 != 0 && Ri2 != 0){
            fx_stretch[tid] = -K_l * (Ri_1[0] * (Ri_12 - li_1)/Ri_12 - Ri[0] * (Ri2 - li)/Ri2);
            fy_stretch[tid] = -K_l * (Ri_1[1] * (Ri_12 - li_1)/Ri_12 - Ri[1] * (Ri2 - li)/Ri2);
            fz_stretch[tid] = -K_l * (Ri_1[2] * (Ri_12 - li_1)/Ri_12 - Ri[2] * (Ri2 - li)/Ri2);
        }

        fx[tid] = fx[tid] + fx_stretch[tid];
        fy[tid] = fy[tid] + fy_stretch[tid];
        fz[tid] = fz[tid] + fz_stretch[tid];


    }

}

//Active_noslip_calc_acceleration

__host__ void Active_noslip_calc_acceleration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz, double *Fx_bend, double *Fy_bend, double *Fz_bend, double *Fx_stretch, double *Fy_stretch, double *Fz_stretch,
double *Ax , double *Ay , double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T, 
double *L, int size, int m, int topology, double real_time, int grid_size, double mass_fluid, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot, double *Ax_tot_lab, double *Ay_tot_lab, double *Az_tot_lab, double *fa_x, double *fa_y, double *fa_z,double *fb_x, double *fb_y, double *fb_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array, double u_scale, double K_FENE, double K_bend, double K_l)

{
  

    Active_noslip_nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux, mass, real_time , m , topology, K_FENE, K_bend);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_noslip_bending_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz, Fx_bend, Fy_bend, Fz_bend, L , size , ux, mass, real_time , m , topology, K_FENE, K_bend);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /*Active_noslip_stretching_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz, Fx_stretch, Fy_stretch, Fz_stretch, L , size , ux, mass, real_time , m , topology, K_FENE, K_l);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/


    

    sum_kernel<<<grid_size,blockSize>>>(Fx , Fy, Fz, Ax , Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //printf("**GAMA=%f\n",*gama_T);
    

    noslip_monomer_active_backward_forces(x, y ,z ,
    Ax , Ay, Az, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, ex, ey, ez, ux, mass, gama_T, 
    L, size , mass_fluid, real_time, m, topology, grid_size, N , random_array, seed , Ax_tot, Ay_tot, Az_tot, Ax_tot_lab, Ay_tot_lab, Az_tot_lab, fa_x, fa_y, fa_z, fb_x, fb_y, fb_z, Ax_cm, Ay_cm, Az_cm, block_sum_ex, block_sum_ey, block_sum_ez, flag_array, u_scale);
    


    
}

///////// functions we need for noslip part:

//CM_md_wall_sign
//a function to consider velocity sign of particles and determine which sides of the box it should interact with 
__global__ void CM_md_wall_sign(double *mdvx, double *mdvy, double *mdvz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, int Nmd, double *Vxcm, double *Vycm, double *Vzcm){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){
        if (mdvx[tid] > -*Vxcm )  wall_sign_x[tid] = 1;
        else if (mdvx[tid] < -*Vxcm)  wall_sign_x[tid] = -1;
        else if(mdvx[tid] == -*Vxcm)  wall_sign_x[tid] = 0;
        
        if (mdvy[tid] > -*Vycm ) wall_sign_y[tid] = 1;
        else if (mdvy[tid] < -*Vycm) wall_sign_y[tid] = -1;
        else if (mdvy[tid] == -*Vycm )  wall_sign_y[tid] = 0;

        if (mdvz[tid] > -*Vzcm) wall_sign_z[tid] = 1;
        else if (mdvz[tid] < -*Vzcm) wall_sign_z[tid] = -1;
        else if (mdvz[tid] == -*Vzcm)  wall_sign_z[tid] = 0;

        (isnan(mdvx[tid])|| isnan(mdvy[tid]) || isnan(mdvz[tid])) ? printf("00vx[%i]=%f, vy[%i]=%f, vz[%i]=%f \n", tid, mdvx[tid], tid, mdvy[tid], tid, mdvz[tid])
                                                            : printf("");


    }
}

//CM_md_distance_from_walls
//a function to calculate distance of particles which are inside the box from the corresponding walls:
__global__ void CM_md_distance_from_walls(double *mdx, double *mdy, double *mdz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){
        if (wall_sign_x[tid] == 1)   x_wall_dist[tid] = L[0]/2-((mdx[tid]) + *Xcm);
        else if (wall_sign_x[tid] == -1)  x_wall_dist[tid] = L[0]/2+((mdx[tid]) + *Xcm);
        else if(wall_sign_x[tid] == 0)  x_wall_dist[tid] = L[0]/2 -((mdx[tid]) + *Xcm);//we can change it as we like . it doesn't matter.


        if (wall_sign_y[tid] == 1)   y_wall_dist[tid] = L[1]/2-((mdy[tid]) + *Ycm);
        else if (wall_sign_y[tid] == -1)  y_wall_dist[tid] = L[1]/2+((mdy[tid]) + *Ycm);
        else if(wall_sign_y[tid] == 0)  y_wall_dist[tid] = L[1]/2 -((mdy[tid]) + *Ycm);//we can change it as we like . it doesn't matter.


        if (wall_sign_z[tid] == 1)   z_wall_dist[tid] = L[2]/2-((mdz[tid]) + *Zcm);
        else if (wall_sign_z[tid] == -1)  z_wall_dist[tid] = L[2]/2+((mdz[tid]) + *Zcm);
        else if(wall_sign_z[tid] == 0)  z_wall_dist[tid] = L[2]/2 -((mdz[tid]) + *Zcm);//we can change it as we like . it doesn't matter.



        //printf("***dist_x[%i]=%f, dist_y[%i]=%f, dist_z[%i]=%f\n", tid, x_wall_dist[tid], tid, y_wall_dist[tid], tid, z_wall_dist[tid]);
        int idxx;
        idxx = (int(mdx[tid] + L[0] / 2 + 2) + (L[0] + 4) * int(mdy[tid] + L[1] / 2 + 2) + (L[0] + 4) * (L[1] + 4) * int(mdz[tid] + L[2] / 2 + 2));
        //printf("index[%i]=%i, x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, idxx, tid, x[tid], tid, y[tid], tid, z[tid]);//checking

    }    


}


//************
//Active_noslip_md_deltaT
//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_noslip_md_deltaT(double *mdvx, double *mdvy, double *mdvz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double delta_x;
    double delta_y;
    double delta_z;
    double delta_x_p; double delta_x_n; double delta_y_p; double delta_y_n; double delta_z_p; double delta_z_n;
    if (tid<Nmd){
        
        

        if(wall_sign_x[tid] == 0 ){
            if(mdAx_tot[tid] == 0) md_dt_x[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAx_tot[tid] > 0.0)  md_dt_x[tid] = sqrt(2*x_wall_dist[tid]/mdAx_tot[tid]);
            else if(mdAx_tot[tid] < 0.0)  md_dt_x[tid] = sqrt(2*(x_wall_dist[tid]-L[0])/mdAx_tot[tid]);
        }


        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
            if(mdAx_tot[tid] == 0.0)   md_dt_x[tid] = abs(x_wall_dist[tid]/mdvx[tid]);

            else if (mdAx_tot[tid] != 0.0){

                delta_x = ((mdvx[tid]*mdvx[tid])+(2*x_wall_dist[tid]*(mdAx_tot[tid])));

                if(delta_x >= 0.0){
                        if(mdvx[tid] > 0.0)         md_dt_x[tid] = ((-mdvx[tid] + sqrt(delta_x))/(mdAx_tot[tid]));
                        else if(mdvx[tid] < 0.0)    md_dt_x[tid] = ((-mdvx[tid] - sqrt(delta_x))/(mdAx_tot[tid]));
                        
                } 
                else if (delta_x < 0.0){
                        delta_x_p = ((mdvx[tid]*mdvx[tid])+(2*(x_wall_dist[tid]-L[0])*(mdAx_tot[tid])));
                        delta_x_n = ((mdvx[tid]*mdvx[tid])+(2*(x_wall_dist[tid]+L[0])*(mdAx_tot[tid])));

                        if(mdvx[tid] > 0.0)        md_dt_x[tid] = ((-mdvx[tid] - sqrt(delta_x_p))/(mdAx_tot[tid]));
                        else if(mdvx[tid] < 0.0)   md_dt_x[tid] = ((-mdvx[tid] + sqrt(delta_x_n))/(mdAx_tot[tid]));
                }
                
            }
        }  

        if(wall_sign_y[tid] == 0 ){
            if(mdAy_tot[tid] == 0) md_dt_y[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAy_tot[tid] > 0.0)  md_dt_y[tid] = sqrt(2*y_wall_dist[tid]/mdAy_tot[tid]);
            else if(mdAy_tot[tid] < 0.0)  md_dt_y[tid] = sqrt(2*(y_wall_dist[tid]-L[1])/mdAy_tot[tid]);
        }

        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(mdAy_tot[tid]  == 0.0)   md_dt_y[tid] = abs(y_wall_dist[tid]/mdvy[tid]);
            
            else if (mdAy_tot[tid] != 0.0){

                delta_y = (mdvy[tid]*mdvy[tid])+(2*y_wall_dist[tid]*(mdAy_tot[tid]));

                if (delta_y >= 0){

                    if(mdvy[tid] > 0.0)              md_dt_y[tid] = ((-mdvy[tid] + sqrt(delta_y))/(mdAy_tot[tid]));
                    else if (mdvy[tid] < 0.0)        md_dt_y[tid] = ((-mdvy[tid] - sqrt(delta_y))/(mdAy_tot[tid]));
                }
                else if(delta_y < 0){

                    delta_y_p = ((mdvy[tid]*mdvy[tid])+(2*(y_wall_dist[tid]-L[1])*(mdAy_tot[tid])));
                    delta_y_n = ((mdvy[tid]*mdvy[tid])+(2*(y_wall_dist[tid]+L[1])*(mdAy_tot[tid])));

                    if(mdvy[tid] > 0.0)        md_dt_y[tid] = ((-mdvy[tid] - sqrt(delta_y_p))/(mdAy_tot[tid]));
                    else if(mdvy[tid] < 0.0)   md_dt_y[tid] = ((-mdvy[tid] + sqrt(delta_y_n))/(mdAy_tot[tid]));

                }        
            }
        }
  

        if(wall_sign_z[tid] == 0 ){
            if(mdAz_tot[tid] == 0)        md_dt_z[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAz_tot[tid] > 0.0)  md_dt_z[tid] = sqrt(2*z_wall_dist[tid]/mdAz_tot[tid]);
            else if(mdAz_tot[tid] < 0.0)  md_dt_z[tid] = sqrt(2*(z_wall_dist[tid]-L[2])/mdAz_tot[tid]);
        }
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(mdAz_tot[tid] == 0.0)   md_dt_z[tid] = abs(z_wall_dist[tid]/mdvz[tid]);

            else if (mdAz_tot[tid] != 0.0){

                delta_z = (mdvz[tid]*mdvz[tid])+(2*z_wall_dist[tid]*(mdAz_tot[tid]));

                if (delta_z >= 0.0){
                    
                    if(mdvz[tid] > 0.0)             md_dt_z[tid] = ((-mdvz[tid] + sqrt(delta_z))/(mdAz_tot[tid]));
                    else if(mdvz[tid] < 0.0)        md_dt_z[tid] = ((-mdvz[tid] - sqrt(delta_z))/(mdAz_tot[tid]));  
                }

                else if (delta_z < 0.0){
                
                    delta_z_p = ((mdvz[tid]*mdvz[tid])+(2*(z_wall_dist[tid]-L[2])*(mdAz_tot[tid])));
                    delta_z_n = ((mdvz[tid]*mdvz[tid])+(2*(z_wall_dist[tid]+L[2])*(mdAz_tot[tid])));

                    if(mdvz[tid] > 0.0)        md_dt_z[tid] = ((-mdvz[tid] - sqrt(delta_z_p))/(mdAz_tot[tid]));
                    else if(mdvz[tid] < 0.0)   md_dt_z[tid] = ((-mdvz[tid] + sqrt(delta_z_n))/(mdAz_tot[tid]));
                    
                }
                
            }
        }
    //printf("md_dt_x[%i]=%f, md_dt_y[%i]=%f, md_dt_z[%i]=%f\n", tid, md_dt_x[tid], tid, md_dt_y[tid], tid, md_dt_z[tid]);
    //printf("mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f\n", tid, mdvx[tid], tid, mdvy[tid], tid, mdvz[tid]);
    //printf("mdAx_tot[%i]=%f, mdAy_tot[%i]=%f, mdAz_tot[%i]=%f\n", tid, mdAx_tot[tid], tid, mdAy_tot[tid], tid, mdAz_tot[tid]);
    //if (md_dt_x[tid] <0.002 || md_dt_y[tid] < 0.002 || md_dt_z[tid]< 0.002)  printf("the %i th particle will go out of box\n", tid);
   
    }
}


//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_noslip_md_deltaT_opposite(double *mdvx, double *mdvy, double *mdvz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *md_dt_x_opp, double *md_dt_y_opp, double *md_dt_z_opp, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double delta_x;
    double delta_y;
    double delta_z;
    double delta_x_plus; double delta_x_minus; double delta_y_plus; double delta_y_minus; double delta_z_plus; double delta_z_minus;
    if (tid<Nmd){
        
        

        if(wall_sign_x[tid] == 0 ){
            if(mdAx_tot[tid] == 0) md_dt_x_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAx_tot[tid] > 0.0)  md_dt_x_opp[tid] = sqrt(2*x_wall_dist[tid]/mdAx_tot[tid]);
            else if(mdAx_tot[tid] < 0.0)  md_dt_x_opp[tid] = sqrt(2*(x_wall_dist[tid]-L[0])/mdAx_tot[tid]);
        }


        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
            if(mdAx_tot[tid] == 0.0){
              if(mdvx[tid]>0) md_dt_x_opp[tid] = abs((x_wall_dist[tid]-L[0])/(-mdvx[tid]));

              else if(mdvx[tid]<0) md_dt_x_opp[tid] = abs((x_wall_dist[tid]+L[0])/(-mdvx[tid]));
            }

            else if (mdAx_tot[tid] != 0.0){

                delta_x_plus = ((mdvx[tid]*mdvx[tid])+(2*(x_wall_dist[tid]+L[0])*(mdAx_tot[tid])));
                delta_x_minus = ((mdvx[tid]*mdvx[tid])+(2*(x_wall_dist[tid]-L[0])*(mdAx_tot[tid])));

                delta_x = ((mdvx[tid]*mdvx[tid])+(2*(x_wall_dist[tid])*(mdAx_tot[tid])));

                if(mdvx[tid] > 0.0 && delta_x_minus >= 0.0)    md_dt_x_opp[tid] = ((mdvx[tid] - sqrt(delta_x_minus))/(mdAx_tot[tid]));
                            
                else if(mdvx[tid] < 0.0 && delta_x_plus >= 0.0)      md_dt_x_opp[tid] = ((mdvx[tid] + sqrt(delta_x_plus))/(mdAx_tot[tid]));
                        
                
                else if(delta_x_minus < 0.0 && mdvx[tid] > 0.0){
                    
                    md_dt_x_opp[tid] = ((mdvx[tid] + sqrt(delta_x))/(mdAx_tot[tid]));

                } 
                else if(delta_x_plus < 0.0 && mdvx[tid] < 0.0){

                    md_dt_x_opp[tid] = ((mdvx[tid] - sqrt(delta_x))/(mdAx_tot[tid]));

                }
                
            }
        }  

        if(wall_sign_y[tid] == 0 ){
            if(mdAy_tot[tid] == 0) md_dt_y_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAy_tot[tid] > 0.0)  md_dt_y_opp[tid] = sqrt(2*y_wall_dist[tid]/mdAy_tot[tid]);
            else if(mdAy_tot[tid] < 0.0)  md_dt_y_opp[tid] = sqrt(2*(y_wall_dist[tid]-L[1])/mdAy_tot[tid]);
        }

        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(mdAy_tot[tid]  == 0.0){
                if(mdvy[tid] > 0.0)   md_dt_y_opp[tid] = abs((y_wall_dist[tid] - L[1])/mdvy[tid]);
                else if(mdvy[tid] < 0.0)  md_dt_y_opp[tid] = abs((y_wall_dist[tid] + L[1])/mdvy[tid]);
            }
            else if (mdAy_tot[tid] != 0.0){

                delta_y_plus = (mdvy[tid]*mdvy[tid])+(2*(y_wall_dist[tid] + L[1])*(mdAy_tot[tid]));
                delta_y_minus = (mdvy[tid]*mdvy[tid])+(2*(y_wall_dist[tid] - L[1])*(mdAy_tot[tid]));

                delta_y = (mdvy[tid]*mdvy[tid])+(2*y_wall_dist[tid]*(mdAy_tot[tid]));

                if(mdvy[tid] > 0.0 && delta_y_minus >= 0.0)    md_dt_y_opp[tid] = ((mdvy[tid] - sqrt(delta_y_minus))/(mdAy_tot[tid]));
                            
                else if(mdvy[tid] < 0.0 && delta_y_plus >= 0.0)      md_dt_y_opp[tid] = ((mdvy[tid] + sqrt(delta_y_plus))/(mdAy_tot[tid]));
                        
                
                else if(delta_y_minus < 0.0 && mdvy[tid] > 0.0){
                    
                    md_dt_y_opp[tid] = ((mdvy[tid] + sqrt(delta_y))/(mdAy_tot[tid]));

                } 
                else if(delta_y_plus < 0.0 && mdvy[tid] < 0.0){

                    md_dt_y_opp[tid] = ((mdvy[tid] - sqrt(delta_y))/(mdAy_tot[tid]));

                }

                    
            }
        }
  
        if(wall_sign_z[tid] == 0 ){
            if(mdAz_tot[tid] == 0) md_dt_z_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(mdAz_tot[tid] > 0.0)  md_dt_z_opp[tid] = sqrt(2*z_wall_dist[tid]/mdAz_tot[tid]);
            else if(mdAz_tot[tid] < 0.0)  md_dt_z_opp[tid] = sqrt(2*(z_wall_dist[tid]-L[2])/mdAz_tot[tid]);
        }


        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(mdAz_tot[tid] == 0.0){
              if(mdvz[tid]>0) md_dt_z_opp[tid] = abs((z_wall_dist[tid]-L[2])/(-mdvz[tid]));

              else if(mdvz[tid]<0) md_dt_z_opp[tid] = abs((z_wall_dist[tid]+L[2])/(-mdvz[tid]));
            }

            else if (mdAz_tot[tid] != 0.0){

                delta_z_plus = ((mdvz[tid]*mdvz[tid])+(2*(z_wall_dist[tid]+L[2])*(mdAz_tot[tid])));
                delta_z_minus = ((mdvz[tid]*mdvz[tid])+(2*(z_wall_dist[tid]-L[2])*(mdAz_tot[tid])));

                delta_z = ((mdvz[tid]*mdvz[tid])+(2*(z_wall_dist[tid])*(mdAz_tot[tid])));

                if(mdvz[tid] > 0.0 && delta_z_minus >= 0.0)    md_dt_z_opp[tid] = ((mdvz[tid] - sqrt(delta_z_minus))/(mdAz_tot[tid]));
                            
                else if(mdvz[tid] < 0.0 && delta_z_plus >= 0.0)      md_dt_z_opp[tid] = ((mdvz[tid] + sqrt(delta_z_plus))/(mdAz_tot[tid]));
                        
                
                else if(mdvz[tid] > 0.0 && delta_z_minus < 0.0 ){
                    
                    md_dt_z_opp[tid] = ((mdvz[tid] + sqrt(delta_z))/(mdAz_tot[tid]));

                } 
                else if(mdvz[tid] < 0.0 && delta_z_plus < 0.0 ){

                    md_dt_z_opp[tid] = ((mdvz[tid] - sqrt(delta_z))/(mdAz_tot[tid]));

                }
                
            }
        }  

    //printf("md_dt_x_opp[%i]=%f, md_dt_y_opp[%i]=%f, md_dt_z_opp[%i]=%f\n", tid, md_dt_x_opp[tid], tid, md_dt_y_opp[tid], tid, md_dt_z_opp[tid]);
    //printf("mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f\n", tid, mdvx[tid], tid, mdvy[tid], tid, mdvz[tid]);
    //printf("mdAx_tot[%i]=%f, mdAy_tot[%i]=%f, mdAz_tot[%i]=%f\n", tid, mdAx_tot[tid], tid, mdAy_tot[tid], tid, mdAz_tot[tid]);
    //if (md_dt_x_opp[tid] <0.002 || md_dt_y_opp[tid] < 0.002 || md_dt_z_opp[tid]< 0.002)  printf("the %i th particle will go out of box\n", tid);
   
    }
}


//Active_md_crossing_location
//calculate the crossing location where the particles intersect with one wall:
__global__ void Active_md_crossing_location(double *mdx, double *mdy, double *mdz, double *mdvx, double *mdvy, double *mdvz, double *mdx_o, double *mdy_o, double *mdz_o, double *md_dt_min, double md_dt, double *L, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd){

    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){
        //if( ((mdx[tid] + md_dt * mdvx[tid]) >L[0]/2 || (mdx[tid] + md_dt * mdvx[tid])<-L[0]/2 || (mdy[tid] + md_dt * mdvy[tid])>L[1]/2 || (mdy[tid] + md_dt * mdvy[tid])<-L[1]/2 || (mdz[tid]+ md_dt * mdvz[tid])>L[2]/2 || (mdz[tid] + md_dt * mdvz[tid])<-L[2]/2) && md_dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, md_dt_min[tid]);
        mdx_o[tid] = mdx[tid] + mdvx[tid] * md_dt_min[tid] + 0.5 * mdAx_tot[tid] * md_dt_min[tid] * md_dt_min[tid];
        mdy_o[tid] = mdy[tid] + mdvy[tid] * md_dt_min[tid] + 0.5 * mdAy_tot[tid] * md_dt_min[tid] * md_dt_min[tid];
        mdz_o[tid] = mdz[tid] + mdvz[tid] * md_dt_min[tid] + 0.5 * mdAz_tot[tid] * md_dt_min[tid] * md_dt_min[tid];
    }

}

//calculate the crossing location where the particles intersect with one wall:
__global__ void Active_md_opposite_crossing_location(double *mdx, double *mdy, double *mdz, double *mdvx, double *mdvy, double *mdvz, double *mdx_o_opp, double *mdy_o_opp, double *mdz_o_opp, double *md_dt_min_opp, double md_dt, double *L, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd){

    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){
        //if( ((mdx[tid] + md_dt * mdvx[tid]) >L[0]/2 || (mdx[tid] + md_dt * mdvx[tid])<-L[0]/2 || (mdy[tid] + md_dt * mdvy[tid])>L[1]/2 || (mdy[tid] + md_dt * mdvy[tid])<-L[1]/2 || (mdz[tid]+ md_dt * mdvz[tid])>L[2]/2 || (mdz[tid] + md_dt * mdvz[tid])<-L[2]/2) && md_dt_min[tid]>0.1) printf("dt_min_opp[%i] = %f\n", tid, md_dt_min_opp[tid]);
        mdx_o_opp[tid] = mdx[tid] + (-mdvx[tid]) * md_dt_min_opp[tid] + 0.5 * mdAx_tot[tid] * md_dt_min_opp[tid] * md_dt_min_opp[tid];
        mdy_o_opp[tid] = mdy[tid] + (-mdvy[tid]) * md_dt_min_opp[tid] + 0.5 * mdAy_tot[tid] * md_dt_min_opp[tid] * md_dt_min_opp[tid];
        mdz_o_opp[tid] = mdz[tid] + (-mdvz[tid]) * md_dt_min_opp[tid] + 0.5 * mdAz_tot[tid] * md_dt_min_opp[tid] * md_dt_min_opp[tid];
    }

}



//Active_md_crossing_velocity
__global__ void Active_md_crossing_velocity(double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *md_dt_min, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd){


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

        //calculate v(t+dt1) : in this case that we don't have acceleration it is equal to v(t).
        //then we put the velocity equal to v(t+dt1):
        //this part in this case is not necessary but we do it for generalization.
        mdvx_o[tid] = mdvx[tid] + md_dt_min[tid] * mdAx_tot[tid];
        mdvy_o[tid] = mdvy[tid] + md_dt_min[tid] * mdAy_tot[tid];
        mdvz_o[tid] = mdvz[tid] + md_dt_min[tid] * mdAz_tot[tid];
    }
    
}

//Active_md_crossing_velocity
__global__ void Active_md_opposite_crossing_velocity(double *mdvx, double *mdvy, double *mdvz, double *mdvx_o_opp, double *mdvy_o_opp, double *mdvz_o_opp, double *md_dt_min_opp, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, int Nmd){


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

        //calculate v(t+dt1) : in this case that we don't have acceleration it is equal to v(t).
        //then we put the velocity equal to v(t+dt1):
        //this part in this case is not necessary but we do it for generalization.
        mdvx_o_opp[tid] = -mdvx[tid] + md_dt_min_opp[tid] * mdAx_tot[tid];
        mdvy_o_opp[tid] = -mdvy[tid] + md_dt_min_opp[tid] * mdAy_tot[tid];
        mdvz_o_opp[tid] = -mdvz[tid] + md_dt_min_opp[tid] * mdAz_tot[tid];
    }
    
}

//Active_md_velocityverlet1
__global__ void Active_md_velocityverlet1(double *mdX, double *mdY , double *mdZ , 
double *mdVx , double *mdVy , double *mdVz,
double *mdAx_tot , double *mdAy_tot , double *mdAz_tot,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *L, double h, int Nmd)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Nmd)
    {
        // Particle velocities are updated by half a time step, and particle positions are updated based on the new velocities.

        
        mdX[particleID] = mdX[particleID] + h * mdVx[particleID] + 0.5 * h * h * mdAx_tot[particleID];
        mdY[particleID] = mdY[particleID] + h * mdVy[particleID] + 0.5 * h * h * mdAy_tot[particleID];
        mdZ[particleID] = mdZ[particleID] + h * mdVz[particleID] + 0.5 * h * h * mdAz_tot[particleID];

        if((mdX[particleID] + *Xcm )>L[0]/2 || (mdX[particleID] + *Xcm)<-L[0]/2 || (mdY[particleID] + *Ycm )>L[1]/2 || (mdY[particleID] + *Ycm )<-L[1]/2 || (mdZ[particleID] + *Zcm )>L[2]/2 || (mdZ[particleID] + *Zcm )<-L[2]/2){
            
            printf("the %i th particle went out mdX[%i]=%f, mdY[%i]=%f, mdZ[%i]=%f]\n ", particleID, particleID, mdX[particleID] + *Xcm, particleID, mdY[particleID] + *Ycm, particleID, mdZ[particleID] + *Zcm );
            printf("the %i th particle went out mdVx[%i]=%f, mdVy[%i]=%f, mdVz[%i]=%f\n ", particleID, particleID, mdVx[particleID] + *Vxcm, particleID, mdVy[particleID] + *Vycm, particleID, mdVz[particleID] + *Vzcm );
        }

        mdVx[particleID] +=  h * mdAx_tot[particleID];// * 0.5;
        mdVy[particleID] +=  h * mdAy_tot[particleID];// * 0.5;
        mdVz[particleID] +=  h * mdAz_tot[particleID];// * 0.5;


        //printf("mdAx_tot[%i]=%f, mdAy_tot[%i]=%f, mdAz_tot[%i]=%f\n", particleID, mdAx_tot[particleID], particleID, mdAy_tot[particleID], particleID, mdAz_tot[particleID]);
        
    }
}

//*********************
//Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1
__global__ void Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *md_dt_min, double dt, double *L, int Nmd){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

     

        if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o[tid];
            mdy[tid] = mdy_o[tid];
            mdz[tid] = mdz_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o[tid];
            mdvy[tid] = -mdvy_o[tid];
            mdvz[tid] = -mdvz_o[tid];
            //let the particle move during dt-dt1 with the reversed velocity:
            mdx[tid] += (dt - (md_dt_min[tid])) * mdvx[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAx_tot[tid];
            mdy[tid] += (dt - (md_dt_min[tid])) * mdvy[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAy_tot[tid];
            mdz[tid] += (dt - (md_dt_min[tid])) * mdvz[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAz_tot[tid];
            mdvx[tid]=mdvx[tid]+ (dt - (md_dt_min[tid])) * mdAx_tot[tid];
            mdvy[tid]=mdvy[tid]+ (dt - (md_dt_min[tid])) * mdAy_tot[tid];
            mdvz[tid]=mdvz[tid]+ (dt - (md_dt_min[tid])) * mdAz_tot[tid];

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        
    }

}
__global__ void md_particles_on_crossing_points(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *md_dt_min, double md_dt, double *L, int Nmd, int *n_out_flag){



    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

        //if(md_dt_min[tid] < md_dt){
        if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o[tid];
            mdy[tid] = mdy_o[tid];
            mdz[tid] = mdz_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o[tid];
            mdvy[tid] = -mdvy_o[tid];
            mdvz[tid] = -mdvz_o[tid];
            n_out_flag[tid] = 1;
        }
        else  n_out_flag[tid]=0;
    }

}

__global__ void md_particles_on_opposite_crossing_points(double *mdx, double *mdy, double *mdz, double *mdx_o_opp, double *mdy_o_opp, double *mdz_o_opp, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o_opp, double *mdvy_o_opp, double *mdvz_o_opp, double *md_dt_min, double *md_dt_min_opp, double md_dt, double *L, int Nmd, int *n_out_flag_opp){



    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

        //if(md_dt_min_opp[tid] < (md_dt - 2*md_dt_min[tid]) ){
        if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o_opp[tid];
            mdy[tid] = mdy_o_opp[tid];
            mdz[tid] = mdz_o_opp[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o_opp[tid];
            mdvy[tid] = -mdvy_o_opp[tid];
            mdvz[tid] = -mdvz_o_opp[tid];
            n_out_flag_opp[tid] = 1;
        }
        else  n_out_flag_opp[tid]=0;
    }

}

//Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1
__global__ void Active_CM_md_bounceback_velocityverlet1(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *Ax_cm, double *Ay_cm, double *Az_cm, double *md_dt_min, double md_dt, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm, int *errorFlag, int *n_out_flag){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

     

        //if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
    if((mdx[tid]+*Xcm)>L[0]/2 || (mdx[tid]+*Xcm)<-L[0]/2 || (mdy[tid]+*Ycm)>L[1]/2 || (mdy[tid]+*Ycm)<-L[1]/2 || (mdz[tid]+*Zcm)>L[2]/2 || (mdz[tid]+*Zcm)<-L[2]/2){
        
        if(n_out_flag[tid] == 1){
            
            if (md_dt_min[tid] > md_dt) {
                printf("*********************md_dt_min[%i]=%f\n", tid, md_dt_min[tid]);
                md_dt_min[tid]=md_dt;
                mdAx_tot[tid]=-*Ax_cm;
                mdAy_tot[tid]=-*Ay_cm;
                mdAz_tot[tid]=-*Az_cm;
                *errorFlag = 1;  // Set the error flag
                return;  // Early exit
            }
            //let the particle move during dt-dt1 with the reversed velocity:
            mdx[tid] += (md_dt - (md_dt_min[tid])) * mdvx[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * (-*Ax_cm);// mdAx_tot[tid];
            mdy[tid] += (md_dt - (md_dt_min[tid])) * mdvy[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * (-*Ay_cm);// mdAy_tot[tid];
            mdz[tid] += (md_dt - (md_dt_min[tid])) * mdvz[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * (-*Az_cm);// mdAz_tot[tid];
            mdvx[tid]=mdvx[tid] +   (md_dt - (md_dt_min[tid])) * (-*Ax_cm);// mdAx_tot[tid];// * 0.5;
            mdvy[tid]=mdvy[tid] +   (md_dt - (md_dt_min[tid])) * (-*Ay_cm);//mdAy_tot[tid];// * 0.5;
            mdvz[tid]=mdvz[tid] +   (md_dt - (md_dt_min[tid])) * (-*Az_cm);//mdAz_tot[tid];// * 0.5;
        
            if((mdx_o[tid] + *Xcm )>L[0]/2 || (mdx_o[tid] + *Xcm)<-L[0]/2 || (mdy_o[tid] + *Ycm )>L[1]/2 || (mdy_o[tid] + *Ycm )<-L[1]/2 || (mdz_o[tid] + *Zcm )>L[2]/2 || (mdz_o[tid] + *Zcm )<-L[2]/2)  printf("wrong mdx_o[%i]=%f, mdY_o[%i]=%f, mdz_o[%i]=%f\n", tid, (mdx_o[tid] + *Xcm), tid, (mdy_o[tid] + *Ycm), tid, (mdz_o[tid] + *Zcm));

            printf("after bounceback in lab mdx[%i]=%f, mdy[%i]=%f, mdz[%i]=%f\n ", tid, (mdx[tid] + *Xcm), tid, (mdy[tid] + *Ycm), tid, (mdz[tid] + *Zcm));

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        if((mdx[tid] + *Xcm )>L[0]/2 || (mdx[tid] + *Xcm)<-L[0]/2 || (mdy[tid] + *Ycm )>L[1]/2 || (mdy[tid] + *Ycm )<-L[1]/2 || (mdz[tid] + *Zcm )>L[2]/2 || (mdz[tid] + *Zcm )<-L[2]/2){

            *errorFlag = 1;  // Set the error flag
            return;  // Early exit
        }
        
    }

}

}


__global__ void Active_CM_md_opposite_bounceback_velocityverlet1(double *mdx, double *mdy, double *mdz, double *mdx_o_opp, double *mdy_o_opp, double *mdz_o_opp, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o_opp, double *mdvy_o_opp, double *mdvz_o_opp, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *Ax_cm, double *Ay_cm, double *Az_cm, double *md_dt_min, double *md_dt_min_opp, double md_dt, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm, int *errorFlag, int *n_out_flag_opp){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<Nmd){

  

    //if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
    if((mdx[tid]+*Xcm)>L[0]/2 || (mdx[tid]+*Xcm)<-L[0]/2 || (mdy[tid]+*Ycm)>L[1]/2 || (mdy[tid]+*Ycm)<-L[1]/2 || (mdz[tid]+*Zcm)>L[2]/2 || (mdz[tid]+*Zcm)<-L[2]/2){
        
        if(n_out_flag_opp[tid] == 1){
            
            if (md_dt_min_opp[tid] > (md_dt - 2* md_dt_min[tid])) {
                printf("*********************md_dt_min[%i]=%f\n", tid, md_dt_min_opp[tid]);
                md_dt_min_opp[tid]=md_dt-2*md_dt_min[tid];
                
                *errorFlag = 1;  // Set the error flag
                return;  // Early exit
            }
            //let the particle move during dt-dt1 with the reversed velocity:
            mdx[tid] += (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * mdvx[tid] + 0.5 * ((md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])*(md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])) * (-*Ax_cm);// mdAx_tot[tid] in CM or in lab;
            mdy[tid] += (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * mdvy[tid] + 0.5 * ((md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])*(md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])) * (-*Ay_cm);// mdAy_tot[tid] in CM or in lab;
            mdz[tid] += (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * mdvz[tid] + 0.5 * ((md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])*(md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid])) * (-*Az_cm);// mdAz_tot[tid] in CM or in lab;
            mdvx[tid]= mdvx[tid] +   (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * (-*Ax_cm);// mdAx_tot[tid] in CM or in lab;// * 0.5;
            mdvy[tid]= mdvy[tid] +   (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * (-*Ay_cm);// mdAy_tot[tid] in CM or in lab;// * 0.5;
            mdvz[tid]= mdvz[tid] +   (md_dt - 2*(md_dt_min[tid])-md_dt_min_opp[tid]) * (-*Az_cm);// mdAz_tot[tid] in CM or in lab;// * 0.5;
        
            if((mdx_o_opp[tid] + *Xcm )>L[0]/2 || (mdx_o_opp[tid] + *Xcm)<-L[0]/2 || (mdy_o_opp[tid] + *Ycm )>L[1]/2 || (mdy_o_opp[tid] + *Ycm )<-L[1]/2 || (mdz_o_opp[tid] + *Zcm )>L[2]/2 || (mdz_o_opp[tid] + *Zcm )<-L[2]/2)  printf("wrong mdx_o_opp[%i]=%f, mdy_o_opp[%i]=%f, mdz_o_opp[%i]=%f\n", tid, (mdx_o_opp[tid] + *Xcm), tid, (mdy_o_opp[tid] + *Ycm), tid, (mdz_o_opp[tid] + *Zcm));

            printf("location after the second bounceback in lab mdx[%i]=%f, mdy[%i]=%f, mdz[%i]=%f\n ", tid, (mdx[tid] + *Xcm), tid, (mdy[tid] + *Ycm), tid, (mdz[tid] + *Zcm));
            printf("velocity after the second bounceback in lab mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f\n ", tid, (mdvx[tid] ), tid, (mdvy[tid] ), tid, (mdvz[tid] ));
        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        if((mdx[tid] + *Xcm )>L[0]/2 || (mdx[tid] + *Xcm)<-L[0]/2 || (mdy[tid] + *Ycm )>L[1]/2 || (mdy[tid] + *Ycm )<-L[1]/2 || (mdz[tid] + *Zcm )>L[2]/2 || (mdz[tid] + *Zcm )<-L[2]/2){

            

            *errorFlag = 1;  // Set the error flag
            return;  // Early exit
        }
        
    }

}

}


//Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1
__global__ void Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *md_dt_min, double md_dt, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm, int *errorFlag){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

     

        //if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
        if(md_dt_min[tid] < md_dt){

            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o[tid];
            mdy[tid] = mdy_o[tid];
            mdz[tid] = mdz_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o[tid];
            mdvy[tid] = -mdvy_o[tid];
            mdvz[tid] = -mdvz_o[tid];
            
            //let the particle move during dt-dt1 with the reversed velocity:
            mdx[tid] += (md_dt - (md_dt_min[tid])) * mdvx[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * mdAx_tot[tid];
            mdy[tid] += (md_dt - (md_dt_min[tid])) * mdvy[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * mdAy_tot[tid];
            mdz[tid] += (md_dt - (md_dt_min[tid])) * mdvz[tid] + 0.5 * ((md_dt - (md_dt_min[tid]))*(md_dt - (md_dt_min[tid]))) * mdAz_tot[tid];
            mdvx[tid]=mdvx[tid] +   (md_dt - (md_dt_min[tid])) * mdAx_tot[tid];// * 0.5;
            mdvy[tid]=mdvy[tid] +   (md_dt - (md_dt_min[tid])) * mdAy_tot[tid];// * 0.5;
            mdvz[tid]=mdvz[tid] +   (md_dt - (md_dt_min[tid])) * mdAz_tot[tid];// * 0.5;
        
            if((mdx_o[tid] + *Xcm )>L[0]/2 || (mdx_o[tid] + *Xcm)<-L[0]/2 || (mdy_o[tid] + *Ycm )>L[1]/2 || (mdy_o[tid] + *Ycm )<-L[1]/2 || (mdz_o[tid] + *Zcm )>L[2]/2 || (mdz_o[tid] + *Zcm )<-L[2]/2)  printf("wrong mdx_o[%i]=%f, mdY_o[%i]=%f, mdz_o[%i]=%f\n", tid, (mdx_o[tid] + *Xcm), tid, (mdy_o[tid] + *Ycm), tid, (mdz_o[tid] + *Zcm));
        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        if((mdx[tid] + *Xcm )>L[0]/2 || (mdx[tid] + *Xcm)<-L[0]/2 || (mdy[tid] + *Ycm )>L[1]/2 || (mdy[tid] + *Ycm )<-L[1]/2 || (mdz[tid] + *Zcm )>L[2]/2 || (mdz[tid] + *Zcm )<-L[2]/2){

            *errorFlag = 1;  // Set the error flag
            return;  // Early exit
        }
        
    }

}

//Active_md_velocityverlet2
//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void Active_md_velocityverlet2(double *mdx , double *mdy , double *mdz, double *mdVx , double *mdVy , double *mdVz,
double *mdAx_tot , double *mdAy_tot , double *mdAz_tot, double h, int Nmd)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Nmd)
    {
        mdVx[particleID] += 0.5 * h * mdAx_tot[particleID];
        mdVy[particleID] += 0.5 * h * mdAy_tot[particleID];
        mdVz[particleID] += 0.5 * h * mdAz_tot[particleID];
    }
}


//Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2
__global__ void Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *md_dt_min, double dt, double *L, int Nmd){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

    

        if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o[tid];
            mdy[tid] = mdy_o[tid];
            mdz[tid] = mdz_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o[tid];
            mdvy[tid] = -mdvy_o[tid];
            mdvz[tid] = -mdvz_o[tid];
            //let the particle move during dt-dt1 with the reversed velocity:
            mdx[tid] += (dt - (md_dt_min[tid])) * mdvx[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAx_tot[tid];
            mdy[tid] += (dt - (md_dt_min[tid])) * mdvy[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAy_tot[tid];
            mdz[tid] += (dt - (md_dt_min[tid])) * mdvz[tid] + 0.5 * ((dt - (md_dt_min[tid]))*(dt - (md_dt_min[tid]))) * mdAz_tot[tid];
            mdvx[tid]=mdvx[tid]+ (dt - (md_dt_min[tid])) * mdAx_tot[tid];
            mdvy[tid]=mdvy[tid]+ (dt - (md_dt_min[tid])) * mdAy_tot[tid];
            mdvz[tid]=mdvz[tid]+ (dt - (md_dt_min[tid])) * mdAz_tot[tid];

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
    }

}
 
//Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2
__global__ void Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *md_dt_min, double md_dt, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

    

        //if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
        if(md_dt_min[tid] < md_dt){
            
            mdvx[tid]=mdvx[tid]+ 0.5 * (md_dt - (md_dt_min[tid])) * mdAx_tot[tid];
            mdvy[tid]=mdvy[tid]+ 0.5 * (md_dt - (md_dt_min[tid])) * mdAy_tot[tid];
            mdvz[tid]=mdvz[tid]+ 0.5 * (md_dt - (md_dt_min[tid])) * mdAz_tot[tid];

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        if((mdx[tid] + *Xcm )>L[0]/2 || (mdx[tid] + *Xcm)<-L[0]/2 || (mdy[tid] + *Ycm )>L[1]/2 || (mdy[tid] + *Ycm )<-L[1]/2 || (mdz[tid] + *Zcm )>L[2]/2 || (mdz[tid] + *Zcm )<-L[2]/2)  printf("*************************goes out %i\n", tid);
        
    }

}






//first velocityverletKernel version 1:
__host__ void Active_noslip_md_velocityverletKernel1(double *mdX, double *mdY , double *mdZ, double *x, double *y, double *z,
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_Ax_tot_lab, double *d_Ay_tot_lab, double *d_Az_tot_lab,
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

    //CM_system
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    //gotoCMframe
    gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    CM_md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd, Vxcm, Vycm, Vzcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    CM_md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_noslip_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdX_o, mdY_o, mdZ_o, md_dt_min, h_md, L, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min,  d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, L, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_outside_particles
    /*outerParticles_CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
    mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_mpcd_tot, dn_md_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, 
    Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz,
    CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_n_outbox_mpcd, CMsumblock_n_outbox_md, topology, L);
    */
    
    //gotoOUTBOXCMframe  go to out of box cm frame for mpcd particles:
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go to out of box cm frame for md particles:(should I???)
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go back to the old CM frame mpcd
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

     //go back to the old CM frame md
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    //gotoLabFrame for mpcd particles:
    backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_system: now the CM has changed.
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

   
}

//first velocityverletKernel version 2:
__host__ void Active_noslip_md_velocityverletKernel3(double *mdX, double *mdY , double *mdZ, double *x, double *y, double *z,
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz, double *Aa_kx, double *Aa_ky, double *Aa_kz, double *Ab_kx, double *Ab_ky, double *Ab_kz, double *Ax_cm, double *Ay_cm, double *Az_cm,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_Ax_tot_lab, double *d_Ay_tot_lab, double *d_Az_tot_lab, 
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, int *hostErrorFlag, int *n_out_flag){

    //CM_system : calculate CM of the whole system.
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    //calculate md_wall_sign to determine which wall the initial velocity vector of the particle is pointing to. 
    md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box in Lab system:
    md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate 3 time periods needed for the particle to reach the walls in 3 directions.
    Active_noslip_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //the actual time period is the minimum of those three.
    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    
    //crossing location is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab which is the particle's acceleration in lab system)
    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdX_o, mdY_o, mdZ_o, md_dt_min, h_md, L, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //crossing velocity is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab)
    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min,  d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //take all mpcd particles to CM system. might not be necessary here.
    //gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //with this function call MD particles go to box's center of mass frame.
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    //a velocity verlet is performed in x and v 
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, L, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //take mpcd particles back to the lab frame, might not be necessary here.
    //backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //take all MD particles back to the Lab system.
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    
    //we put the particles that had gone outside the box, on the box's boundaries and set its velocity equal to the negative of the crossing velocity in Lab system.
    md_particles_on_crossing_points<<<grid_size,blockSize>>>(mdX, mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, h_md, L, Nmd, n_out_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    //CM_outside_particles
    /*outerParticles_CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
    mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_mpcd_tot, dn_md_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, 
    Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz,
    CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_n_outbox_mpcd, CMsumblock_n_outbox_md, topology, L);
    
    //gotoOUTBOXCMframe  go to out of box cm frame for mpcd particles:
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go to out of box cm frame for md particles:(should I???)
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    //CM_system : we call CM system to calculate the new CM after streaming. we should check if this is alright or not. we could also use the former CM system for bounceback part.
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    //take mpcd particles to CM frame.
    //gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //with this function call MD particles go to box's center of mass frame.
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //now we take crossing location and velocity to CM system.
    Take_o_to_CM_system<<<grid_size,blockSize>>>(mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    

    int *d_errorFlag;
    *hostErrorFlag = 0;
    cudaMalloc(&d_errorFlag, sizeof(int));
    cudaMemcpy(d_errorFlag, hostErrorFlag, sizeof(int), cudaMemcpyHostToDevice);

    double *Axcm, *Aycm, *Azcm;
    cudaMalloc(&Axcm, sizeof(double));
    cudaMalloc(&Aycm, sizeof(double));
    cudaMalloc(&Azcm, sizeof(double));
    cudaMemcpy(Axcm, Ax_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Aycm, Ay_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Azcm, Az_cm, sizeof(double), cudaMemcpyHostToDevice);

    //after putting the particles that had traveled outside of the box on its boundaries, we let them stream in the opposite direction for the time they had spent outside the box. 
    Active_CM_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, Axcm, Aycm, Azcm, md_dt_min, h_md, L, Nmd, Xcm, Ycm, Zcm, d_errorFlag, n_out_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Check for kernel errors and sync
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_errorFlag);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }

    // Check the error flag
    cudaMemcpy(hostErrorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);
    if (*hostErrorFlag) {
        printf("Error condition met in kernel. Exiting.\n");
        // Clean up and exit
        cudaFree(d_errorFlag);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }

    //go back to the old CM frame mpcd
    /*gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

     //go back to the old CM frame md
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    //take all mpcd particles back to the lab frame.
    //backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //take all the md particles back to the lab frame.
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    

    cudaFree(d_errorFlag);
    cudaFree(d_errorFlag);
    cudaFree(Axcm); cudaFree(Aycm); cudaFree(Azcm);

    

    
   
}


//first velocityverletKernel version 3:
__host__ void Active_noslip_md_velocityverletKernel5(double *mdX, double *mdY , double *mdZ, double *x, double *y, double *z,
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz, double *Aa_kx, double *Aa_ky, double *Aa_kz, double *Ab_kx, double *Ab_ky, double *Ab_kz, double *Ax_cm, double *Ay_cm, double *Az_cm,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min, double *md_dt_x_opp, double *md_dt_y_opp, double *md_dt_z_opp, double *md_dt_min_opp,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_o_opp, double *mdY_o_opp, double *mdZ_o_opp, double *mdvx_o_opp, double *mdvy_o_opp, double *mdvz_o_opp, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_Ax_tot_lab, double *d_Ay_tot_lab, double *d_Az_tot_lab, 
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, int *hostErrorFlag, int *hostErrorFlag_opp, int *n_out_flag, int *n_out_flag_opp, double *d_zero){

    //CM_system : calculate CM of the whole system.
    //CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    //Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    //calculate md_wall_sign to determine which wall the initial velocity vector of the particle is pointing to. 
    md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box in Lab system:
    md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate 3 time periods needed for the particle to reach the walls in 3 directions.
    Active_noslip_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //the actual time period is the minimum of those three.
    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    
    //crossing location is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab which is the particle's acceleration in lab system)
    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdX_o, mdY_o, mdZ_o, md_dt_min, h_md, L, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //crossing velocity is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab)
    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min,  d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    Active_noslip_md_deltaT_opposite<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x_opp, md_dt_y_opp, md_dt_z_opp, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x_opp, md_dt_y_opp, md_dt_z_opp, md_dt_min_opp, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    //crossing location is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab which is the particle's acceleration in lab system)
    Active_md_opposite_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdX_o_opp, mdY_o_opp, mdZ_o_opp, md_dt_min_opp, h_md, L, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //crossing velocity is calculated in Lab frame bacause we're using the accelerations in Lab system (A_tot_lab)
    Active_md_opposite_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o_opp, mdvy_o_opp, mdvz_o_opp, md_dt_min_opp, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    double *zero;
    cudaMalloc(&zero, sizeof(double));
    *d_zero = 0.0;
    cudaMemcpy(zero, d_zero, sizeof(double), cudaMemcpyHostToDevice);


    //a velocity verlet is performed in x and v 
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, zero, zero, zero, zero, zero, zero, L, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    
    //we put the particles that had gone outside the box, on the box's boundaries and set its velocity equal to the negative of the crossing velocity in Lab system.
    md_particles_on_crossing_points<<<grid_size,blockSize>>>(mdX, mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, h_md, L, Nmd, n_out_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
  
    //CM_system : we call CM system to calculate the new CM after streaming. we should check if this is alright or not. we could also use the former CM system for bounceback part.
    //CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    //Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    

    int *d_errorFlag_md;
    *hostErrorFlag = 0;
    cudaMalloc(&d_errorFlag_md, sizeof(int));
    cudaMemcpy(d_errorFlag_md, hostErrorFlag, sizeof(int), cudaMemcpyHostToDevice);

    double *Axcm, *Aycm, *Azcm;
    cudaMalloc(&Axcm, sizeof(double));
    cudaMalloc(&Aycm, sizeof(double));
    cudaMalloc(&Azcm, sizeof(double));
    cudaMemcpy(Axcm, Ax_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Aycm, Ay_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Azcm, Az_cm, sizeof(double), cudaMemcpyHostToDevice);

    

    //after putting the particles that had traveled outside of the box on its boundaries, we let them stream in the opposite direction for the time they had spent outside the box. 
    Active_CM_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, zero, zero, zero, md_dt_min, h_md, L, Nmd, zero, zero, zero, d_errorFlag_md, n_out_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Check for kernel errors and sync
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_errorFlag_md);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }

    // Check the error flag
    cudaMemcpy(hostErrorFlag, d_errorFlag_md, sizeof(int), cudaMemcpyDeviceToHost);
    if (*hostErrorFlag) {
        printf("Error condition met in kernel (MPCD). Exiting.\n");
        // Clean up and exit
        cudaFree(d_errorFlag_md);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }


     int *d_errorFlag_md_opp;
    *hostErrorFlag_opp = 0;
    cudaMalloc(&d_errorFlag_md_opp, sizeof(int));
    cudaMemcpy(d_errorFlag_md_opp, hostErrorFlag_opp, sizeof(int), cudaMemcpyHostToDevice);

    double *zer;
    cudaMalloc(&zer, sizeof(double));
    *d_zero = 0.0;
    cudaMemcpy(zer, d_zero, sizeof(double), cudaMemcpyHostToDevice); 

    md_particles_on_opposite_crossing_points<<<grid_size,blockSize>>>(mdX, mdY, mdZ, mdX_o_opp, mdY_o_opp, mdZ_o_opp, mdvx, mdvy, mdvz, mdvx_o_opp, mdvy_o_opp, mdvz_o_opp, md_dt_min, md_dt_min_opp, h_md, L, Nmd, n_out_flag_opp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_md_opposite_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o_opp, mdY_o_opp, mdZ_o_opp, mdvx, mdvy, mdvz, mdvx_o_opp, mdvy_o_opp, mdvz_o_opp, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, zer, zer, zer, md_dt_min, md_dt_min_opp, h_md, L, Nmd, zer, zer, zer, d_errorFlag_md_opp, n_out_flag_opp);
    gpuErrchk( cudaPeekAtLastError() );               
    gpuErrchk( cudaDeviceSynchronize() );

    // Check for kernel errors and sync
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_errorFlag_md_opp);
        *hostErrorFlag_opp = -1;  // Set error flag
        return;
    }

    // Check the error flag
    cudaMemcpy(hostErrorFlag_opp, d_errorFlag_md_opp, sizeof(int), cudaMemcpyDeviceToHost);
    if (*hostErrorFlag_opp) {
        printf("Error condition met in kernel (second bounceback MD). Exiting.\n");
        // Clean up and exit
        cudaFree(d_errorFlag_md_opp);
        *hostErrorFlag_opp = -1;  // Set error flag
        return;
    }


    
    

    //cudaFree(d_errorFlag_md);
    cudaFree(Axcm); cudaFree(Aycm); cudaFree(Azcm);
    cudaFree(zero); cudaFree(zer);

    
   
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//second velocityverletKernel version 1:
__host__ void Active_noslip_md_velocityverletKernel2(double *mdX, double *mdY , double *mdZ, double *x, double *y, double *z,
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, 
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

    //CM_system
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    //gotoCMframe
    gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    CM_md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd, Vxcm, Vycm, Vzcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    CM_md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    Active_noslip_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx, mdvy, mdvz, mdX_o, mdY_o, mdZ_o, md_dt_min, h_md, L, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_velocityverlet2<<<grid_size, blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_outside_particles
    /*outerParticles_CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
    mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_mpcd_tot, dn_md_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, 
    Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz,
    CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_n_outbox_mpcd, CMsumblock_n_outbox_md, topology, L);
    */

    //gotoOUTBOXCMframe  go to out of box cm frame for mpcd particles:
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go to out of box cm frame for md particles:(should I???)
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go back to the old CM frame mpcd
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, vx, vy, vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

     //go back to the old CM frame md
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    //gotoLabFrame for mpcd particles:
    backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_system: now the CM has changed.
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


 }

//second velocityverletKernel version 2:
__host__ void Active_noslip_md_velocityverletKernel4(double *mdX, double *mdY , double *mdZ, double *x, double *y, double *z,
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, 
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

    

    //gotoCMframe
    //gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, md_dt_min, h_md, L, Nmd, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    Active_md_velocityverlet2<<<grid_size, blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    



    //gotoLabFrame for mpcd particles:
    //backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_system: now the CM has changed.
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


 }












__host__ void Active_noslip_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ, double *d_x, double *d_y, double *d_z,
    double *d_mdvx, double *d_mdvy, double *d_mdvz, double *d_vx, double *d_vy, double *d_vz, double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
    double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,  int *n_outbox_md, int *n_outbox_mpcd, 
    double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
    double *d_Fx, double *d_Fy, double *d_Fz, double *d_Fx_bend, double *d_Fy_bend, double *d_Fz_bend, double *d_Fx_stretch , double *d_Fy_stretch, double *d_Fz_stretch,
 double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, double *d_fb_kx, double *d_fb_ky, double *d_fb_kz, double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz, double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_Ax_tot_lab, double *d_Ay_tot_lab, double *d_Az_tot_lab, double *d_ex, double *d_ey, double *d_ez, double *h_fa_x, double *h_fa_y, double *h_fa_z, double *h_fb_x, double *h_fb_y, double *h_fb_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md, int Nmd, int m_md, int N, double mass, double mass_fluid, double *d_L , double ux, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, int delta, double real_time, double *gama_T, int *random_array, unsigned int seed, int topology, int *flag_array, double u_scale,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, double K_FENE, double K_bend, double K_l){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        
        Active_noslip_md_velocityverletKernel1(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min ,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab,
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        Active_noslip_calc_acceleration( d_mdX, d_mdY, d_mdZ, 
        d_Fx, d_Fy, d_Fz, d_Fx_bend, d_Fy_bend, d_Fz_bend, d_Fx_stretch, d_Fy_stretch, d_Fz_stretch,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd, m_md , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, Ax_cm, Ay_cm, Az_cm, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale, K_FENE, K_bend, K_l);


        sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

       

        
        //velocityverletKernel2 is called to complete the velocity verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        Active_noslip_md_velocityverletKernel2(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min ,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, 
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
        


        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;

        double *mdX, *mdY, *mdZ, *mdvx, *mdvy , *mdvz, *mdAx , *mdAy, *mdAz;
        //host allocation:
        mdX = (double*)malloc(sizeof(double) * Nmd);  mdY = (double*)malloc(sizeof(double) * Nmd);  mdZ = (double*)malloc(sizeof(double) * Nmd);
        mdvx = (double*)malloc(sizeof(double) * Nmd); mdvy = (double*)malloc(sizeof(double) * Nmd); mdvz = (double*)malloc(sizeof(double) * Nmd);
        cudaMemcpy(mdX , d_mdX, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdY , d_mdY, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdZ , d_mdZ, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdvx , d_mdvx, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdvy , d_mdvy, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdvz , d_mdvz, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        //std::cout<<potential(Nmd , mdX , mdY , mdZ , L , ux, h_md)+kinetinc(density,Nmd , mdvx , mdvy ,mdvz)<<std::endl;
        free(mdX);
        free(mdY);
        free(mdZ);

        
    }


}



__host__ void Active_noslip_MD_streaming2(double *d_mdX, double *d_mdY, double *d_mdZ, double *d_x, double *d_y, double *d_z,
    double *d_mdvx, double *d_mdvy, double *d_mdvz, double *d_vx, double *d_vy, double *d_vz, double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
    double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,  int *n_outbox_md, int *n_outbox_mpcd, 
    double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
    double *d_Fx, double *d_Fy, double *d_Fz, double *d_Fx_bending, double *d_Fy_bending, double *d_Fz_bending, double *d_Fx_stretching , double *d_Fy_stretching, double *d_Fz_stretching,
    double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, double *d_fb_kx, double *d_fb_ky, double *d_fb_kz, double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz, double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_Ax_tot_lab, double *d_Ay_tot_lab, double *d_Az_tot_lab, double *d_ex, double *d_ey, double *d_ez,
    double *h_fa_x, double *h_fa_y, double *h_fa_z, double *h_fb_x, double *h_fb_y, double *h_fb_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md, int Nmd, int m_md, int N, double mass, double mass_fluid, double *d_L , double ux, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, int delta, double real_time, double *gama_T, int *random_array, unsigned int seed, int topology, int *flag_array, double u_scale,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min_opp, double *md_dt_x_opp, double *md_dt_y_opp, double *md_dt_z_opp, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_o_opp, double *mdY_o_opp, double *mdZ_o_opp, double *mdvx_o_opp, double *mdvy_o_opp, double *mdvz_o_opp, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, int *hostErrorFlag, int *hostErrorFlag_opp, int *n_out_flag, int *n_out_flag_opp, double *d_zero, double K_FENE, double K_bend, double K_l){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        
        Active_noslip_md_velocityverletKernel5(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, Ax_cm, Ay_cm, Az_cm,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min, md_dt_x_opp, md_dt_y_opp, md_dt_z_opp, md_dt_min_opp,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, mdX_o_opp, mdY_o_opp, mdZ_o_opp, mdvx_o_opp, mdvy_o_opp, mdvz_o_opp, d_Ax_tot, d_Ay_tot, d_Az_tot, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab,
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, hostErrorFlag, hostErrorFlag_opp, n_out_flag, n_out_flag_opp, d_zero);
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        Active_noslip_calc_acceleration( d_mdX, d_mdY, d_mdZ, 
        d_Fx, d_Fy, d_Fz, d_Fx_bending, d_Fy_bending, d_Fz_bending, d_Fx_stretching, d_Fy_stretching, d_Fz_stretching,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd, m_md , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, d_Ax_tot_lab, d_Ay_tot_lab, d_Az_tot_lab, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, Ax_cm, Ay_cm, Az_cm, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale, K_FENE, K_bend, K_l);


        //sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

       

        
        //velocityverletKernel2 is called to complete the velocity verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        /*Active_noslip_md_velocityverletKernel4(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min ,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, 
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
        */


        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;

        double *mdX, *mdY, *mdZ, *mdvx, *mdvy , *mdvz, *mdAx , *mdAy, *mdAz;
        //host allocation:
        mdX = (double*)malloc(sizeof(double) * Nmd);  mdY = (double*)malloc(sizeof(double) * Nmd);  mdZ = (double*)malloc(sizeof(double) * Nmd);
        mdvx = (double*)malloc(sizeof(double) * Nmd); mdvy = (double*)malloc(sizeof(double) * Nmd); mdvz = (double*)malloc(sizeof(double) * Nmd);
        cudaMemcpy(mdX , d_mdX, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdY , d_mdY, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdZ , d_mdZ, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdvx , d_mdvx, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdvy , d_mdvy, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdvz , d_mdvz, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        //std::cout<<potential(Nmd , mdX , mdY , mdZ , L , ux, h_md)+kinetinc(density,Nmd , mdvx , mdvy ,mdvz)<<std::endl;
        free(mdX);
        free(mdY);
        free(mdZ);

        
    }


}


