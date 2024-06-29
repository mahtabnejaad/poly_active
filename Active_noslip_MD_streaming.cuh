

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





__host__ void noslip_monomer_active_backward_forces(double *mdX, double *mdY , double *mdZ ,
double *Ax, double *Ay, double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T,
double *L, int size, double mass_fluid, double real_time, int m, int topology, int grid_size, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array,double u_scale)
{
    double Q = -mass/(size*mass+mass_fluid*N);
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

        totalActive_calc_acceleration<<<grid_size,blockSize>>>(Ax, Ay, Az, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, flag_array, Ax_tot, Ay_tot, Az_tot, 1, topology);
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
           
    
            //*fa_x=*fa_x* *gama_T*u_scale;
            //*fa_y=*fa_y* *gama_T*u_scale;
            //*fa_z=*fa_z* *gama_T*u_scale;
            *fb_x=*fa_x*Q;
            *fb_y=*fa_y*Q;
            *fb_z=*fa_z*Q;

            cudaFree(gamaT);
            free(sumx);  free(sumy);  free(sumz);
     
        }
  
    }
}






//calculating interaction matrix of the system in the given time when BC is periodic
__global__ void Active_noslip_nb_b_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *L,int size , double ux, double mass, double real_time, int m , int topology)
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
        
            if (topology == 1)
            {
                if (int(ID1/m) == int(ID2/m)) //checks if the interacting particles belong to the same chain (monomer). This is achieved by dividing the particle indices by m (monomer size) and checking if they are in the same division.
                {
                    //check if the interacting particles are next to each other in the same chain. If they are, it calculates the FENE interaction contribution,
                    if( ID2 - ID1 == 1 || ID2 - ID1 == m-1 ) 
                    {
                        f -= 30/(1 - r_sqr/2.25);
                    }

                    if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                    {
                        f -= 30/(1 - r_sqr/2.25);
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
                        f -= 30/(1 - r_sqr/2.25);
                    }

                    if( ID1 - ID2 == 1 || ID1 - ID2 == m-1 ) 
                    {
                        f -= 30/(1 - r_sqr/2.25);
                    }
                }
            
                if (ID1==int(m/4) && ID2 ==m+int(3*m/4))
                {
                
                    f -= 30/(1 - r_sqr/2.25);
                }
                
                if (ID2==int(m/4) && ID1 ==m+int(3*m/4))
                {
                    f -= 30/(1 - r_sqr/2.25);
                }
            }
            f/=mass; //After the interaction forces are calculated (f), they are divided by the mass of the particles to obtain the correct acceleration.

            fx[tid] = f * r[0] ;
            fy[tid] = f * r[1] ;
            fz[tid] = f * r[2] ;

            //printf("fx[%i]=%f, fy[%i]=%f, fz[%i]=%f\n", tid, fx[tid], tid, fy[tid], tid, fz[tid]);
            }
    
            /*else
            {
                fx[tid] = 0;
                fy[tid] = 0;
                fz[tid] = 0;
            }*/
      

        }

    }
}


//Active_noslip_calc_acceleration

    __host__ void Active_noslip_calc_acceleration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz, 
double *Ax , double *Ay , double *Az,double *fa_kx, double *fa_ky, double *fa_kz, double *fb_kx, double *fb_ky, double *fb_kz,
double *Aa_kx, double *Aa_ky, double *Aa_kz,double *Ab_kx, double *Ab_ky, double *Ab_kz, double *ex, double *ey, double *ez, double ux, double mass, double *gama_T, 
double *L, int size, int m, int topology, double real_time, int grid_size, double mass_fluid, int N, int *random_array, unsigned int seed, double *Ax_tot, double *Ay_tot, double *Az_tot, double *fa_x, double *fa_y, double *fa_z,double *fb_x, double *fb_y, double *fb_z, double *block_sum_ex, double *block_sum_ey, double *block_sum_ez, int *flag_array, double u_scale)

{
  

    Active_noslip_nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux, mass, real_time , m , topology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    sum_kernel<<<grid_size,blockSize>>>(Fx , Fy, Fz, Ax , Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //printf("**GAMA=%f\n",*gama_T);
    

    noslip_monomer_active_backward_forces(x, y ,z ,
    Ax , Ay, Az, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, ex, ey, ez, ux, mass, gama_T, 
    L, size , mass_fluid, real_time, m, topology, grid_size, N , random_array, seed , Ax_tot, Ay_tot, Az_tot, fa_x, fa_y, fa_z, fb_x, fb_y, fb_z, block_sum_ex, block_sum_ey, block_sum_ez, flag_array, u_scale);
    


    
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
    printf("md_dt_x[%i]=%f, md_dt_y[%i]=%f, md_dt_z[%i]=%f\n", tid, md_dt_x[tid], tid, md_dt_y[tid], tid, md_dt_z[tid]);
    printf("mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f\n", tid, mdvx[tid], tid, mdvy[tid], tid, mdvz[tid]);
    printf("mdAx_tot[%i]=%f, mdAy_tot[%i]=%f, mdAz_tot[%i]=%f\n", tid, mdAx_tot[tid], tid, mdAy_tot[tid], tid, mdAz_tot[tid]);
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

//Active_md_velocityverlet1
__global__ void Active_md_velocityverlet1(double *mdX, double *mdY , double *mdZ , 
double *mdVx , double *mdVy , double *mdVz,
double *mdAx_tot , double *mdAy_tot , double *mdAz_tot,
 double h, int Nmd)
{
    int particleID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (particleID < Nmd)
    {
        // Particle velocities are updated by half a time step, and particle positions are updated based on the new velocities.

        
        mdX[particleID] = mdX[particleID] + h * mdVx[particleID] + 0.5 * h * h * mdAx_tot[particleID];
        mdY[particleID] = mdY[particleID] + h * mdVy[particleID] + 0.5 * h * h * mdAy_tot[particleID];
        mdZ[particleID] = mdZ[particleID] + h * mdVz[particleID] + 0.5 * h * h * mdAz_tot[particleID];

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
__global__ void particles_on_crossing_points(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *md_dt_min, double dt, double *L, int Nmd){



    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

        if(md_dt_min[tid] < md_dt){
            //make the position of particle equal to (xo, yo, zo):
            mdx[tid] = mdx_o[tid];
            mdy[tid] = mdy_o[tid];
            mdz[tid] = mdz_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdvx[tid] = -mdvx_o[tid];
            mdvy[tid] = -mdvy_o[tid];
            mdvz[tid] = -mdvz_o[tid];
        }
    }

}

//Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1
__global__ void Active_CM_md_bounceback_velocityverlet1(double *mdx, double *mdy, double *mdz, double *mdx_o, double *mdy_o, double *mdz_o, double *mdvx, double *mdvy, double *mdvz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdAx_tot, double *mdAy_tot, double *mdAz_tot, double *md_dt_min, double md_dt, double *L, int Nmd, double *Xcm, double *Ycm, double *Zcm, int *errorFlag){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<Nmd){

     

        //if(mdx[tid]>L[0]/2 || mdx[tid]<-L[0]/2 || mdy[tid]>L[1]/2 || mdy[tid]<-L[1]/2 || mdz[tid]>L[2]/2 || mdz[tid]<-L[2]/2){
        if(md_dt_min[tid] < md_dt){
            
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

    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdX_o, mdY_o, mdZ_o, md_dt_min, h_md, L, d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min,  d_Ax_tot, d_Ay_tot, d_Az_tot, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_outside_particles
    outerParticles_CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
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
double *mdvx, double *mdvy, double *mdvz, double *vx, double *vy, double *vz, double *mdAx , double *mdAy , double *mdAz,
double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *X_tot, double *Y_tot, double *Z_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, int *dn_md_tot, int *dn_mpcd_tot,
double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz, double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz, double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, int *CMsumblock_n_outbox_md, int *CMsumblock_n_outbox_mpcd,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out,
double h_md, int Nmd, int N, int *n_outbox_md, int *n_outbox_mpcd, double mass, double mass_fluid, double *L, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, 
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, int *hostErrorFlag){

    //CM_system
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd);
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

    //gotoCMframe
    //gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    //Take_o_to_CM_system<<<grid_size,blockSize>>>(mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Nmd);
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    
    
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd);
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

    particles_on_crossing_points<<<grid_size,blockSize>>>(mdx, mdy, mdz, mdx_o, mdy_o, mdz_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, dt, L, Nmd);
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

    //CM_system
    CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    //gotoCMframe
    //gotoCMframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    int *d_errorFlag;
    *hostErrorFlag = 0;
    cudaMalloc(&d_errorFlag, sizeof(int));
    cudaMemcpy(d_errorFlag, hostErrorFlag, sizeof(int), cudaMemcpyHostToDevice);


    //put the particles that had traveled outside of the box , on box boundaries.
    Active_CM_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot, md_dt_min, h_md, L, Nmd, Xcm, Ycm, Zcm, d_errorFlag);
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

    //gotoLabFrame for mpcd particles:
    //backtoLabframe<<<grid_size,blockSize>>>(x, y, z, Xcm, Ycm, Zcm, vx, vy, vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    backtoLabframe<<<grid_size,blockSize>>>(mdX, mdY, mdZ, Xcm, Ycm, Zcm, mdvx, mdvy, mdvz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    

    cudaFree(d_errorFlag);

    
   
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
    outerParticles_CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
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
    double *d_Fx, double *d_Fy, double *d_Fz, double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, double *d_fb_kx, double *d_fb_ky, double *d_fb_kz, double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz, double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_ex, double *d_ey, double *d_ez, double *h_fa_x, double *h_fa_y, double *h_fa_z, double *h_fb_x, double *h_fb_y, double *h_fb_z, double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md, int Nmd, int m_md, int N, double mass, double mass_fluid, double *d_L , double ux, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, int delta, double real_time, double *gama_T, int *random_array, unsigned int seed, int topology, int *flag_array, double u_scale,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        
        Active_noslip_md_velocityverletKernel1(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min ,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot,
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        Active_noslip_calc_acceleration( d_mdX, d_mdY, d_mdZ, 
        d_Fx, d_Fy, d_Fz,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd, m_md , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);


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
    double *d_Fx, double *d_Fy, double *d_Fz, double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, double *d_fb_kx, double *d_fb_ky, double *d_fb_kz, double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz, double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz, double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot, double *d_ex, double *d_ey, double *d_ez, double *h_fa_x, double *h_fa_y, double *h_fa_z, double *h_fb_x, double *h_fb_y, double *h_fb_z, double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
    double h_md, int Nmd, int m_md, int N, double mass, double mass_fluid, double *d_L , double ux, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_, int delta, double real_time, double *gama_T, int *random_array, unsigned int seed, int topology, int *flag_array, double u_scale,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, int *hostErrorFlag){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        
        Active_noslip_md_velocityverletKernel3(d_mdX, d_mdY , d_mdZ, d_x, d_y, d_z,
        d_mdvx, d_mdvy, d_mdvz, d_vx, d_vy, d_vz, d_mdAx, d_mdAy, d_mdAz,
        mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_md_tot, dn_mpcd_tot,
        CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_n_outbox_md, CMsumblock_n_outbox_mpcd,
        Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out,
        h_md, Nmd, N, n_outbox_md, n_outbox_mpcd, mass, mass_fluid, d_L, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, md_dt_x, md_dt_y, md_dt_z, md_dt_min ,
        mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, d_Ax_tot, d_Ay_tot, d_Az_tot,
        mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, hostErrorFlag);
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        Active_noslip_calc_acceleration( d_mdX, d_mdY, d_mdZ, 
        d_Fx, d_Fy, d_Fz,
        d_mdAx , d_mdAy, d_mdAz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz,
        d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez,
        ux, mass, gama_T, d_L, Nmd, m_md , topology, real_time,  grid_size, mass_fluid, N, random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);


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


