
// position and velocity of MD particles are initialized here ( currenty it only supports rings)
//This function initializes the position and velocity of MD (Molecular Dynamics) particles. The positions and velocities are stored in arrays d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, and d_mdVz.
// It also takes in various parameters like ux (velocity scaling factor), xx (initial position), m (number of particles in each ring), n (number of rings), topology (an integer representing the type of particle arrangement), and mass (mass of the particles).
//The ux parameter represents a velocity scaling factor used in generating initial velocities.
__host__ void initMD(double *Id_mdX, double *Id_mdY , double *Id_mdZ ,
 double *Id_mdVx , double *Id_mdVy , double *Id_mdVz, 
 double *Id_mdAx , double *Id_mdAy , double *Id_mdAz,
double *Id_Fx_holder , double *Id_Fy_holder, double *Id_Fz_holder,
 double *Id_L, double Iux, double Ixx[3], int In, int Im, int Itopology, int Imass)
{
    int Nmd = In * Im;//Nmd is the total number of MD particles, calculated as the product of n and m.
    //mdX, mdY, etc., are temporary arrays used for host-side initialization before transferring data to the GPU.
    double *mdX, *mdY, *mdZ, *mdVx, *mdVy , *mdVz, *mdAx , *mdAy, *mdAz;
    //host allocation:
    mdX = (double*)malloc(sizeof(double) * Nmd);  mdY = (double*)malloc(sizeof(double) * Nmd);  mdZ = (double*)malloc(sizeof(double) * Nmd);
    mdVx = (double*)malloc(sizeof(double) * Nmd); mdVy = (double*)malloc(sizeof(double) * Nmd); mdVz = (double*)malloc(sizeof(double) * Nmd);
    mdAx = (double*)malloc(sizeof(double) * Nmd); mdAy = (double*)malloc(sizeof(double) * Nmd); mdAz = (double*)malloc(sizeof(double) * Nmd);
    std::normal_distribution<double> normaldistribution(0, 0.44);//normaldistribution is an instance of the normal distribution with a mean of 0 and a standard deviation of 0.44. It will be used to generate random initial velocities.
    double theta = 4 * M_PI_2 / Im;  //theta is the angle increment between particles in a ring.
    double r=Im/(4 * M_PI_2);    //r is a scaling factor used to set the initial position of particles based on the chosen topology.
    if (topology == 4) //one particle
    {
        
        {
            mdAx[0]=0;
            mdAy[0]=0;
            mdAz[0]=0;
            //monomer[i].init(kT ,box, mass);
            //mdVx[0] = normaldistribution(generator);
            //mdVy[0] = normaldistribution(generator);
            //mdVz[0] = normaldistribution(generator);

            mdVx[0] = 0;
            mdVy[0] = 0;
            mdVz[0] = 0; 

            //monomer[i].x[0]  = xx[0] //+ r * sin(i *theta);
            mdX[0]  = Ixx[0]; // + r * sin(i *theta);
            //monomer[i].x[1]  = xx[1] //+ r * cos(i *theta);
            mdY[0]  = Ixx[1]; // + r * cos(i *theta);
            //monomer[i].x[2]  = xx[2];
            mdZ[0]  = Ixx[2]; //pos is equal to {0,0,0} which is the origin of cartesian coordinates. this is the initial location of the MD single particle.
        }
    }
    if (Itopology == 1)  //poly [2] catenane
    {
        for (unsigned int j = 0 ; j< In ; j++) 
        //For each value of j (ring index), and for each value of i (particle index within the ring), properties like mdAx, mdAy, mdAz, mdVx, mdVy, and mdVz are initialized to zero.
        //The velocity components mdVx, mdVy, and mdVz are initialized with random values from the normal distribution using normaldistribution(generator)
        {
            
            for (unsigned int i =0 ; i<Im ; i++)
            {
                
                mdAx[i+j*Im]=0;
                mdAy[i+j*Im]=0;
                mdAz[i+j*Im]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*Im] = normaldistribution(generator);
                mdVy[i+j*Im] = normaldistribution(generator);
                mdVz[i+j*Im] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*Im]  = Ixx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);
                if ( j%2 == 0 )
                {
                    mdY[i+j*Im]  = Ixx[1] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdZ[i+j*Im]  = Ixx[2];
                }
                if(j%2==1)
                {
                    mdZ[i+j*Im]  = Ixx[2] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdY[i+j*Im]  = Ixx[1];

                }

            
            }   
            //The variable xx is incremented by 1.2 * r after each complete ring to adjust the position of the rings.
            Ixx[0]+=1.2*r;
        }
    }
    if (Itopology == 2)  //linked rings
    {
        for (unsigned int j = 0 ; j< In ; j++)
        {
            
            for (unsigned int i =0 ; i<Im ; i++)
            {
                
                mdAx[i+j*Im]=0;
                mdAy[i+j*Im]=0;
                mdAz[i+j*Im]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*Im] = normaldistribution(generator);
                mdVy[i+j*Im] = normaldistribution(generator);
                mdVz[i+j*Im] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*Im]  = Ixx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*Im]  = Ixx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*Im]  = Ixx[2];

            
            }
            
            Ixx[0]+=(2*r+1) ;
        }
    }   
   
    if (Itopology == 3)
    {
        for (unsigned int j = 0 ; j< In ; j++)
        {
            
            for (unsigned int i =0 ; i<Im ; i++)
            {
                
                mdAx[i+j*Im]=0;
                mdAy[i+j*Im]=0;
                mdAz[i+j*Im]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*Im] = normaldistribution(generator);
                mdVy[i+j*Im] = normaldistribution(generator);
                mdVz[i+j*Im] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*Im]  = Ixx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*Im]  = Ixx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*Im]  = Ixx[2];

            
            }
            
            Ixx[0]+=(2.5*r+1) ;
        } 
                
    }
    //This section calculates and subtracts the center-of-mass velocity from the particle velocities to ensure the system's total momentum is zero.
    double px =0 , py =0 ,pz =0;
    for (unsigned int i =0 ; i<Nmd ; i++)
    {
        px+=mdVx[i] ; 
        py+=mdVy[i] ; 
        pz+=mdVz[i] ;
    }

    for (unsigned int i =0 ; i<Nmd ; i++)
    {
        mdVx[i]-=px/Nmd ;
        mdVy[i]-=py/Nmd ;
        mdVz[i]-=pz/Nmd ;
    }
    //the arrays mdX, mdY, etc., containing particle properties are transferred from the host to the GPU  using cudaMemcpy.
    cudaMemcpy(Id_mdX ,mdX, Nmd*sizeof(double), cudaMemcpyHostToDevice);   cudaMemcpy(Id_mdY ,mdY, Nmd*sizeof(double), cudaMemcpyHostToDevice);   cudaMemcpy(Id_mdZ ,mdZ, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Id_mdVx ,mdVx, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(Id_mdVy ,mdVy, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(Id_mdVz ,mdVz, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Id_mdAx ,mdAx, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(Id_mdAy ,mdAy, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(Id_mdAz ,mdAz, Nmd*sizeof(double), cudaMemcpyHostToDevice);


    //the dynamically allocated host memory is freed to avoid memory leaks.
    free(mdX);    free(mdY);    free(mdZ);
    free(mdVx);   free(mdVy);   free(mdVz);
    free(mdAx);   free(mdAy);   free(mdAz);

}


// a tool for resetting a vector to zero!
//This is a simple kernel that resets the values of three arrays F1, F2, and F3 to zero. It uses the thread ID to determine the array index and sets the values to zero.
__global__ void reset_vector_to_zero(double *F1 , double *F2 , double *F3 , int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<size)
    {
        F1[tid] = 0 ;
        F2[tid] = 0 ;
        F3[tid] = 0 ;
    }
}
//sum kernel: is used for sum the interaction matrices:F in one axis and calculate acceleration.
//This kernel is used to sum the interaction matrix F along each axis and calculate the  (A).
//The kernel takes in F1, F2, and F3 (interaction matrices) and calculates the sum along each row. The resulting sums are stored in A1, A2, and A3, respectively.
//The size parameter determines the size of the interaction matrix.
__global__ void sum_kernel(double *F1 ,double *F2 , double *F3,
 double *A1 ,double *A2 , double *A3,
  int Ssize)
{
    //size=Nmd
    // The three if conditions ensure that each thread processes a specific range of particles and calculates the accelerations separately for each axis.
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;//The thread ID corresponds to the particle index.
    if (tid<Ssize)  //The first if statement ensures that each thread processes particles from index 0 to size - 1. This is where the calculation for A1 happens.
    {
        double sum =0;
        for (int i = 0 ; i<Ssize ; ++i)
        {
            int index = tid *Ssize +i;//For each particle tid, it iterates through all particles (i) and accumulates the force value from the F1 array.
                                     // The index is calculated to access the correct force value for the current particle and the current particle being looped over.
            sum += F1[index];  //After the loop, the calculated sum is stored in the A1 array at the index corresponding to the particle's index (tid).
        }
        A1[tid] = sum;
    }
    if (Ssize<tid+1 && tid<2*Ssize) //The second if statement handles particles from index size to 2 * size - 1. The tid is adjusted to be within this range, and then the calculation for A2 is performed.
    {
        tid -=Ssize;
        double sum =0;
        for (int i = 0 ; i<Ssize ; ++i)
        {
            int index = tid *Ssize +i ;
            sum += F2[index];
        }
        A2[tid] = sum;        
    }
    if (2*Ssize<tid+1 && tid<3*Ssize) //The third if statement handles particles from index 2 * size to 3 * size - 1. Similar to the second if statement, tid is adjusted and the calculation for A3 is performed.
    {
        tid -=2*Ssize;
        double sum =0;
        for (int i = 0 ; i<Ssize ; ++i)
        {
            int index = tid *Ssize +i;
            sum += F3[index];
        }
        A3[tid] = sum;        
    }

}

//calculating interaction matrix of the system in the given time
__global__ void nb_b_interaction( 
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
        LeeEdwNearestImage(NmdX[ID1], NmdY[ID1], NmdZ[ID1] , NmdX[ID2] , NmdY[ID2] , NmdZ[ID2] , r,NL, Nux, Nreal_time);
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

__host__ void calc_accelaration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz,
double *Ax , double *Ay , double *Az,
double *L,int size ,int m ,int topology, double ux,double real_time, int grid_size)
{
    nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux,density, real_time , m , topology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sum_kernel<<<grid_size,blockSize>>>(Fx ,Fy,Fz, Ax ,Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}



//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void velocityVerletKernel2(double *VmdVx , double *VmdVy , double *VmdVz,
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

__global__ void velocityVerletKernel1(double *mdX, double *mdY , double *mdZ , 
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


// the MD_streaming function represents a time-stepping loop in molecular dynamics simulations
__host__ void MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdVx, double *d_mdVy, double *d_mdVz,
    double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *d_Fx, double *d_Fy, double *d_Fz,
    double h_md ,int Nmd, int density, double *d_L ,double ux,int grid_size ,int delta, double real_time)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {

        
        velocityVerletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz , h_md,Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        //After updating particle positions, a kernel named LEBC is called to apply boundary conditions to ensure that particles stay within the simulation box.
        LEBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx , ux , d_L, real_time , Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //a function to consider non slip boundary conditions in y and z planes and have periodic BC in x plane.
        //nonslipXperiodicBC<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdVx ,d_mdVy, d_mdVz, ux , d_L, real_time , Nmd);
        //gpuErrchk( cudaPeekAtLastError() );
        //gpuErrchk( cudaDeviceSynchronize() );
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        calc_accelaration(d_mdX, d_mdY , d_mdZ , d_Fx , d_Fy , d_Fz , d_mdAx , d_mdAy , d_mdAz, d_L , Nmd ,m_md ,topology, ux ,real_time, grid_size);
        
        
        //velocityVerletKernel2 is called to complete the velocity Verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        velocityVerletKernel2<<<grid_size,blockSize>>>(d_mdVx, d_mdVy, d_mdVz, d_mdAx, d_mdAy, d_mdAz, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;

        double *mdX, *mdY, *mdZ, *mdVx, *mdVy , *mdVz, *mdAx , *mdAy, *mdAz;
        //host allocation:
        mdX = (double*)malloc(sizeof(double) * Nmd);  mdY = (double*)malloc(sizeof(double) * Nmd);  mdZ = (double*)malloc(sizeof(double) * Nmd);
        mdVx = (double*)malloc(sizeof(double) * Nmd); mdVy = (double*)malloc(sizeof(double) * Nmd); mdVz = (double*)malloc(sizeof(double) * Nmd);
        cudaMemcpy(mdX , d_mdX, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdY , d_mdY, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdZ , d_mdZ, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mdVx , d_mdVx, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdVy , d_mdVy, Nmd*sizeof(double), cudaMemcpyDeviceToHost);   cudaMemcpy(mdVz , d_mdVz, Nmd*sizeof(double), cudaMemcpyDeviceToHost);
        std::cout<<potential(Nmd , mdX , mdY , mdZ , L , ux, h_md)+kinetinc(density,Nmd , mdVx , mdVy ,mdVz)<<std::endl;
        free(mdX);
        free(mdY);
        free(mdZ);

        
    }
}







