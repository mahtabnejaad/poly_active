
// position and velocity of MD particles are initialized here ( currenty it only supports rings)
//This function initializes the position and velocity of MD (Molecular Dynamics) particles. The positions and velocities are stored in arrays d_mdX, d_mdY, d_mdZ, d_mdVx, d_mdVy, and d_mdVz.
// It also takes in various parameters like ux (velocity scaling factor), xx (initial position), m (number of particles in each ring), n (number of rings), topology (an integer representing the type of particle arrangement), and mass (mass of the particles).
//The ux parameter represents a velocity scaling factor used in generating initial velocities.

__host__ void initMD(double *d_mdX, double *d_mdY , double *d_mdZ ,
 double *d_mdVx , double *d_mdVy , double *d_mdVz, 
 double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *d_Fx_holder , double *d_Fy_holder, double *d_Fz_holder,
 double *d_L, double ux, double xx[3], int n, int m, int topology, double mass)
{
    int Nmd = n * m;//Nmd is the total number of MD particles, calculated as the product of n and m.
    //mdX, mdY, etc., are temporary arrays used for host-side initialization before transferring data to the GPU.
    double *mdX, *mdY, *mdZ, *mdVx, *mdVy , *mdVz, *mdAx , *mdAy, *mdAz;
    //host allocation:
    mdX = (double*)malloc(sizeof(double) * Nmd);  mdY = (double*)malloc(sizeof(double) * Nmd);  mdZ = (double*)malloc(sizeof(double) * Nmd);
    mdVx = (double*)malloc(sizeof(double) * Nmd); mdVy = (double*)malloc(sizeof(double) * Nmd); mdVz = (double*)malloc(sizeof(double) * Nmd);
    mdAx = (double*)malloc(sizeof(double) * Nmd); mdAy = (double*)malloc(sizeof(double) * Nmd); mdAz = (double*)malloc(sizeof(double) * Nmd);
    std::normal_distribution<double> normaldistribution(0, 0.44);//normaldistribution is an instance of the normal distribution with a mean of 0 and a standard deviation of 0.44. It will be used to generate random initial velocities.
    double theta = 4 * M_PI_2 / m;  //theta is the angle increment between particles in a ring.
    double r=m/(4 * M_PI_2);    //r is a scaling factor used to set the initial position of particles based on the chosen topology.
    //double r = 0.05;
    printf("r=%f\n", r);
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
            mdX[0]  = xx[0]; // + r * sin(i *theta);
            //monomer[i].x[1]  = xx[1] //+ r * cos(i *theta);
            mdY[0]  = xx[1]; // + r * cos(i *theta);
            //monomer[i].x[2]  = xx[2];
            mdZ[0]  = xx[2]; //pos is equal to {0,0,0} which is the origin of cartesian coordinates. this is the initial location of the MD single particle.
        }
    }
    if (topology == 1)  //poly [2] catenane
    {
        for (unsigned int j = 0 ; j< n ; j++) 
        //For each value of j (ring index), and for each value of i (particle index within the ring), properties like mdAx, mdAy, mdAz, mdVx, mdVy, and mdVz are initialized to zero.
        //The velocity components mdVx, mdVy, and mdVz are initialized with random values from the normal distribution using normaldistribution(generator)
        {
            
            for (unsigned int i =0 ; i<m ; i++)
            {
                
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);
                if ( j%2 == 0 )
                {
                    mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdZ[i+j*m]  = xx[2];
                }
                if(j%2==1)
                {
                    mdZ[i+j*m]  = xx[2] + r * cos(i *theta);
                    //monomer[i].x[2]  = xx[2];
                    mdY[i+j*m]  = xx[1];

                }

            
            }   
            //The variable xx is incremented by 1.2 * r after each complete ring to adjust the position of the rings.
            xx[0]+=1.2*r;
        }
    }
    if (topology == 2)  //linked rings
    {
        for (unsigned int j = 0 ; j< n ; j++)
        {
            
            for (unsigned int i =0 ; i<m ; i++)
            {
                
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*m]  = xx[2];

            
            }
            
            xx[0]+=(2*r+1) ;
        }
    }   
   
    if (topology == 3)
    {
        for (unsigned int j = 0 ; j< n ; j++)
        {
            
            for (unsigned int i =0 ; i<m ; i++)
            {
                
                mdAx[i+j*m]=0;
                mdAy[i+j*m]=0;
                mdAz[i+j*m]=0;
                //monomer[i].init(kT ,box, mass);
                mdVx[i+j*m] = normaldistribution(generator);
                mdVy[i+j*m] = normaldistribution(generator);
                mdVz[i+j*m] = normaldistribution(generator);
                //monomer[i].x[0]  = xx[0] + r * sin(i *theta);
                mdX[i+j*m]  = xx[0] + r * sin(i *theta);
                //monomer[i].x[1]  = xx[1] + r * cos(i *theta);

                mdY[i+j*m]  = xx[1] + r * cos(i *theta);
                //monomer[i].x[2]  = xx[2];
                mdZ[i+j*m]  = xx[2];

            
            }
            
            xx[0]+=(2.5*r+1) ;
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
    cudaMemcpy(d_mdX ,mdX, Nmd*sizeof(double), cudaMemcpyHostToDevice);   cudaMemcpy(d_mdY ,mdY, Nmd*sizeof(double), cudaMemcpyHostToDevice);   cudaMemcpy(d_mdZ ,mdZ, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdVx ,mdVx, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_mdVy ,mdVy, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_mdVz ,mdVz, Nmd*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdAx ,mdAx, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_mdAy ,mdAy, Nmd*sizeof(double), cudaMemcpyHostToDevice); cudaMemcpy(d_mdAz ,mdAz, Nmd*sizeof(double), cudaMemcpyHostToDevice);


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

//calculating interaction matrix of the system in the given time when BC is periodic
__global__ void nb_b_interaction( 
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
        LeeEdwNearestImage(mdX[ID1], mdY[ID1], mdZ[ID1] , mdX[ID2] , mdY[ID2] , mdZ[ID2] , r,L, ux, real_time);
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
        }
    
        /*else
        {
            fx[tid] = 0;
            fy[tid] = 0;
            fz[tid] = 0;
        }*/
      

    }

}

__global__ void bending_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *fx_bend, double *fy_bend, double *fz_bend,
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


__host__ void calc_accelaration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz,
double *Fx_bend , double *Fy_bend , double *Fz_bend,
double *Ax , double *Ay , double *Az,
double *L,int size ,int m ,int topology, double ux,double real_time, int grid_size, double K_FENE, double K_bend)
{
    nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux,density, real_time , m , topology, K_FENE, K_bend);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    bending_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz , Fx_bend, Fy_bend, Fz_bend, L , size , ux,density, real_time , m , topology, K_FENE, K_bend);
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
    double *d_Fx_bend, double *d_Fy_bend, double *d_Fz_bend,
    double h_md ,int Nmd, int density, double *d_L ,double ux,int grid_size ,int delta, double real_time, double K_FENE, double K_bend)
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
        calc_accelaration(d_mdX, d_mdY , d_mdZ , d_Fx , d_Fy , d_Fz , d_Fx_bend, d_Fy_bend, d_Fz_bend, d_mdAx , d_mdAy , d_mdAz, d_L , Nmd ,m_md ,topology, ux ,real_time, grid_size, K_FENE, K_bend);
        
        
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







