#include <stdlib.h>
__device__ __host__ void regular_distance(double x0,double x1, double x2, double y0, double y1, double y2, double *r,double *L,double ux, double t){

    r[0] = x0 - y0;
    r[1] = x1 - y1;
    r[2] = x2 - y2;

}


//calculating interaction matrix of the system in the given time when BC is periodic
__global__ void noslip_nb_b_interaction( 
double *mdX, double *mdY , double *mdZ ,
double *fx , double *fy , double *fz, 
double *L,int size , double ux, int mass, double real_time, int m , int topology)
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
        }
    
        /*else
        {
            fx[tid] = 0;
            fy[tid] = 0;
            fz[tid] = 0;
        }*/
      

    }

}

__host__ void noslip_calc_accelaration( double *x ,double *y , double *z , 
double *Fx , double *Fy , double *Fz,
double *Ax , double *Ay , double *Az,
double *L,int size ,int m ,int topology, double ux,double real_time, int grid_size)
{
    noslip_nb_b_interaction<<<grid_size,blockSize>>>(x , y , z, Fx , Fy , Fz ,L , size , ux,density, real_time , m , topology);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sum_kernel<<<grid_size,blockSize>>>(Fx ,Fy,Fz, Ax ,Ay, Az, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}



//a function to consider velocity sign of particles and determine which sides of the box it should interact with 
__global__ void md_wall_sign(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if (vx[tid] > 0 )  wall_sign_x[tid] = 1;
        else if (vx[tid] < 0)  wall_sign_x[tid] = -1;
        else if(vx[tid] == 0)  wall_sign_x[tid] = 0;
        
        if (vy[tid] > 0 ) wall_sign_y[tid] = 1;
        else if (vy[tid] < 0) wall_sign_y[tid] = -1;
        else if (vy[tid] == 0)  wall_sign_y[tid] = 0;

        if (vz[tid] > 0) wall_sign_z[tid] = 1;
        else if (vz[tid] < 0) wall_sign_z[tid] = -1;
        else if (vz[tid] == 0)  wall_sign_z[tid] = 0;

        (isnan(vx[tid])|| isnan(vy[tid]) || isnan(vz[tid])) ? printf(" MD.. vx[%i]=%f, vy[%i]=%f, vz[%i]=%f \n", tid, vx[tid], tid, vy[tid], tid, vz[tid])
                                                            : printf("");


    }

}
//a function to calculate distance of particles which are inside the box from the corresponding walls:
__global__ void md_distance_from_walls(double *x, double *y, double *z, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *L, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if (wall_sign_x[tid] == 1)   x_wall_dist[tid] = L[0]/2-(x[tid]);
        else if (wall_sign_x[tid] == -1)  x_wall_dist[tid] = L[0]/2+(x[tid]);
        else if(wall_sign_x[tid] == 0)  x_wall_dist[tid] = L[0]/2 -(x[tid]);//we can change it as we like . it doesn't matter.


        if (wall_sign_y[tid] == 1)   y_wall_dist[tid] = L[1]/2-(y[tid]);
        else if (wall_sign_y[tid] == -1)  y_wall_dist[tid] = L[1]/2+(y[tid]);
        else if(wall_sign_y[tid] == 0)  y_wall_dist[tid] = L[1]/2 -(y[tid]);//we can change it as we like . it doesn't matter.


        if (wall_sign_z[tid] == 1)   z_wall_dist[tid] = L[2]/2-(z[tid]);
        else if (wall_sign_z[tid] == -1)  z_wall_dist[tid] = L[2]/2+(z[tid]);
        else if(wall_sign_z[tid] == 0)  z_wall_dist[tid] = L[2]/2 -(z[tid]);//we can change it as we like . it doesn't matter.



        //printf("***dist_x[%i]=%f, dist_y[%i]=%f, dist_z[%i]=%f\n", tid, x_wall_dist[tid], tid, y_wall_dist[tid], tid, z_wall_dist[tid]);
        int idxx;
        idxx = (int(x[tid] + L[0] / 2 + 2) + (L[0] + 4) * int(y[tid] + L[1] / 2 + 2) + (L[0] + 4) * (L[1] + 4) * int(z[tid] + L[2] / 2 + 2));
        //printf("index[%i]=%i, x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, idxx, tid, x[tid], tid, y[tid], tid, z[tid]);//checking

        (isnan(x_wall_dist[tid])|| isnan(y_wall_dist[tid]) || isnan(z_wall_dist[tid])) ? printf("x_wall_dist[%i]=%f, y_wall_dist[%i]=%f, z_wall_dist[%i]=%f \n", tid, x_wall_dist[tid], tid, y_wall_dist[tid], tid, z_wall_dist[tid])
                                                            : printf("");

        

    }    


}






//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void md_deltaT(double *mdvx, double *mdvy, double *mdvz, double *mdAx, double *mdAy, double *mdAz, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *md_dt_x, double *md_dt_y, double *md_dt_z, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){

        printf("mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f \n", ID, mdvx[ID], ID, mdvy[ID], ID, mdvz[ID]);
        printf("mdAx[%i]=%f, mdAy[%i]=%f, mdAz[%i]=%f \n", ID, mdAx[ID], ID, mdAy[ID], ID, mdAz[ID]);

        (isnan(md_dt_x[ID])|| isnan(md_dt_y[ID]) || isnan(md_dt_z[ID])) ? printf("md_dt_x[%i]=%f, md_dt_y[%i]=%f, md_dt_z[%i]=%f \n", ID, md_dt_x[ID], ID, md_dt_y[ID], ID, md_dt_z[ID])
                                                            : printf("");
        (isnan(mdvx[ID])|| isnan(mdvy[ID]) || isnan(mdvz[ID])) ? printf("mdvx[%i]=%f, mdvy[%i]=%f, mdvz[%i]=%f \n", ID, mdvx[ID], ID, mdvy[ID], ID, mdvz[ID])
                                                            : printf("");
         (isnan(mdAx[ID])|| isnan(mdAy[ID]) || isnan(mdAz[ID])) ? printf("mdAx[%i]=%f, mdAy[%i]=%f, mdAz[%i]=%f \n", ID, mdAx[ID], ID, mdAy[ID], ID, mdAz[ID])
                                                            : printf("");
         (isnan(mdX_wall_dist[ID])|| isnan(mdY_wall_dist[ID]) || isnan(mdZ_wall_dist[ID])) ? printf("mdX_wall_dist[%i]=%f, mdY_wall_dist[%i]=%f, mdZ_wall_dist[%i]=%f \n", ID, mdX_wall_dist[ID], ID, mdY_wall_dist[ID], ID, mdZ_wall_dist[ID])
                                                            : printf("");
        if(wall_sign_mdX[ID] == 0 ) md_dt_x[ID] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_mdX[ID] == 1 || wall_sign_mdX[ID] == -1){
            
            if(mdAx[ID]  == 0.0)   md_dt_x[ID] = abs(mdX_wall_dist[ID]/mdvx[ID]);

            else if (mdAx[ID]  != 0.0)  md_dt_x[ID] = ((-mdvx[ID]+sqrt(abs((mdvx[ID]*mdvx[ID])+(2*mdX_wall_dist[ID]*mdAx[ID]))))/mdAx[ID]);

        }  

        if(wall_sign_mdY[ID] == 0 ) md_dt_y[ID] == 10000;
        else if(wall_sign_mdY[ID] == 1 || wall_sign_mdY[ID] == -1){

            if(mdAy[ID] == 0.0)   md_dt_y[ID] = abs(mdY_wall_dist[ID]/mdvy[ID]);

            else if(mdAy[ID]  != 0.0)   md_dt_y[ID] = ((-mdvy[ID]+sqrt(abs((mdvy[ID]*mdvy[ID])+(2*mdY_wall_dist[ID]*mdAy[ID]))))/mdAy[ID]);

        }

        if(wall_sign_mdZ[ID] == 0 ) md_dt_z[ID] == 10000;
        else if(wall_sign_mdZ[ID] == 1 || wall_sign_mdZ[ID] == -1){

            if(mdAz[ID]  == 0.0)   md_dt_z[ID] = abs(mdZ_wall_dist[ID]/mdvz[ID]);

            else if(mdAz[ID]  != 0.0)   md_dt_z[ID] = ((-mdvz[ID]+sqrt(abs((mdvz[ID]*mdvz[ID])+(2*mdZ_wall_dist[ID]*mdAz[ID]))))/mdAz[ID]);

        }

        

    }


}

//calculate the crossing location where the particles intersect with one wall:
__global__ void md_crossing_location(double *mdX, double *mdY, double *mdZ, double *mdvx, double *mdvy, double *mdvz, double *mdAx, double *mdAy, double *mdAz, double *mdX_o, double *mdY_o, double *mdZ_o, double *md_dt_min, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){

        
        //calculate the crossing location where the particles intersect with one wall:
        mdX_o[ID] = mdX[ID] + mdvx[ID]*md_dt_min[ID] + 0.5 * mdAx[ID] * md_dt_min[ID] * md_dt_min[ID];
        mdY_o[ID] = mdY[ID] + mdvy[ID]*md_dt_min[ID] + 0.5 * mdAy[ID] * md_dt_min[ID] * md_dt_min[ID];
        mdZ_o[ID] = mdZ[ID] + mdvz[ID]*md_dt_min[ID] + 0.5 * mdAz[ID] * md_dt_min[ID] * md_dt_min[ID];

        

    }

}



__global__ void md_crossing_velocity(double *mdvx, double *mdvy, double *mdvz, double *mdAx, double *mdAy, double *mdAz, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *md_dt_min, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){

        //calculate v(t+dt1) at crossing point:
        mdvx[ID] = mdvx[ID] + mdAx[ID] * md_dt_min[ID];
        mdvy[ID] = mdvy[ID] + mdAy[ID] * md_dt_min[ID];
        mdvz[ID] = mdvz[ID] + mdAz[ID] * md_dt_min[ID];
    }
    
}



__global__ void md_velocityverlet1(double *mdX, double *mdY, double *mdZ, double *mdvx, double *mdvy, double *mdvz, double *mdAx, double *mdAy, double *mdAz, double dt_md, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){

        mdvx[ID] += 0.5 * dt_md * mdAx[ID];
        mdvy[ID] += 0.5 * dt_md * mdAy[ID];
        mdvz[ID] += 0.5 * dt_md * mdAz[ID];

        mdX[ID] = mdX[ID] + dt_md * mdvx[ID] ;
        mdY[ID] = mdY[ID] + dt_md * mdvy[ID] ;
        mdZ[ID] = mdZ[ID] + dt_md * mdvz[ID] ;
    }
}

__global__ void particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1(double *mdX, double *mdY, double *mdZ, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdVx, double *mdVy, double *mdVz, double *mdVx_o, double *mdVy_o, double *mdVz_o, double *mdAx, double *mdAy, double *mdAz, double *md_dt_min, double dt_md, double *L, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){
           if(mdX[ID]>L[0]/2 || mdX[ID]<-L[0]/2 || mdY[ID]>L[1]/2 || mdY[ID]<-L[1]/2 || mdZ[ID]>L[2]/2 || mdZ[ID]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdVx[ID] = -mdVx_o[ID];
            mdVy[ID] = -mdVy_o[ID];
            mdVz[ID] = -mdVz_o[ID];
            //let the particle stream during dt-dt1 with the reversed velocity:
            mdVx[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAx[ID];
            mdVy[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAy[ID];
            mdVz[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt_md - md_dt_min[ID]) * mdVx[ID] ;
            mdY[ID] = mdY[ID] + (dt_md - md_dt_min[ID]) * mdVy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt_md - md_dt_min[ID]) * mdVz[ID] ;
        }
    }

}



__host__ void noslip_md_velocityverletKernel1(double *mdX, double *mdY , double *mdZ , 
double *mdvx , double *mdvy , double *mdvz, double *mdAx , double *mdAy , double *mdAz,
double h_md, int Nmd, double *L, int grid_size, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o,
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

    md_wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

        //calculate particle's distance from walls if the particle is inside the box:
    md_distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, mdX_o, mdY_o, mdZ_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdAx, mdAy, mdAz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, mdAx, mdAy, mdAz, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, mdAx, mdAy, mdAz, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
   
}




__global__ void md_velocityverlet2(double *mdX, double *mdY, double *mdZ, double *mdvx, double *mdvy, double *mdvz, double *mdAx, double *mdAy, double *mdAz, double dt_md, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){

        mdvx[ID] += 0.5 * dt_md * mdAx[ID];
        mdvy[ID] += 0.5 * dt_md * mdAy[ID];
        mdvz[ID] += 0.5 * dt_md * mdAz[ID];
    }
}

__global__ void particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2(double *mdX, double *mdY, double *mdZ, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdVx, double *mdVy, double *mdVz, double *mdVx_o, double *mdVy_o, double *mdVz_o, double *mdAx, double *mdAy, double *mdAz, double *md_dt_min, double dt_md, double *L, int Nmd){

    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID<Nmd){
        if(mdX[ID]>L[0]/2 || mdX[ID]<-L[0]/2 || mdY[ID]>L[1]/2 || mdY[ID]<-L[1]/2 || mdZ[ID]>L[2]/2 || mdZ[ID]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];
            //make the velocity equal to the reverse of the velocity in crossing point.
            mdVx[ID] = -mdVx_o[ID];
            mdVy[ID] = -mdVy_o[ID];
            mdVz[ID] = -mdVz_o[ID];
            //let the particle stream during dt-dt1 with the reversed velocity:
            mdVx[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAx[ID];
            mdVy[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAy[ID];
            mdVz[ID] += 0.5 * (dt_md - md_dt_min[ID]) * mdAz[ID];
            
          
        }
    }

}



__host__ void noslip_md_velocityverletKernel2(double *mdX, double *mdY , double *mdZ , 
double *mdvx , double *mdvy , double *mdvz, double *mdAx , double *mdAy , double *mdAz,
double h_md, int Nmd, double *L, int grid_size, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *md_dt_min ,
double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o,
double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

    wall_sign<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz , wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

        //calculate particle's distance from walls if the particle is inside the box:
    distance_from_walls<<<grid_size,blockSize>>>(mdX , mdY, mdZ, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, mdX_o, mdY_o, mdZ_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdAx, mdAy, mdAz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    md_velocityverlet2<<<grid_size, blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, mdAx, mdAy, mdAz, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, mdAx, mdAy, mdAz, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

 }


__host__ void noslip_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdvx, double *d_mdvy, double *d_mdvz, double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *d_Fx, double *d_Fy, double *d_Fz, double h_md, int Nmd, int density, double *d_L , double ux, int grid_size , int delta, double real_time,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        
        noslip_md_velocityverletKernel1(d_mdX, d_mdY, d_mdZ, d_mdvx, d_mdvy, d_mdvz, d_mdAx, d_mdAy, d_mdAz , h_md, Nmd, d_L, grid_size, md_dt_x, md_dt_y, md_dt_z, md_dt_min, mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
      
        
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        noslip_calc_accelaration(d_mdX, d_mdY , d_mdZ , d_Fx , d_Fy , d_Fz , d_mdAx , d_mdAy , d_mdAz, d_L , Nmd ,m_md ,topology, ux ,real_time, grid_size);
        
        
        //velocityverletKernel2 is called to complete the velocity verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        noslip_md_velocityverletKernel2(d_mdX, d_mdY, d_mdZ, d_mdvx, d_mdvy, d_mdvz, d_mdAx, d_mdAy, d_mdAz , h_md, Nmd, d_L, grid_size, md_dt_x, md_dt_y, md_dt_z, md_dt_min, mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
       

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















/*
////////////////////////////////////////////////
//Active:

//first kernel: x+= hv(half time) + 0.5hha(new) ,v += 0.5ha(new)
__global__ void noslip_ActivevelocityverletKernel1(double *mdX, double *mdY , double *mdZ , 
double *mdvx , double *mdvy , double *mdvz,
double *mdAx , double *mdAy , double *mdAz,
 double h, int size, double *L, double *mdX_latest, double *mdY_latest, double *mdZ_latest, double *mdvx_latest, double *mdvy_latest, double *mdvz_latest, double *md_dt1, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdX_wallDist_P, double *mdY_wallDist_P, double *mdZ_wallDist_P, double *mdX_wallDist_N, double *mdY_wallDist_N, double *mdZ_wallDist_N)
{
    int ID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (ID < size)
    {


        //calculate particle's distance from walls:
        mdX_wallDist_P[ID] = L[0]/2-mdX[ID];
        mdX_wallDist_N[ID] = L[0]/2+mdX[ID];

        mdY_wallDist_P[ID] = L[1]/2-mdY[ID];
        mdY_wallDist_N[ID] = L[1]/2+mdY[ID];

        mdZ_wallDist_P[ID] = L[2]/2-mdZ[ID];
        mdZ_wallDist_N[ID] = L[2]/2+mdZ[ID];

        //restore the x[ID] because we need it after the first step of mpcd streaming:
        mdX_latest[ID] = x[ID];
        mdY_latest[ID] = y[ID];
        mdZ_latest[ID] = z[ID];

        //restore the v[ID] because we need it after the first step of mpcd streaming:
        mdvx_latest[ID] = mdvx[ID];
        mdvy_latest[ID] = mdvy[ID];
        mdvz_latest[ID] = mdvz[ID];

        double dt = h;

        // Particle velocities are updated by half a time step, and particle positions are updated based on the new velocities.
        mdvx[ID] += 0.5 * h * mdAx[ID];
        mdvy[ID] += 0.5 * h * mdAy[ID];
        mdvz[ID] += 0.5 * h * mdAz[ID];

        mdX[ID] = mdX[ID] + h * mdvx[ID] ;
        mdY[ID] = mdY[ID] + h * mdvy[ID] ;
        mdZ[ID] = mdZ[ID] + h * mdvz[ID] ;

        if (mdX[ID]>L[0]/2){

            //calculate md_dt1
          
            md_dt1[ID]= (-mdvx_latest[ID]+sqrt((mdvx_latest[ID]*mdvx_latest[ID])+(2*mdX_wallDist_P[ID]*mdAx[ID])))/mdAx[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx_latest[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];//or we could say L[0]/2
            mdY_o[ID] = mdY_latest[ID] + mdvy_latest[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];
            mdZ_o[ID] = mdZ_latest[ID] + mdvz_latest[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];//L[0]/2
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];

         

            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;

            

        }
        else if (mdX[ID]<-L[0]/2){

            //calculate md_dt1
          
            md_dt1[ID]= (-mdvx_latest[ID]+sqrt((mdvx_latest[ID]*mdvx_latest[ID])+(2*mdX_wallDist_N[ID]*mdAx[ID])))/mdAx[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx_latest[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];//or we could say -L[0]/2
            mdY_o[ID] = mdY_latest[ID] + mdvy_latest[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];
            mdZ_o[ID] = mdZ_latest[ID] + mdvz_latest[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];//-L[0]/2
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];


            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;

            

        }
        else if (mdY[ID]>L[1]/2){

            //calculate md_dt1
          
            md_dt1[ID]= (-mdvy_latest[ID]+sqrt((mdvy_latest[ID]*mdvy_latest[ID])+(2*mdY_wallDist_P[ID]*mdAy[ID])))/mdAy[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx_latest[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];
            mdY_o[ID] = mdY_latest[ID] + mdvy_latest[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];//or we could say L[1]/2
            mdZ_o[ID] = mdZ_latest[ID] + mdvz_latest[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];//L[1]/2
            mdZ[ID] = mdZ_o[ID];

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];

            
            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;

        
            
        }
        else if (mdY[ID]<-L[1]/2){

            //calculate md_dt1
          
            md_dt1[ID]= (-mdvy_latest[ID]+sqrt((mdvy_latest[ID]*mdvy_latest[ID])+(2*mdY_wallDist_N[ID]*mdAy[ID])))/mdAy[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx_latest[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];
            mdY_o[ID] = mdY_latest[ID] + mdvy_latest[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];//or we could say -L[1]/2
            mdZ_o[ID] = mdZ_latest[ID] + mdvz_latest[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];//-L[1]/2
            mdZ[ID] = mdZ_o[ID];

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];


            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;



        }
        else if (mdZ[ID]>L[2]/2){

            //calculate md_dt1
          
            md_dt1[ID]= (-mdvz_latest[ID]+sqrt((mdvz_latest[ID]*mdvz_latest[ID])+(2*mdZ_wallDist_P[ID]*mdAz[ID])))/mdAz[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx_latest[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];
            mdY_o[ID] = mdY_latest[ID] + mdvy_latest[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];
            mdZ_o[ID] = mdZ_latest[ID] + mdvz_latest[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];//or we could say L[2]/2


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];//L[2]/2

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];


            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;

           
        }
        else if (mdZ[ID]<-L[2]/2){

             //calculate md_dt1
          
            md_dt1[ID]= (-mdvz_latest[ID]+sqrt((mdvz_latest[ID]*mdvz_latest[ID])+(2*mdZ_wallDist_N[ID]*mdAz[ID])))/mdAz[ID];

            //calculate v(t+dt1) at crossing point:
            mdvx[ID] = mdvx_latest[ID] + mdAx[ID] * md_dt1[ID];
            mdvy[ID] = mdvy_latest[ID] + mdAy[ID] * md_dt1[ID];
            mdvz[ID] = mdvz_latest[ID] + mdAz[ID] * md_dt1[ID];

            //calculate the crossing location where the particles intersect with one wall:
            mdX_o[ID] = mdX_latest[ID] + mdvx[ID]*md_dt1[ID] + 0.5 * mdAx[ID] * md_dt1[ID] * md_dt1[ID];
            mdY_o[ID] = mdY_latest[ID] + mdvy[ID]*md_dt1[ID] + 0.5 * mdAy[ID] * md_dt1[ID] * md_dt1[ID];
            mdZ_o[ID] = mdZ_latest[ID] + mdvz[ID]*md_dt1[ID] + 0.5 * mdAz[ID] * md_dt1[ID] * md_dt1[ID];//or we could say -L[2]/2


            //make the position of particle equal to (xo, yo, zo):
            mdX[ID] = mdX_o[ID];
            mdY[ID] = mdY_o[ID];
            mdZ[ID] = mdZ_o[ID];//-L[2]/2

            //reverse the velocity direction:
            mdvx[ID] = - mdvx[ID];
            mdvy[ID] = - mdvy[ID];
            mdvz[ID] = - mdvz[ID];


            //let the particle stream during dt-dt1 with the reversed velocity:
            mdvx[ID] += 0.5 * (dt - md_dt1[ID]) * mdAx[ID];
            mdvy[ID] += 0.5 * (dt - md_dt1[ID]) * mdAy[ID];
            mdvz[ID] += 0.5 * (dt - md_dt1[ID]) * mdAz[ID];

            mdX[ID] = mdX[ID] + (dt - md_dt1[ID]) * mdvx[ID] ;
            mdY[ID] = mdY[ID] + (dt - md_dt1[ID]) * mdvy[ID] ;
            mdZ[ID] = mdZ[ID] + (dt - md_dt1[ID]) * mdvz[ID] ;


        }



    }
}

//second Kernel of velocity verelt: v += 0.5ha(old)
__global__ void noslip_ActivevelocityverletKernel2(double *mdvx , double *mdvy , double *mdvz,
double *mdAx , double *mdAy , double *mdAz, double h, int size)
{
    int ID =  blockIdx.x * blockDim.x + threadIdx.x ;
    if (ID < size)
    {
        mdvx[ID] += 0.5 * h * mdAx[ID];
        mdvy[ID] += 0.5 * h * mdAy[ID];
        mdvz[ID] += 0.5 * h * mdAz[ID];
    }
}












__host__ void noslip_Active_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdvx, double *d_mdvy, double *d_mdvz,
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
    double h_md ,int Nmd, int density, double *d_L , double ux, int grid_size, 
    int delta, double real_time, int m, int N, double mass, double mass_fluid,
    double *gama_T, int *random_array, unsigned int seed, int topology, double *Xcm, double *Ycm, double *Zcm, int *flag_array, double u_scale,
    double *mdX_latest, double *mdY_latest, double *mdZ_latest, double *md_dt1, double *mdX_o, double *mdY_o, double *mdZ_o, 
    double *mdX_wallDist_P, double *mdY_wallDist_P, double *mdZ_wallDist_P, double *mdX_wallDist_N, double *mdY_wallDist_N, double *mdZ_wallDist_N)
{
    for (int tt = 0 ; tt < delta ; tt++)
    {


        gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        noslip_ActivevelocityverletKernel1<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, d_mdvx, d_mdvy, d_mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot , h_md,Nmd, d_L, mdX_latest, mdY_latest, mdZ_latest, mdvx_latest, mdvy_latest, mdvz_latest, md_dt1, mdX_o, mdY_o, mdZ_o, mdX_wallDist_P, mdY_wallDist_P, mdZ_wallDist_P, mdX_wallDist_N, mdY_wallDist_N, mdZ_wallDist_N);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        
        
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
     

        
        
        //velocityverletKernel2 is called to complete the velocity verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.

        //***
        noslip_ActivevelocityverletKernel2<<<grid_size,blockSize>>>(d_mdvx, d_mdvy, d_mdvz, d_mdAx, d_mdAy, d_mdAz, h_md, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        //The real_time is incremented by the time step h_md, effectively moving the simulation time forward.
        real_time += h_md;


        
    }
}

*/