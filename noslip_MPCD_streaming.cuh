#include "gpu_md.h"

//a function to consider velocity sign of particles and determine which sides of the box it should interact with 
__global__ void wall_sign(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, int N){

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

        //(isnan(vx[tid])|| isnan(vy[tid]) || isnan(vz[tid])) ? printf("00vx[%i]=%f, vy[%i]=%f, vz[%i]=%f \n", tid, vx[tid], tid, vy[tid], tid, vz[tid])
                                                            //: printf("");


    }

}
//a function to calculate distance of particles which are inside the box from the corresponding walls:
__global__ void distance_from_walls(double *x, double *y, double *z, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *L, int N){

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

    }    


}


//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void mpcd_deltaT(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x, double *dt_y, double *dt_z, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if(wall_sign_x[tid] == 0 ) dt_x[tid] == 10000;//a big number because next step is to consider the minimum of dt .

        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1)  dt_x[tid] = abs(x_wall_dist[tid]/vx[tid]);

        if(wall_sign_y[tid] == 0 ) dt_y[tid] == 10000;
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1)  dt_y[tid] = abs(y_wall_dist[tid]/vy[tid]);

        if(wall_sign_z[tid] == 0 ) dt_z[tid] == 10000;
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1)  dt_z[tid] = abs(z_wall_dist[tid]/vz[tid]);
    }


}
//a function to calculate minimum of 3 items  (dt_x, dt_y and dt_z) :
__global__ void deltaT_min(double *dt_x, double *dt_y, double *dt_z, double *dt_min, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        dt_min[tid] = min(min(dt_x[tid], dt_y[tid]) , dt_z[tid]);
        //printf("dt_min[%i] = %f", tid, dt_min[tid]);

    }

}
//calculate the crossing location where the particles intersect with one wall:
__global__ void mpcd_crossing_location(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *x_o, double *y_o, double *z_o, double *dt_min, double dt, double *L, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        //if( ((x[tid] + dt * vx[tid]) >L[0]/2 || (x[tid] + dt * vx[tid])<-L[0]/2 || (y[tid] + dt * vy[tid])>L[1]/2 || (y[tid] + dt * vy[tid])<-L[1]/2 || (z[tid]+dt * vz[tid])>L[2]/2 || (z[tid] + dt * vz[tid])<-L[2]/2) && dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, dt_min[tid]);
        x_o[tid] = x[tid] + vx[tid]*dt_min[tid];
        y_o[tid] = y[tid] + vy[tid]*dt_min[tid];
        z_o[tid] = z[tid] + vz[tid]*dt_min[tid];
    }

}

//calculate the crossing location where the particles intersect with one wall:
__global__ void mpcd_opposite_crossing_location(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *x_o_opp, double *y_o_opp, double *z_o_opp, double *dt_min_opp, double dt, double *L, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        //if( ((x[tid] + dt * vx[tid]) >L[0]/2 || (x[tid] + dt * vx[tid])<-L[0]/2 || (y[tid] + dt * vy[tid])>L[1]/2 || (y[tid] + dt * vy[tid])<-L[1]/2 || (z[tid]+dt * vz[tid])>L[2]/2 || (z[tid] + dt * vz[tid])<-L[2]/2) && dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, dt_min[tid]);
        x_o_opp[tid] = x[tid] + (-vx[tid])*dt_min_opp[tid];
        y_o_opp[tid] = y[tid] + (-vy[tid])*dt_min_opp[tid];
        z_o_opp[tid] = z[tid] + (-vz[tid])*dt_min_opp[tid];
    }

}





__global__ void mpcd_crossing_velocity(double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        //calculate v(t+dt1) : in this case that we don't have acceleration it is equal to v(t).
        //then we put the velocity equal to v(t+dt1):
        //this part in this case is not necessary but we do it for generalization.
        vx_o[tid] = vx[tid];
        vy_o[tid] = vy[tid];
        vz_o[tid] = vz[tid];
    }
    
}

__global__ void mpcd_opposite_crossing_velocity(double *vx, double *vy, double *vz, double *vx_o_opp, double *vy_o_opp, double *vz_o_opp, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        //calculate v(t+dt1) : in this case that we don't have acceleration it is equal to v(t).
        //then we put the velocity equal to v(t+dt1):
        //this part in this case is not necessary but we do it for generalization.
        vx_o_opp[tid] = -vx[tid];
        vy_o_opp[tid] = -vy[tid];
        vz_o_opp[tid] = -vz[tid];
    }
    
}


__global__ void mpcd_velocityverlet(double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt, int N, double *L, double *T){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2) printf("********** x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, x[tid], tid, y[tid], tid, z[tid]);
        x[tid] += dt * vx[tid];
        y[tid] += dt * vy[tid];
        z[tid] += dt * vz[tid];
        T[tid]+=dt;
        /*if(tid == 0) {
            printf("T[0] = %f", T[0]);
        }*/
    }
}
__global__ void particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            x[tid] = x_o[tid];
            y[tid] = y_o[tid];
            z[tid] = z_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            vx[tid] = -vx_o[tid];
            vy[tid] = -vy_o[tid];
            vz[tid] = -vz_o[tid];
            //let the particle move during dt-dt1 with the reversed velocity:
            x[tid] += (dt - (dt_min[tid])) * vx[tid];
            y[tid] += (dt - (dt_min[tid])) * vy[tid];
            z[tid] += (dt - (dt_min[tid])) * vz[tid];
        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
    }

}



__host__ void noslip_MPCD_streaming(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double h_mpcd, int N, int grid_size, double *L, double *dt_x, double *dt_y, double *dt_z, double *dt_min, double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *T)
{

    wall_sign<<<grid_size,blockSize>>>(d_vx , d_vy , d_vz, wall_sign_x, wall_sign_y, wall_sign_z, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

        //calculate particle's distance from walls if the particle is inside the box:
    distance_from_walls<<<grid_size,blockSize>>>(d_x , d_y , d_z, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    deltaT_min<<<grid_size,blockSize>>>(dt_x, dt_y, dt_z, dt_min, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    mpcd_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o, y_o, z_o, dt_min, h_mpcd, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    mpcd_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}



