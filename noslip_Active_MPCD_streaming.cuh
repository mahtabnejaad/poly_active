








//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_mpcd_deltaT(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x, double *dt_y, double *dt_z, int N, double fa_x, double fa_y, double fa_z, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        
        double mm = (Nmd*mass+mass_fluid*N);

        if(wall_sign_x[tid] == 0 ) dt_x[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
            if(fa_x/mm == 0.0)   dt_x[tid] = abs(x_wall_dist[tid]/vx[tid]);

            else if (fa_x/mm != 0.0)  dt_x[tid] = ((-vx[tid]+sqrt(abs((vx[tid]*vx[tid])+(2*x_wall_dist[tid]*(fa_x/mm)))))/(fa_x/mm));

        }  

        if(wall_sign_y[tid] == 0 ) dt_y[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(fa_y/mm  == 0.0)   dt_y[tid] = abs(y_wall_dist[tid]/vy[tid]);

            else if (fa_y/mm != 0.0)  dt_y[tid] = ((-vy[tid]+sqrt(abs((vy[tid]*vy[tid])+(2*y_wall_dist[tid]*(fa_y/mm )))))/(fa_y/mm ));

        }  

        if(wall_sign_z[tid] == 0 ) dt_z[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(fa_z/mm == 0.0)   dt_z[tid] = abs(z_wall_dist[tid]/vz[tid]);

            else if (fa_z/mm != 0.0)  dt_z[tid] = ((-vz[tid]+sqrt(abs((vz[tid]*vz[tid])+(2*z_wall_dist[tid]*(fa_z/mm)))))/(fa_z/mm));

        }  



    }


}
//a function to calculate minimum of 3 items  (dt_x, dt_y and dt_z) :
__global__ void Active_deltaT_min(double *dt_x, double *dt_y, double *dt_z, double *dt_min, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        dt_min[tid] = min(min(dt_x[tid], dt_y[tid]) , dt_z[tid]);
        //printf("dt_min[%i] = %f", tid, dt_min[tid]);

    }

}
//calculate the crossing location where the particles intersect with one wall:
__global__ void Active_mpcd_crossing_location(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *x_o, double *y_o, double *z_o, double *dt_min, double dt, double *L, int N, double fa_x, double fa_y, double fa_z, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if( ((x[tid] + dt * vx[tid]) >L[0]/2 || (x[tid] + dt * vx[tid])<-L[0]/2 || (y[tid] + dt * vy[tid])>L[1]/2 || (y[tid] + dt * vy[tid])<-L[1]/2 || (z[tid]+dt * vz[tid])>L[2]/2 || (z[tid] + dt * vz[tid])<-L[2]/2) && dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, dt_min[tid]);
        x_o[tid] = x[tid] + vx[tid]*dt_min[tid];
        y_o[tid] = y[tid] + vy[tid]*dt_min[tid];
        z_o[tid] = z[tid] + vz[tid]*dt_min[tid];
    }

}



__global__ void Active_mpcd_crossing_velocity(double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, int N, double fa_x, double fa_y, double fa_z, int Nmd, double mass, double mass_fluid){

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





__global__ void Active_mpcd_velocityverlet(double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt, int N, double *L, double *T, double fa_x, double fa_y, double fa_z, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        double QQ=-(dt*dt)/(2*(Nmd*mass+mass_fluid*N));
        double Q=-dt/(Nmd*mass+mass_fluid*N);

        if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2) printf("********** x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, x[tid], tid, y[tid], tid, z[tid]);
        x[tid] += dt * vx[tid]+QQ * fa_x;
        y[tid] += dt * vy[tid]+QQ * fa_y;
        z[tid] += dt * vz[tid]+QQ * fa_z;
        vx[tid]=vx[tid]+Q * fa_x;
        vy[tid]=vy[tid]+Q * fa_y;
        vz[tid]=vz[tid]+Q * fa_z;

        T[tid]+=dt;
        /*if(tid == 0) {
            printf("T[0] = %f", T[0]);
        }*/
    }
}
__global__ void Active_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N, double fa_x, double fa_y, double fa_z, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        double QQ2=-((dt - (dt_min[tid]))*(dt - (dt_min[tid])))/(2*(Nmd*mass+mass_fluid*N));
        double Q2=-(dt - (dt_min[tid]))/(Nmd*mass+mass_fluid*N);

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
            x[tid] += (dt - (dt_min[tid])) * vx[tid] + QQ2 * fa_x;
            y[tid] += (dt - (dt_min[tid])) * vy[tid] + QQ2 * fa_y;
            z[tid] += (dt - (dt_min[tid])) * vz[tid] + QQ2 * fa_z;
            vx[tid]=vx[tid]+Q2 * fa_x;
            vy[tid]=vy[tid]+Q2 * fa_y;
            vz[tid]=vz[tid]+Q2 * fa_z;

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
    }

}


__host__ void noslip_Active_MPCD_streaming(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double h_mpcd, int N, int grid_size, double *fa_x, double *fa_y, double *fa_z, 
double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
int Nmd , double ux, int mass, int mass_fluid, double real_time, int m, int topology, int shared_mem_size, double *L, double *dt_x, double *dt_y, double *dt_z, double *dt_min, 
double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *T)
{

    wall_sign<<<grid_size,blockSize>>>(d_vx , d_vy , d_vz, wall_sign_x, wall_sign_y, wall_sign_z, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

        //calculate particle's distance from walls if the particle is inside the box:
    distance_from_walls<<<grid_size,blockSize>>>(d_x , d_y , d_z, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(dt_x, dt_y, dt_z, dt_min, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_mpcd_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o, y_o, z_o, dt_min, h_mpcd, L, N, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_mpcd_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    Active_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

