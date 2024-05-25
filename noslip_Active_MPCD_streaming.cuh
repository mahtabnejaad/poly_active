


















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
            z[tid] += (dt - (dt_min[tid])) * vz[tid] + QQ2s * fa_z;
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
    

    Active_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, *fa_x, *fa_y, *fa_z, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

