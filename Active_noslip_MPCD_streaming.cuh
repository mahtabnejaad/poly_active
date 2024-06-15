
//a function to consider velocity sign of particles and determine which sides of the box it should interact with 
__global__ void CM_wall_sign(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, int N, double *Vxcm, double *Vycm, double *Vzcm){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if (vx[tid] > -*Vxcm )  wall_sign_x[tid] = 1;
        else if (vx[tid] < -*Vxcm)  wall_sign_x[tid] = -1;
        else if(vx[tid] == -*Vxcm)  wall_sign_x[tid] = 0;
        
        if (vy[tid] > -*Vycm ) wall_sign_y[tid] = 1;
        else if (vy[tid] < -*Vycm) wall_sign_y[tid] = -1;
        else if (vy[tid] == -*Vycm )  wall_sign_y[tid] = 0;

        if (vz[tid] > -*Vzcm) wall_sign_z[tid] = 1;
        else if (vz[tid] < -*Vzcm) wall_sign_z[tid] = -1;
        else if (vz[tid] == -*Vzcm)  wall_sign_z[tid] = 0;

        //(isnan(vx[tid])|| isnan(vy[tid]) || isnan(vz[tid])) ? printf("00vx[%i]=%f, vy[%i]=%f, vz[%i]=%f \n", tid, vx[tid], tid, vy[tid], tid, vz[tid])
                                                            //: printf("");


    }
}

//a function to calculate distance of particles which are inside the box from the corresponding walls:
__global__ void CM_distance_from_walls(double *x, double *y, double *z, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *L, int N, double *Xcm, double *Ycm, double *Zcm){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if (wall_sign_x[tid] == 1)   x_wall_dist[tid] = L[0]/2-((x[tid]) + *Xcm);
        else if (wall_sign_x[tid] == -1)  x_wall_dist[tid] = L[0]/2+((x[tid]) + *Xcm);
        else if(wall_sign_x[tid] == 0)  x_wall_dist[tid] = L[0]/2 -((x[tid]) + *Xcm);//we can change it as we like . it doesn't matter.


        if (wall_sign_y[tid] == 1)   y_wall_dist[tid] = L[1]/2-((y[tid]) + *Ycm);
        else if (wall_sign_y[tid] == -1)  y_wall_dist[tid] = L[1]/2+((y[tid]) + *Ycm);
        else if(wall_sign_y[tid] == 0)  y_wall_dist[tid] = L[1]/2 -((y[tid]) + *Ycm);//we can change it as we like . it doesn't matter.


        if (wall_sign_z[tid] == 1)   z_wall_dist[tid] = L[2]/2-((z[tid]) + *Zcm);
        else if (wall_sign_z[tid] == -1)  z_wall_dist[tid] = L[2]/2+((z[tid]) + *Zcm);
        else if(wall_sign_z[tid] == 0)  z_wall_dist[tid] = L[2]/2 -((z[tid]) + *Zcm);//we can change it as we like . it doesn't matter.



        //printf("***dist_x[%i]=%f, dist_y[%i]=%f, dist_z[%i]=%f\n", tid, x_wall_dist[tid], tid, y_wall_dist[tid], tid, z_wall_dist[tid]);
        int idxx;
        idxx = (int(x[tid] + L[0] / 2 + 2) + (L[0] + 4) * int(y[tid] + L[1] / 2 + 2) + (L[0] + 4) * (L[1] + 4) * int(z[tid] + L[2] / 2 + 2));
        //printf("index[%i]=%i, x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, idxx, tid, x[tid], tid, y[tid], tid, z[tid]);//checking

    }    


}






//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_noslip_mpcd_deltaT(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x, double *dt_y, double *dt_z, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, int mass, int mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("---fa_x=%f, fa_y=%f, fa_z=%f\n", *fa_x, *fa_y, *fa_z);
    if (tid<N){
        
        double mm = (Nmd*mass+mass_fluid*N);

        if(wall_sign_x[tid] == 0 ) dt_x[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
            if(*fa_x/mm == 0.0)   dt_x[tid] = abs(x_wall_dist[tid]/vx[tid]);

            else if (*fa_x/mm != 0.0)  dt_x[tid] = ((-vx[tid]+sqrt(abs((vx[tid]*vx[tid])+(2*x_wall_dist[tid]*(*fa_x/mm)))))/(*fa_x/mm));

        }  

        if(wall_sign_y[tid] == 0 ) dt_y[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(*fa_y/mm  == 0.0)   dt_y[tid] = abs(y_wall_dist[tid]/vy[tid]);

            else if (*fa_y/mm != 0.0)  dt_y[tid] = ((-vy[tid]+sqrt(abs((vy[tid]*vy[tid])+(2*y_wall_dist[tid]*(*fa_y/mm )))))/(*fa_y/mm ));

        }  

        if(wall_sign_z[tid] == 0 ) dt_z[tid] == 10000;//a big number because next step is to consider the minimum of dt .
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(*fa_z/mm == 0.0)   dt_z[tid] = abs(z_wall_dist[tid]/vz[tid]);

            else if (*fa_z/mm != 0.0)  dt_z[tid] = ((-vz[tid]+sqrt(abs((vz[tid]*vz[tid])+(2*z_wall_dist[tid]*(*fa_z/mm)))))/(*fa_z/mm));

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
__global__ void Active_mpcd_crossing_location(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *x_o, double *y_o, double *z_o, double *dt_min, double dt, double *L, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, int mass, int mass_fluid){

    double mm = (Nmd*mass+mass_fluid*N);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        //if( ((x[tid] + dt * vx[tid]) >L[0]/2 || (x[tid] + dt * vx[tid])<-L[0]/2 || (y[tid] + dt * vy[tid])>L[1]/2 || (y[tid] + dt * vy[tid])<-L[1]/2 || (z[tid]+dt * vz[tid])>L[2]/2 || (z[tid] + dt * vz[tid])<-L[2]/2) && dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, dt_min[tid]);
        x_o[tid] = x[tid] + vx[tid]*dt_min[tid] + 0.5 * *fa_x * dt_min[tid] * dt_min[tid] / mm;
        y_o[tid] = y[tid] + vy[tid]*dt_min[tid] + 0.5 * *fa_y * dt_min[tid] * dt_min[tid] / mm;
        z_o[tid] = z[tid] + vz[tid]*dt_min[tid] + 0.5 * *fa_z * dt_min[tid] * dt_min[tid] / mm;
    }

}



__global__ void Active_mpcd_crossing_velocity(double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, int mass, int mass_fluid){

    double mm = (Nmd*mass+mass_fluid*N);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        //calculate v(t+dt1) : in this case that we don't have acceleration it is equal to v(t).
        //then we put the velocity equal to v(t+dt1):
        //this part in this case is not necessary but we do it for generalization.
        vx_o[tid] = vx[tid] + *fa_x * dt_min[tid] / mm ;
        vy_o[tid] = vy[tid] + *fa_y * dt_min[tid] / mm;
        vz_o[tid] = vz[tid] + *fa_z * dt_min[tid] / mm;
    }
    
}





__global__ void Active_mpcd_velocityverlet(double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt, int N, double *L, double *T, double *fa_x, double *fa_y, double *fa_z, int Nmd, int mass, int mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        double QQ=-(dt*dt)/(2*(Nmd*mass+mass_fluid*N));
        double Q=-dt/(Nmd*mass+mass_fluid*N);

        //if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2) printf("********** x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, x[tid], tid, y[tid], tid, z[tid]);
        x[tid] += dt * vx[tid]+QQ * *fa_x;
        y[tid] += dt * vy[tid]+QQ * *fa_y;
        z[tid] += dt * vz[tid]+QQ * *fa_z;
        vx[tid]=vx[tid]+Q * *fa_x;
        vy[tid]=vy[tid]+Q * *fa_y;
        vz[tid]=vz[tid]+Q * *fa_z;

        T[tid]+=dt;
        /*if(tid == 0) {
            printf("T[0] = %f", T[0]);
        }*/
    }
}
__global__ void Active_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, int mass, int mass_fluid){

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
            x[tid] += (dt - (dt_min[tid])) * vx[tid] + QQ2 * *fa_x;
            y[tid] += (dt - (dt_min[tid])) * vy[tid] + QQ2 * *fa_y;
            z[tid] += (dt - (dt_min[tid])) * vz[tid] + QQ2 * *fa_z;
            vx[tid]=vx[tid]+Q2 * *fa_x;
            vy[tid]=vy[tid]+Q2 * *fa_y;
            vz[tid]=vz[tid]+Q2 * *fa_z;

        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
    }

}


__host__ void Active_noslip_MPCD_streaming(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double* d_mdX, double* d_mdY, double* d_mdZ, double* d_mdVx , double* d_mdVy, double* d_mdVz,
double *X_tot, double *Y_tot, double *Z_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot,
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, double h_mpcd, int N, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L, int Nmd , double ux, int mass, int mass_fluid, double real_time, int m, int topology, double *dt_x, double *dt_y, double *dt_z, double *dt_min, 
double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *T, int *n_outbox_mpcd, int *n_outbox_md, int *dn_mpcd_tot, int *dn_md_tot, int *CMsumblock_n_outbox_mpcd, int *CMsumblock_n_outbox_md)

{

    double *fax, *fay, *faz;
    cudaMalloc((void**)&fax, sizeof(double)); cudaMalloc((void**)&fay, sizeof(double)); cudaMalloc((void**)&faz, sizeof(double));
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice); 
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fay, fa_y, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(faz, fa_z, sizeof(double) , cudaMemcpyHostToDevice);


    CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    //take all MPCD particles to CM reference frame:
    gotoCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    CM_wall_sign<<<grid_size,blockSize>>>(d_vx , d_vy , d_vz, wall_sign_x, wall_sign_y, wall_sign_z, N, Vxcm, Vycm, Vzcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    CM_distance_from_walls<<<grid_size,blockSize>>>(d_x , d_y , d_z, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, L, N, Xcm, Ycm, Zcm);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_noslip_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(dt_x, dt_y, dt_z, dt_min, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_mpcd_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o, y_o, z_o, dt_min, h_mpcd, L, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_mpcd_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    Active_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_outside_particles
    outerParticles_CM_system(d_mdX, d_mdY, d_mdZ, d_x, d_y, d_z,  d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
    mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_mpcd_tot, dn_md_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, 
    Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz,
    CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_n_outbox_mpcd, CMsumblock_n_outbox_md, topology, L);
    
    //gotoOUTBOXCMframe  go to out of box cm frame for mpcd particles:
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go to out of box cm frame for md particles:(should I???)
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );



    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go back to the old CM frame:
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    //gotoLabFrame for mpcd particles:
    backtoLabframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_system: now the CM has changed.
    CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);



}

