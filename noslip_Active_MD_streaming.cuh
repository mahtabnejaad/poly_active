



__host__ void Active_noslip_md_velocityverletKernel1(double *mdX, double *mdY , double *mdZ , 
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

    Active_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, mdX_o, mdY_o, mdZ_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdAx, mdAy, mdAz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    Active_md_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, mdAx, mdAy, mdAz, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, mdAx, mdAy, mdAz, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
   
}


__host__ void Active_noslip_md_velocityverletKernel2(double *mdX, double *mdY , double *mdZ , 
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

    Active_md_deltaT<<<grid_size,blockSize>>>(mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ , mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, md_dt_x, md_dt_y, md_dt_z, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_deltaT_min<<<grid_size,blockSize>>>(md_dt_x, md_dt_y, md_dt_z, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_location<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy , mdvz, mdAx, mdAy, mdAz, mdX_o, mdY_o, mdZ_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_crossing_velocity<<<grid_size,blockSize>>>(mdvx, mdvy, mdvz, mdAx, mdAy, mdAz, mdvx_o, mdvy_o, mdvz_o, md_dt_min, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_md_velocityverlet2<<<grid_size, blockSize>>>(mdX , mdY, mdZ, mdvx , mdvy, mdvz, mdAx, mdAy, mdAz, h_md, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet2<<<grid_size,blockSize>>>(mdX , mdY, mdZ, mdX_o, mdY_o, mdZ_o, mdvx, mdvy, mdvz, mdvx_o, mdvy_o, mdvz_o, mdAx, mdAy, mdAz, md_dt_min, h_md, L, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

 }



__host__ void Active_noslip_calc_acceleration( double *x ,double *y , double *z , 
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

    monomer_active_backward_forces(x, y ,z ,
    Ax , Ay, Az, fa_kx, fa_ky, fa_kz, fb_kx, fb_ky, fb_kz, Aa_kx, Aa_ky, Aa_kz, Ab_kx, Ab_ky, Ab_kz, ex, ey, ez, ux, mass, gama_T, 
    L, size , mass_fluid, real_time, m, topology, grid_size, N , random_array, seed , Ax_tot, Ay_tot, Az_tot, fa_x, fa_y, fa_z, fb_x, fb_y, fb_z, block_sum_ex, block_sum_ey, block_sum_ez, flag_array, u_scale);
    

 }














__host__ void Active_noslip_MD_streaming(double *d_mdX, double *d_mdY, double *d_mdZ,
    double *d_mdvx, double *d_mdvy, double *d_mdvz, double *d_mdAx, double *d_mdAy, double *d_mdAz,
    double *d_Fx, double *d_Fy, double *d_Fz, double h_md, int Nmd, int density, double *d_L , double ux, int grid_size , int delta, double real_time,
    double *md_dt_min, double *md_dt_x, double *md_dt_y, double *md_dt_z, double *mdX_o, double *mdY_o, double *mdZ_o, double *mdvx_o, double *mdvy_o, double *mdvz_o, double *mdX_wall_dist, double *mdY_wall_dist, double *mdZ_wall_dist, double *wall_sign_mdX, double *wall_sign_mdY, double *wall_sign_mdZ){

        for (int tt = 0 ; tt < delta ; tt++)
    {

        noslip_gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd); //should I consider virtual particles here?
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );


        Active_noslip_md_velocityverletKernel1(d_mdX, d_mdY, d_mdZ, d_mdvx, d_mdvy, d_mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd, d_L, grid_size, md_dt_x, md_dt_y, md_dt_z, md_dt_min, mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
      
        
        
        
        //The function calc_accelaration is called to compute the new accelerations for each particle based on their positions and interactions.
        //These accelerations are used in the subsequent time step to update particle velocities.
        Active_noslip_calc_acceleration(d_mdX, d_mdY , d_mdZ , d_Fx , d_Fy , d_Fz , d_mdAx , d_mdAy , d_mdAz, d_L , Nmd ,m_md ,topology, ux ,real_time, grid_size);
        


        sum_kernel<<<grid_size,blockSize>>>(d_Fx ,d_Fy,d_Fz, d_mdAx ,d_mdAy, d_mdAz, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

       

        
        //velocityverletKernel2 is called to complete the velocity verlet algorithm by updating particle velocities using the second half of the time step. 
        //This ensures that the velocities are synchronized with the newly calculated accelerations.
        Active_noslip_md_velocityverletKernel2(d_mdX, d_mdY, d_mdZ, d_mdvx, d_mdvy, d_mdvz, d_Ax_tot, d_Ay_tot, d_Az_tot, h_md, Nmd, d_L, grid_size, md_dt_x, md_dt_y, md_dt_z, md_dt_min, mdX_o, mdY_o, mdZ_o, mdvx_o, mdvy_o, mdvz_o, mdX_wall_dist, mdY_wall_dist, mdZ_wall_dist, wall_sign_mdX, wall_sign_mdY, wall_sign_mdZ);
       


        noslip_backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Nmd);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

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

