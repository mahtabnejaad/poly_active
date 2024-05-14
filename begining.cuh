__global__ void kernelInit(double *x, double*y, double *z,double *vx, double *vy , double *vz, double *L,double px , double py, double pz, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid<N)
    {
        x[tid] *= L[0];
        y[tid] *= L[1];
        z[tid] *= L[2];
        x[tid] -= L[0]/2;
        y[tid] -= L[1]/2;
        z[tid] -= L[2]/2;
        vx[tid] -= px;
        vy[tid] -= py;
        vz[tid] -= pz;
    }

}
__host__ void mpcd_init(curandGenerator_t gen,
double *d_x,
double *d_y,
double *d_z,
double *d_vx,
double *d_vy,
double *d_vz,
int grid_size,
int N)
{

        //initialisation postions:
        curandGenerateUniformDouble(gen, d_x, N);//This line generates random double-precision values between 0 and 1 for the d_x array. It uses the gen generator to populate the d_x array with random positions.
        curandGenerateUniformDouble(gen, d_y, N);// Similarly, this line generates random double-precision values for the d_y array.
        curandGenerateUniformDouble(gen, d_z, N);// This line generates random double-precision values for the d_z array.
        
        //initialisation velocity:
        curandGenerateNormalDouble(gen, d_vx, N, 0, 1);//This line generates random double-precision values for the d_vx array from a normal distribution with mean 0 and standard deviation 1.
        curandGenerateNormalDouble(gen, d_vy, N, 0, 1);//Similarly, this line generates random double-precision values for the d_vy array.
        curandGenerateNormalDouble(gen, d_vz, N, 0, 1);//This line generates random double-precision values for the d_vz array.

}

__global__ void makezero(double *T, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)  T[tid] = 0.0;

}


__host__ void start_simulation(std::string file_name, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_x , double *d_y , double *d_z ,
double *d_vx , double *d_vy , double *d_vz,
curandGenerator_t gen, int grid_size, double *T)
{
    std::string log_name = file_name + "_log.log";
    std::string trj_name = file_name + "_traj.xyz";
    std::ofstream log (log_name);
    std::ofstream trj (trj_name);
    printf( "***WELCOME TO MPCD_MD_LeesEdwards CUDA CODE!***\nBy: Reyhaneh Afghahi Farimani, rhn_a_farimani@yahoo.com.\nThis code comes with a python code to analyse the results.");
    printf("\n ***The Active Polymer MPCD_MD simulation is done***\nBy: Mahtab Taghavinejad, mahtabnejadt@gmail.com.");
    printf("\ninput system:\nensemble:NVT, thermostat= cell_level_Maxwell_Boltzaman_thermostat, Lx=%i,Ly=%i,Lz=%i,shear_rate=%f,density=%i\n", int(L[0]), int(L[1]),int(L[2]), shear_rate, density);
    printf( "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition: shear direction:x , gradiant direction:z , vorticity direction: y\n");
    if (topology==1)
        printf("A poly[%i]catenane with %i monomer in each ring is embeded in the MPCD fluid.\n" , n_md , m_md);
    if (topology==2)
    printf("A linked[%i]ring with %i monomer in each ring is embeded in the MPCD fluid.\nWarning: the code currently support only linked[2]rings.\n" , n_md , m_md);
    printf("simulation time = %i, measurments accur every %i step.\n", simuationtime, swapsize);
    

    log<<"***WELCOME TO MPCD_MD_LeesEdwards CUDA CODE!***\nBy: Reyhaneh Afghahi Farimani, rhn_a_farimani@yahoo.com.***The Active Polymer MPCD_MD simulation is done***\nBy: Mahtab Taghavinejad, mahtabnejadt@gmail.com. \nThis code comes with a python code to analyse the results.";
    log<< "\ninput system:\nensemble:NVT, thermostat= cell_level_Maxwell_Boltzaman_thermostat, Lx="<<int(L[0])<<",Ly="<<int(L[1])<<",Lz="<<int(L[2])<<",shear_rate="<<shear_rate<<",density="<<density<<std::endl;
    if (topology==1)
        log<<"A poly["<<n_md<<"]catenane with "<<m_md<<" monomer in each ring is embeded in the MPCD fluid.\n";
    if (topology==2)
        log<<"A linked["<<n_md<<"]ring with "<<m_md<<" monomer in each ring is embeded in the MPCD fluid.\n";
    log<< "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition: shear direction:x , gradiant direction:z , vorticity direction: y\n";
    log<<"simulation time ="<<simulationtime<<", measurments accur every "<<swapsize<<" step.\n" ;
   

    double ux =shear_rate * L[2];
    //int Nc = L[0]*L[1]*L[2];
    //int Nc = L[0] + L[0]*L[1] + L[0]*L[1]*L[2];
    //int Nc = (L[0] + 2) + (L[0] + 2)*(L[1] + 2) + (L[0] + 2)*(L[1] + 2)*(L[2] + 2) + 2;
    int Nc =  (L[0] + 4)*(L[1] + 4)*(L[2] + 4);
    //int N =density* Nc;
    int N = density * L[0]*L[1]*L[2];
    int Nmd = n_md * m_md;

    //help variable:
    double *d_tmp;
    double px,py,pz;
    cudaMalloc((void**)&d_tmp, sizeof(double)*grid_size);
    mpcd_init(gen, d_x, d_y, d_z, d_vx, d_vy, d_vz, grid_size, N);
    //sum over d_vx blocks 
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vx, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //sum over all the blocks ( d_tmp in this case is the total sum of the mpcd particles initial d_vxs.(which is eventually px))
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&px, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    //do the same sum for py and pz.
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vy, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&py, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vz, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(&pz, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    kernelInit<<<grid_size,blockSize>>>(d_x,d_y,d_z,d_vx, d_vy, d_vz,d_L,px/N,py/N,pz/N,N);

    makezero<<<grid_size,blockSize>>>(T , N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double pos[3] ={0,0,0}; //initial position
    //initMD:
    initMD(d_mdX, d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , _holder , d_Fy_holder , d_Fz_holder , d_L , ux ,pos , n_md , m_md , topology , density);
    calc_accelaration(d_mdX , d_mdY, d_mdZ , _holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_L , Nmd , n_md ,topology ,  ux ,h_md, grid_size);
    cudaFree(d_tmp);


}




__host__ void restarting_simulation(std::string file_name,std::string filename2, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_x , double *d_y , double *d_z ,
double *d_vx , double *d_vy , double *d_vz, double ux,
int N, int Nmd, int last_step, int grid_size)
{

    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    std::ofstream log (file_name + "_log.log", std::ios_base::app); 
    log<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    std::cout<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    mpcd_read_restart_file(filename2 , d_x , d_y , d_z , d_vx , d_vy , d_vz , N);
    std::cout<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    log<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    md_read_restart_file(filename2 , d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
    std::cout<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    log<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    reset_vector_to_zero<<<grid_size,blockSize>>>(d_mdAx, d_mdAy, d_mdAz, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    calc_accelaration(d_mdX , d_mdY, d_mdZ , _holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_L , Nmd ,m_md , topology, ux ,h_md, grid_size);
    
}










//building start_simulation and restart_simulation kernels for ACTIVE part of the code

__host__ void Active_start_simulation(std::string file_name, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_x , double *d_y , double *d_z ,
double *d_vx , double *d_vy , double *d_vz,
double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, 
double *d_fb_kx, double *d_fb_ky, double *d_fb_kz,
double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz,
double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz,
double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot,
double *d_ex, double *d_ey, double *d_ez,
double *h_fa_x, double *h_fa_y, double *h_fa_z,
double *h_fb_x, double *h_fb_y, double *h_fb_z,
double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
curandGenerator_t gen, int grid_size, double real_time, double *gama_T, int *d_random_array, unsigned int seed, int *flag_array, double u_scale)
{
    std::string log_name = file_name + "_log.log";
    std::string trj_name = file_name + "_traj.xyz";
    std::ofstream log (log_name);
    std::ofstream trj (trj_name);
  
    printf("grid_size=%i", grid_size);
    printf( "***WELCOME TO MPCD_MD_LeesEdwards CUDA CODE!***\nBy: Reyhaneh Afghahi Farimani, rhn_a_farimani@yahoo.com. \nThis code comes with a python code to analyse the results.");
    printf("\ninput system:\nensemble:NVT, thermostat= cell_level_Maxwell_Boltzaman_thermostat, Lx=%i,Ly=%i,Lz=%i,shear_rate=%f,density=%i\n", int(L[0]), int(L[1]),int(L[2]), shear_rate, density);
    printf( "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition: shear direction:x , gradiant direction:z , vorticity direction: y\n");
    printf( "Activity is distributed on the monomers randomly(can be varried) ");
    if(topology == 4)
        printf("one active particle in the MPCD fluid.\n");
    if (topology==1)
        printf("A poly[%i]catenane with %i monomer in each ring is embeded in the MPCD fluid.\n" , n_md , m_md);
    if (topology==2)
        printf("A linked[%i]ring with %i monomer in each ring is embeded in the MPCD fluid.\nWarning: the code currently support only linked[2]rings.\n" , n_md , m_md);
    printf("simulation time = %i, measurments accur every %i step.\n", simuationtime, swapsize);
    
    

    /*log<<"***WELCOME TO MPCD_MD_LeesEdwards CUDA CODE!***\nBy: Reyhaneh Afghahi Farimani, rhn_a_farimani@yahoo.com. \nThis code comes with a python code to analyse the results.";
    log<< "\ninput system:\nensemble:NVT, thermostat= cell_level_Maxwell_Boltzaman_thermostat, Lx="<<int(L[0])<<",Ly="<<int(L[1])<<",Lz="<<int(L[2])<<",shear_rate="<<shear_rate<<",density="<<density<<std::endl;
    if (topology==1)
        log<<"A poly["<<n_md<<"]catenane with "<<m_md<<" monomer in each ring is embeded in the MPCD fluid.\n";
    if (topology==2)
        log<<"A linked["<<n_md<<"]ring with "<<m_md<<" monomer in each ring is embeded in the MPCD fluid.\n";
    log<< "SHEAR_FLOW is produced using Lees_Edwards Periodic Boundry Condition: shear direction:x , gradiant direction:z , vorticity direction: y\n";
    log<<"simulation time ="<<simulationtime<<", measurments accur every "<<swapsize<<" step.\n" ;*/

    
    double ux =shear_rate * L[2];
    int Nc = L[0]*L[1]*L[2];
    int N =density* Nc;
    int Nmd = n_md * m_md;
    
    //help variable:
    double *d_tmp;
    double px,py,pz;
    cudaMalloc((void**)&d_tmp, sizeof(double)*grid_size);

    mpcd_init(gen, d_x, d_y, d_z, d_vx, d_vy, d_vz, grid_size, N);

    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vx, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(&px, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);

    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vy, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(&py, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);

    sumCommMultiBlock<<<grid_size, blockSize>>>(d_vz, N, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(&pz, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    
    kernelInit<<<grid_size,blockSize>>>(d_x,d_y,d_z,d_vx, d_vy, d_vz,d_L,px/N,py/N,pz/N,N);

    double pos[3] ={0,0,0};
   
    
    //initMD:
    initMD(d_mdX, d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , _holder , d_Fy_holder , d_Fz_holder , d_L , ux ,pos , n_md , m_md , topology , density);
   
   
    Active_calc_accelaration(d_mdX , d_mdY, d_mdZ , _holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez, ux, density, gama_T, d_L , Nmd , m_md ,topology , real_time, grid_size,1 ,
     N, d_random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("Im in beginning\n");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    //exit(-1);
    }
    cudaFree(d_tmp);


}




__host__ void Active_restarting_simulation(std::string file_name,std::string filename2, int simulationtime , int swapsize , double *d_L,
double *d_mdX , double *d_mdY , double *d_mdZ,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_mdAx , double *d_mdAy , double *d_mdAz,
double *_holder , double *d_Fy_holder, double *d_Fz_holder,
double *d_x , double *d_y , double *d_z ,
double *d_vx , double *d_vy , double *d_vz, 
double *d_fa_kx, double *d_fa_ky, double *d_fa_kz, 
double *d_fb_kx, double *d_fb_ky, double *d_fb_kz,
double *d_Aa_kx, double *d_Aa_ky, double *d_Aa_kz,
double *d_Ab_kx, double *d_Ab_ky, double *d_Ab_kz,
double *d_Ax_tot, double *d_Ay_tot, double *d_Az_tot,
double *d_ex, double *d_ey, double *d_ez,
double *h_fa_x, double *h_fa_y, double *h_fa_z,
double *h_fb_x, double *h_fb_y, double *h_fb_z,
double *d_block_sum_ex, double *d_block_sum_ey, double *d_block_sum_ez,
double ux,
int N, int Nmd, int last_step, int grid_size, double real_time, double *gama_T, int *d_random_array, unsigned int seed, int *flag_array, double u_scale)
{

    cudaMemcpy(d_L, &L, 3*sizeof(double) , cudaMemcpyHostToDevice);
    std::ofstream log (file_name + "_log.log", std::ios_base::app); 
    log<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    std::cout<<"[INFO]Restarting your simulation from step "<<last_step<<":"<<std::endl<<"reading MPCD particle data:"<<std::endl;
    mpcd_read_restart_file(filename2 , d_x , d_y , d_z , d_vx , d_vy , d_vz , N);
    std::cout<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    log<<"[INFO]MPCD data entered into device memory succesfully!"<<std::endl<<"Reading MD particle data:"<<std::endl;
    md_read_restart_file(filename2 , d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
    std::cout<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    log<<"[INFO]MD data entered into device memory succesfully!"<<std::endl;
    reset_vector_to_zero<<<grid_size,blockSize>>>(d_mdAx, d_mdAy, d_mdAz, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    Active_calc_accelaration(d_mdX , d_mdY, d_mdZ , _holder , d_Fy_holder , d_Fz_holder , d_mdAx , d_mdAy , d_mdAz ,d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_ex, d_ey, d_ez, ux, density, gama_T, d_L , Nmd , m_md ,topology , real_time, grid_size,1 , N, d_random_array, seed, d_Ax_tot, d_Ay_tot, d_Az_tot, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, flag_array, u_scale);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
   
}