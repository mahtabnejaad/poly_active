#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <exception>
#include<unistd.h>
#include "mpcd_macro.cuh"
//#include "nonslipXperiodic.cuh"
#include "LEBC.cuh"
#include "reduction_sum.cuh"
#include "thermostat.cuh"
#include "streaming.cuh"
#include "noslip_MPCD_streaming.cuh"
#include "collision.cuh"
#include "gallileain_inv.cuh"
#include "rerstart_file.cuh"
#include "md_analyser.cuh"
#include "gpu_md.cuh"
#include "Active_gpu_md.cuh"
#include "noslip_MD_streaming.cuh"
#include "begining.cuh"
#include "logging.cuh"
#include <ctime>
#include "center_of_mass.cuh"
#include "reducefileC.cuh"



int main(int argc, const char* argv[])
{

    //I change this (argc!=16) to argc!=18 . because I added 2 other inputs which are "Activity" and "random_flag".
    // update: now I want to add another input which is boudary condition or BC, so I change argc to 19.
    std::cout<<argc<<"\n";
    if( argc !=19)
    {
        std::cout<<argc<<"\n";
        std::cout<<"Argument parsing failed!\n";
        std::string exeName = argv[0];
        std::cout<<exeName<<"\n";
        std::cout<<"Number of given arguments: "<<argc<<"\n";
        return 1;
    }
    std::string inputfile= argv[1]; //restart file name
    std::string basename = argv[2]; //output base name
    L[0] = atof(argv[3]); //dimension of the simulation in x direction
    L[1] = atof(argv[4]); //dimension of the simulation in y direction
    L[2] = atof(argv[5]); //dimension of the simulation in z direction
    density = atoi(argv[6]); //density of the particles
    n_md = atoi(argv[7]); //number of rings
    m_md = atof(argv[8]); //number of monomer in each ring
    shear_rate = atof(argv[9]); //shear rate
    h_md = atof(argv[10]); //md time step
    h_mpcd = atof(argv[11]); //mpcd time step
    swapsize = atoi(argv[12]);//output interval
    simuationtime = atoi(argv[13]);//final simulation step count
    TIME = atoi(argv[14]);//starting
    topology = atoi(argv[15]);
    Activity = atoi(argv[16]);//I added a parameter called activity which is either 0 or 1 ( either we have activity or we don't)
    random_flag = atoi(argv[17]);//a flag to see if we have random activity or not
    BC = atoi(argv[18]); //a parameter called BC which is ether 1 or 2 or 3( either we have periodic BC or no-slip BC Or combined.)
      
    double ux = shear_rate * L[2];
    double u_scale = 1.0;
    double DR = 0.001; //Rotational friction coefficient
    double Rh = 1.0;
    double delta_ratio = 0.33; 
    double *gama_T;
    gama_T = (double*) malloc(sizeof(double));
    *gama_T = 0.8;
    

    double *temperature;
    temperature = (double*) malloc(sizeof(double));
    *temperature = 1.0;
    double Pe = 10.0; //peclet number
    double l_eq = 1.0; // equilibrium length
    u_scale = Pe * l_eq *DR ; 
    int Nc = L[0]*L[1]*L[2]; //number of cells 
    int N =density* Nc; //number of particles
    int Nmd = n_md * m_md;//total number of monomers
     int grid_size = ((N + blockSize) / blockSize);
    int shared_mem_size = 3 * blockSize * sizeof(double); // allocate shared memory for the intermediate reduction results.
    
     //random generator
     curandGenerator_t gen;
     curandCreateGenerator(&gen, 
         CURAND_RNG_PSEUDO_DEFAULT);
     /* Set seed */
     curandSetPseudoRandomGeneratorSeed(gen, 
         4294967296ULL^time(NULL));
     curandState *devStates;
     cudaMalloc((void **)&devStates, blockSize * grid_size *sizeof(curandState));
     setup_kernel<<<grid_size, blockSize>>>(time(NULL), devStates);
    

    // Allocate device memory for mpcd particle:
    double *d_x, *d_vx , *d_y , *d_vy , *d_z , *d_vz;
    int *d_index;
    cudaMalloc((void**)&d_x, sizeof(double) * N);   cudaMalloc((void**)&d_y, sizeof(double) * N);   cudaMalloc((void**)&d_z, sizeof(double) * N);
    cudaMalloc((void**)&d_vx, sizeof(double) * N);  cudaMalloc((void**)&d_vy, sizeof(double) * N);  cudaMalloc((void**)&d_vz, sizeof(double) * N);
    cudaMalloc((void**)&d_index, sizeof(int) *N);

    //Allocate device memory for reduced mpcd files:
    int skipfactor = 3000;
    double *scalefactor;
    scalefactor = (double*) malloc(sizeof(double));
    *scalefactor = 1.0;
    int NN = int (N/skipfactor);

    double *d_xx; double *d_yy; double *d_zz;
    double *d_xx_lim1; double *d_yy_lim1; double *d_zz_lim1;
    double *d_xx_lim2; double *d_yy_lim2; double *d_zz_lim2;
    double *d_xx_lim3; double *d_yy_lim3; double *d_zz_lim3;
    double *d_xx_lim4; double *d_yy_lim4; double *d_zz_lim4;
    double *d_xx_lim5; double *d_yy_lim5; double *d_zz_lim5;
    double *d_xx_lim6; double *d_yy_lim6; double *d_zz_lim6;
    double *d_xx_lim7; double *d_yy_lim7; double *d_zz_lim7;
    cudaMalloc((void**)&d_xx,sizeof(double)*NN); cudaMalloc((void**)&d_yy,sizeof(double)*NN); cudaMalloc((void**)&d_zz,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim1,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim1,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim1,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim2,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim2,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim2,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim3,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim3,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim3,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim4,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim4,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim4,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim5,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim5,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim5,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim6,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim6,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim6,sizeof(double)*NN);
    cudaMalloc((void**)&d_xx_lim7,sizeof(double)*NN); cudaMalloc((void**)&d_yy_lim7,sizeof(double)*NN); cudaMalloc((void**)&d_zz_lim7,sizeof(double)*NN);
    double *d_endp_x; double *d_endp_y; double *d_endp_z;
    cudaMalloc((void**)&d_endp_x,sizeof(double)*NN); cudaMalloc((void**)&d_endp_y,sizeof(double)*NN); cudaMalloc((void**)&d_endp_z,sizeof(double)*NN);
    
    //int decimalPlaces = 3; // Number of decimal places to keep
    double *roundedNumber_x; double *roundedNumber_y; double *roundedNumber_z;
    cudaMalloc((void**)&roundedNumber_x, sizeof(double) *N);
    cudaMalloc((void**)&roundedNumber_y, sizeof(double) *N);
    cudaMalloc((void**)&roundedNumber_z, sizeof(double) *N);


    //Allocate device memory for reduced mpcd velocity files:
    double *d_vxx; double *d_vyy; double *d_vzz;
    double *d_vxx_lim1; double *d_vyy_lim1; double *d_vzz_lim1;
    double *d_vxx_lim2; double *d_vyy_lim2; double *d_vzz_lim2;
    double *d_vxx_lim3; double *d_vyy_lim3; double *d_vzz_lim3;
    double *d_vxx_lim4; double *d_vyy_lim4; double *d_vzz_lim4;
    double *d_vxx_lim5; double *d_vyy_lim5; double *d_vzz_lim5;
    double *d_vxx_lim6; double *d_vyy_lim6; double *d_vzz_lim6;
    double *d_vxx_lim7; double *d_vyy_lim7; double *d_vzz_lim7;
    cudaMalloc((void**)&d_vxx,sizeof(double)*NN); cudaMalloc((void**)&d_vyy,sizeof(double)*NN); cudaMalloc((void**)&d_vzz,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim1,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim1,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim1,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim2,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim2,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim2,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim3,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim3,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim3,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim4,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim4,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim4,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim5,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim5,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim5,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim6,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim6,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim6,sizeof(double)*NN);
    cudaMalloc((void**)&d_vxx_lim7,sizeof(double)*NN); cudaMalloc((void**)&d_vyy_lim7,sizeof(double)*NN); cudaMalloc((void**)&d_vzz_lim7,sizeof(double)*NN);
    
    //int decimalPlacess = 3; // Number of decimal places to keep
    double *roundedNumber_vx; double *roundedNumber_vy; double *roundedNumber_vz;
    cudaMalloc((void**)&roundedNumber_vx, sizeof(double) *N);
    cudaMalloc((void**)&roundedNumber_vy, sizeof(double) *N);
    cudaMalloc((void**)&roundedNumber_vz, sizeof(double) *N);

    
    //Allocate device memory for box attributes:
    double *d_L, *d_r;   
    cudaMalloc((void**)&d_L, sizeof(double) *3);
    cudaMalloc((void**)&d_r, sizeof(double) *3);
    
    // Allocate device memory for cells:
    double *d_ux , *d_uy , *d_uz;
    int  *d_n;
    double *d_m;
    cudaMalloc((void**)&d_ux, sizeof(double) * Nc); cudaMalloc((void**)&d_uy, sizeof(double) * Nc); cudaMalloc((void**)&d_uz, sizeof(double) * Nc);
    cudaMalloc((void**)&d_n, sizeof(int) * Nc);     cudaMalloc((void**)&d_m, sizeof(double) * Nc);
    //Allocate device memory for rotating angles and matrix:
    double *d_phi , *d_theta,*d_rot;
    cudaMalloc((void**)&d_phi, sizeof(double) * Nc);    cudaMalloc((void**)&d_theta , sizeof(double) *Nc);  cudaMalloc((void**)&d_rot, sizeof(double) * Nc *9);

    int *dn_tot;
    cudaMalloc((void**)&dn_tot, sizeof(int));
    double *N_avg;
    cudaMalloc((void**)&N_avg, sizeof(double));
 
    int *sumblock_n;
    cudaMalloc((void**)&sumblock_n, sizeof(int) * grid_size);

    double *dm_tot;
    cudaMalloc((void**)&dm_tot, sizeof(double));
    double *M_avg;
    cudaMalloc((void**)&M_avg, sizeof(double));

    
    double *sumblock_m;
    cudaMalloc((void**)&sumblock_m, sizeof(double) * grid_size);

    double *a_x, *a_y, *a_z;
    cudaMalloc((void**)&a_x, sizeof(double) * Nc); cudaMalloc((void**)&a_y, sizeof(double) * Nc); cudaMalloc((void**)&a_y, sizeof(double) * Nc);

    double* d_variance;
    cudaMalloc((void**)&d_variance,  sizeof(double) * Nc);
    
 

    curandState *d_States;
    cudaMalloc((void**)&d_States, sizeof(curandState) * Nc);

    //Allocate device memory for cell level thermostat atributes:
    double* d_e, *d_scalefactor;
    cudaMalloc((void**)&d_e , sizeof(double) * Nc);
    cudaMalloc((void**)&d_scalefactor , sizeof(double) * Nc);

    //Allocate device memory for md particle:
    double *d_mdX, *d_mdY, *d_mdZ, *d_mdVx, *d_mdVy , *d_mdVz, *d_mdAx , *d_mdAy, *d_mdAz;
    int *d_mdIndex;
    cudaMalloc((void**)&d_mdX, sizeof(double) * Nmd);    cudaMalloc((void**)&d_mdY, sizeof(double) * Nmd);    cudaMalloc((void**)&d_mdZ, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdVx, sizeof(double) * Nmd);   cudaMalloc((void**)&d_mdVy, sizeof(double) * Nmd);   cudaMalloc((void**)&d_mdVz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdAx, sizeof(double) * Nmd);   cudaMalloc((void**)&d_mdAy, sizeof(double) * Nmd);   cudaMalloc((void**)&d_mdAz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_mdIndex, sizeof(int) * Nmd);
    ///////////////NEW MD attributes:
    double *md_Fx_holder , *md_Fy_holder , *md_Fz_holder;
    cudaMalloc((void**)&md_Fx_holder, sizeof(double) * Nmd *(Nmd ));    cudaMalloc((void**)&md_Fy_holder, sizeof(double) * Nmd *(Nmd ));    cudaMalloc((void**)&md_Fz_holder, sizeof(double) * Nmd *(Nmd));
    

    //Allocate device memory for active and backward forces exerted on each MD particle:
    double *d_fa_kx , *d_fa_ky , *d_fa_kz , *d_fb_kx , *d_fb_ky , *d_fb_kz;
    cudaMalloc((void**)&d_fa_kx, sizeof(double) * Nmd);    cudaMalloc((void**)&d_fa_ky, sizeof(double) * Nmd);    cudaMalloc((void**)&d_fa_kz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_fb_kx, sizeof(double) * Nmd);    cudaMalloc((void**)&d_fb_ky, sizeof(double) * Nmd);    cudaMalloc((void**)&d_fb_kz, sizeof(double) * Nmd);
    
    
    //Allocate device memory for total active and backward forces:
    double *h_fa_x , *h_fa_y , *h_fa_z ;
    double *h_fb_x , *h_fb_y , *h_fb_z ;
    // cudaMalloc((void**)&h_fa_x, sizeof(double)); cudaMalloc((void**)&h_fa_y, sizeof(double)); cudaMalloc((void**)&h_fa_z, sizeof(double));
    // cudaMalloc((void**)&h_fb_x, sizeof(double)); cudaMalloc((void**)&h_fb_y, sizeof(double)); cudaMalloc((void**)&h_fb_z, sizeof(double));
    h_fa_x = (double*) malloc(sizeof(double)); h_fa_y = (double*) malloc(sizeof(double)); h_fa_z = (double*) malloc(sizeof(double));
    h_fb_x = (double*) malloc(sizeof(double)); h_fb_y = (double*) malloc(sizeof(double)); h_fb_z = (double*) malloc(sizeof(double));
    *h_fa_x=0.0; *h_fa_y=0.0; *h_fa_z=0.0;
    *h_fb_x=0.0; *h_fb_y=0.0; *h_fb_z=0.0;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //center of mass attributes:
    double *mdX_tot , *mdY_tot, *mdZ_tot ;
    double *dX_tot , *dY_tot, *dZ_tot ;
    mdX_tot = (double*) malloc(sizeof(double)); mdY_tot = (double*) malloc(sizeof(double)); mdZ_tot = (double*) malloc(sizeof(double));
    dX_tot = (double*) malloc(sizeof(double)); dY_tot = (double*) malloc(sizeof(double)); dZ_tot = (double*) malloc(sizeof(double));
    *mdX_tot=0.0; *mdY_tot=0.0; *mdZ_tot=0.0;
    *dX_tot=0.0; *dY_tot=0.0; *dZ_tot=0.0;

    

/////////////////////////////////////////////// I'd maximize the performance by adjusting new grid_size_ amd blockSize_ this way:
    int device = 0; // GPU device number (you can change this)
    cudaSetDevice(device);

    int blockSize_;  // To store the recommended block size
    blockSize_ = 256; 
    int minGridSize; // To store the minimum grid size

    int dataSize = N; // Adjust this to your problem size
    int* d_data; // Device pointer for data array

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_data, dataSize * sizeof(int));

    // Determine the maximum potential block size
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_, reduce_kernel, 0, dataSize);

    // Print the recommended block size
    std::cout << "Recommended Block Size: " << blockSize_ << std::endl;
    printf ("blocksize=%i", blockSize_);

    // Calculate the grid size based on your data size and the block size
    int grid_size_ = (dataSize + blockSize_ - 1) / blockSize_;
/////////////////////////////////////////////////////////////////////

    //allocate memory for counting zero factors in reducing and limiting the data to a specific box around the MD particles. 

    int *zerofactorsumblock; //an array to sum over all zero blocks. 
    int *zerofactorsumblock1, *zerofactorsumblock2, *zerofactorsumblock3, *zerofactorsumblock4, *zerofactorsumblock5, *zerofactorsumblock6, *zerofactorsumblock7;
    cudaMalloc((void**)&zerofactorsumblock, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock1, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock2, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock3, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock4, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock5, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock6, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorsumblock7, sizeof(int) * grid_size_); 
    int *zerofactor; //a 0/1 array 
    int *zerofactor1,*zerofactor2, *zerofactor3, *zerofactor4, *zerofactor5, *zerofactor6, *zerofactor7;
    cudaMalloc((void**)&zerofactor, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor1, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor2, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor3, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor4, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor5, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor6, sizeof(int) * N);
    cudaMalloc((void**)&zerofactor7, sizeof(int) * N);
    int *zerofactorrsumblock; //an array to sum over all zero blocks.
    int *zerofactorrsumblock1,*zerofactorrsumblock2, *zerofactorrsumblock3, *zerofactorrsumblock4, *zerofactorrsumblock5, *zerofactorrsumblock6, *zerofactorrsumblock7; 
    cudaMalloc((void**)&zerofactorrsumblock, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock1, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock2, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock3, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock4, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock5, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock6, sizeof(int) * grid_size_);
    cudaMalloc((void**)&zerofactorrsumblock7, sizeof(int) * grid_size_);
    int *zerofactorr; //a 0/1 array
    int *zerofactorr1, *zerofactorr2, *zerofactorr3, *zerofactorr4, *zerofactorr5, *zerofactorr6, *zerofactorr7;
    cudaMalloc((void**)&zerofactorr, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr1, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr2, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr3, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr4, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr5, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr6, sizeof(int) * N);
    cudaMalloc((void**)&zerofactorr7, sizeof(int) * N);

//////////////////////////////////////////////////////////////////////

    double *CMsumblock_x; double *CMsumblock_y; double *CMsumblock_z;
    double *CMsumblock_mdx; double *CMsumblock_mdy; double *CMsumblock_mdz;

    cudaMalloc((void**)&CMsumblock_x, grid_size_ * sizeof(double)); cudaMalloc((void**)&CMsumblock_y, grid_size_ * sizeof(double)); cudaMalloc((void**)&CMsumblock_z, grid_size_ * sizeof(double));
    cudaMalloc((void**)&CMsumblock_mdx, grid_size * sizeof(double)); cudaMalloc((void**)&CMsumblock_mdy, grid_size * sizeof(double)); cudaMalloc((void**)&CMsumblock_mdz, grid_size * sizeof(double));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double *h_Xcm , *h_Ycm, *h_Zcm ; 
    //h_Xcm = (double*) malloc(sizeof(double)); h_Ycm = (double*) malloc(sizeof(double)); h_Zcm = (double*) malloc(sizeof(double));
    cudaMalloc((void**)&h_Xcm, sizeof(double)); cudaMalloc((void**)&h_Ycm, sizeof(double)); cudaMalloc((void**)&h_Zcm, sizeof(double));
    
    //Allocate device memory for active and backward accelerations exerted on each MD particle:
   
    double *d_Aa_kx , *d_Aa_ky , *d_Aa_kz , *d_Ab_kx , *d_Ab_ky , *d_Ab_kz;
    cudaMalloc((void**)&d_Aa_kx, sizeof(double) * Nmd);    cudaMalloc((void**)&d_Aa_ky, sizeof(double) * Nmd);    cudaMalloc((void**)&d_Aa_kz, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_Ab_kx, sizeof(double) * Nmd);    cudaMalloc((void**)&d_Ab_ky, sizeof(double) * Nmd);    cudaMalloc((void**)&d_Ab_kz, sizeof(double) * Nmd);



    //Allocate device memory for total active and backward accelerations:
    //host memory:
    double h_Aa_x , h_Aa_y, h_Aa_z;
    double h_Ab_x , h_Ab_y, h_Ab_z;
   
    
    //Allocate device memory for total d_Ax_tot, d_Ay_tot and d_Az_tot:
    double *d_Ax_tot , *d_Ay_tot , *d_Az_tot;
    cudaMalloc((void**)&d_Ax_tot, sizeof(double)* Nmd);    cudaMalloc((void**)&d_Ay_tot, sizeof(double)* Nmd);    cudaMalloc((void**)&d_Az_tot, sizeof(double)* Nmd);

    //Allocate device memory for tanfential vectors ex , ey and ez:
    double *d_ex , *d_ey , *d_ez;
    cudaMalloc((void**)&d_ex, sizeof(double) * Nmd);    cudaMalloc((void**)&d_ey, sizeof(double) * Nmd);    cudaMalloc((void**)&d_ez, sizeof(double) * Nmd);
    
    //Allocate device memory for block sum of ex , ey and ez:
    double *d_block_sum_ex , *d_block_sum_ey , *d_block_sum_ez;
    cudaMalloc((void**)&d_block_sum_ex, sizeof(double) * grid_size);    cudaMalloc((void**)&d_block_sum_ey, sizeof(double) * grid_size);    cudaMalloc((void**)&d_block_sum_ez, sizeof(double) * grid_size);

    //Allocate device memory for random array:
    int *d_random_array;
    cudaMalloc((void**)&d_random_array, sizeof(int) * Nmd);

    int *d_flag_array;
    cudaMalloc((void**)&d_flag_array, sizeof(int) * Nmd);

    unsigned int d_seed;
    //is this seed correct?
    d_seed = (unsigned int)(time(NULL));

    //Allocate device memory for no slip boundary condition attributes:
   

    double  *d_x_o , *d_y_o, *d_z_o; //the x, y and z of the point where a mpcd particle crosses the box walls.
    cudaMalloc((void**)&d_x_o, sizeof(double) * N);  cudaMalloc((void**)&d_y_o, sizeof(double) * N);  cudaMalloc((void**)&d_z_o, sizeof(double) * N);
    double  *d_Vx_o , *d_Vy_o, *d_Vz_o; //the VELOCITY of the point where a mpcd particle crosses the box walls.
    cudaMalloc((void**)&d_Vx_o, sizeof(double) * N);  cudaMalloc((void**)&d_Vy_o, sizeof(double) * N);  cudaMalloc((void**)&d_Vz_o, sizeof(double) * N);
    double *totalT;
    cudaMalloc((void**)&totalT, sizeof(double) * N);


    double  *d_mdX_o , *d_mdY_o, *d_mdZ_o;//the x, y and z of the point where a md particle crosses the box walls.
    cudaMalloc((void**)&d_mdX_o, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdY_o, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdZ_o, sizeof(double) * Nmd);
    double  *d_mdVx_o , *d_mdVy_o, *d_mdVz_o;//the x, y and z of the point where a md particle crosses the box walls.
    cudaMalloc((void**)&d_mdVx_o, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdVy_o, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdVz_o, sizeof(double) * Nmd);



    double *d_dt_min , *d_dt_x, *d_dt_y, *d_dt_z;//crossing point with box for MPCD paricles
    cudaMalloc((void**)&d_dt_min, sizeof(double) * N);  cudaMalloc((void**)&d_dt_x, sizeof(double) * N);  cudaMalloc((void**)&d_dt_y, sizeof(double) * N);  cudaMalloc((void**)&d_dt_z, sizeof(double) * N);
    

    double *d_md_dt_min, *d_md_dt_x, *d_md_dt_y, *d_md_dt_z;//crossing point with box for MD paricles
    cudaMalloc((void**)&d_md_dt_min, sizeof(double) * Nmd);  cudaMalloc((void**)&d_md_dt_x, sizeof(double) * Nmd);  cudaMalloc((void**)&d_md_dt_y, sizeof(double) * Nmd);  cudaMalloc((void**)&d_md_dt_z, sizeof(double) * Nmd);

    double *d_x_wall_dist, *d_y_wall_dist, *d_z_wall_dist, *d_wall_sign_x, *d_wall_sign_y, *d_wall_sign_z;
    cudaMalloc((void**)&d_x_wall_dist, sizeof(double) * N);  cudaMalloc((void**)&d_y_wall_dist, sizeof(double) * N);  cudaMalloc((void**)&d_z_wall_dist, sizeof(double) * N);
    cudaMalloc((void**)&d_wall_sign_x, sizeof(double) * N);  cudaMalloc((void**)&d_wall_sign_y, sizeof(double) * N);  cudaMalloc((void**)&d_wall_sign_z, sizeof(double) * N);


    double *d_mdX_wall_dist, *d_mdY_wall_dist, *d_mdZ_wall_dist, *d_wall_sign_mdX, *d_wall_sign_mdY, *d_wall_sign_mdZ;
    cudaMalloc((void**)&d_mdX_wall_dist, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdY_wall_dist, sizeof(double) * Nmd);  cudaMalloc((void**)&d_mdZ_wall_dist, sizeof(double) * Nmd);
    cudaMalloc((void**)&d_wall_sign_mdX, sizeof(double) * Nmd);  cudaMalloc((void**)&d_wall_sign_mdY, sizeof(double) * Nmd);  cudaMalloc((void**)&d_wall_sign_mdZ, sizeof(double) * Nmd);    
    ///////////////////////////////////////////////////////////////////////////////////

    if(Activity==0 && BC == 1){
        if (TIME ==0)start_simulation(basename, simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, gen , grid_size, totalT);
        else restarting_simulation(basename , inputfile , simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, ux , N , Nmd , TIME , grid_size);
    
        
        double real_time = TIME;
        int T =simuationtime/swapsize +TIME/swapsize;
        int delta = h_mpcd / h_md;
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
        for(int t = TIME/swapsize ; t<T; t++)
        {
            for (int i =0;i<int(swapsize/h_mpcd); i++)
            {
                curandGenerateUniformDouble(gen, d_phi, Nc);
                curandGenerateUniformDouble(gen, d_theta, Nc);
                curandGenerateUniformDouble(gen, d_r, 3);

                

                MPCD_streaming(d_x , d_y , d_z , d_vx , d_vy , d_vz , h_mpcd , N , grid_size);
            

                MD_streaming(d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz ,
                    d_mdAx , d_mdAy , d_mdAz ,md_Fx_holder, md_Fy_holder, md_Fz_holder,
                    h_md , Nmd , density , d_L , ux , grid_size, delta,real_time);

                Sort_begin(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , d_mdX , d_mdY , d_mdZ ,
                    d_mdVx, d_mdVy, d_mdVz, d_mdIndex ,ux , d_L , d_r , N , Nmd , real_time, grid_size);

                MPCD_MD_collision(d_vx , d_vy , d_vz , d_index,
                    d_mdVx , d_mdVy , d_mdVz , d_mdIndex,
                    d_ux , d_uy , d_uz , d_e , d_scalefactor , d_n , d_m ,
                    d_rot , d_theta , d_phi , N , Nmd ,Nc ,devStates , grid_size);
            
                Sort_finish(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , 
                    d_mdX , d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, ux, 
                    d_L , d_r , N , Nmd , real_time, grid_size);
            
                real_time += h_mpcd;
                 

            }
            double *temperature;
            logging(basename + "_log.log" , (t+1)*swapsize , d_mdVx , d_mdVy , d_mdVz , d_vx, d_vy , d_vz, N , Nmd, grid_size , temperature);
            xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
            xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy , d_mdVz, Nmd);
       
        }

        md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
        mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    

    
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_rot); cudaFree(d_phi); cudaFree(d_theta);
        cudaFree(devStates); cudaFree(d_e); cudaFree(d_scalefactor);
        //Free memory MD particles:
        cudaFree(d_mdX);    cudaFree(d_mdY);    cudaFree(d_mdZ);
        cudaFree(d_mdVx);   cudaFree(d_mdVy);   cudaFree(d_mdVz);
        cudaFree(d_mdAx);   cudaFree(d_mdAy);   cudaFree(d_mdAz);
        cudaFree(md_Fx_holder); cudaFree(md_Fy_holder); cudaFree(md_Fz_holder);
        curandDestroyGenerator(gen);

        std::cout<<"The program has terminated succesffuly at time:"<<real_time<<std::endl;
    }

    //this part is specific for when we don't have activity and no-slip boundary condition.
    else if (Activity==0 && BC ==2){

        if (TIME ==0)start_simulation(basename, simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, gen , grid_size, totalT);
        else restarting_simulation(basename , inputfile , simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, ux , N , Nmd , TIME , grid_size);
    
        
        double real_time = TIME;
        int T =simuationtime/swapsize +TIME/swapsize;
        int delta = h_mpcd / h_md;
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
        for(int t = TIME/swapsize ; t<T; t++)
        {
            for (int i =0;i<int(swapsize/h_mpcd); i++)
            {
                curandGenerateUniformDouble(gen, d_phi, Nc);
                curandGenerateUniformDouble(gen, d_theta, Nc);
                curandGenerateUniformDouble(gen, d_r, 3);

                

                noslip_MPCD_streaming(d_x , d_y , d_z , d_vx , d_vy , d_vz , h_mpcd , N , grid_size, d_L, d_dt_x, d_dt_y, d_dt_z, d_dt_min, d_x_o, d_y_o, d_z_o, d_Vx_o , d_Vy_o , d_Vz_o, d_x_wall_dist, d_y_wall_dist, d_z_wall_dist, d_wall_sign_x, d_wall_sign_y, d_wall_sign_z, totalT);
            

                noslip_MD_streaming(d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz ,
                    d_mdAx , d_mdAy , d_mdAz ,md_Fx_holder, md_Fy_holder, md_Fz_holder,
                    h_md , Nmd , density , d_L , ux , grid_size, delta, real_time,
                    d_md_dt_min, d_md_dt_x, d_md_dt_y, d_md_dt_z, d_mdX_o, d_mdY_o, d_mdZ_o, d_mdVx_o, d_mdVy_o, d_mdVz_o, d_mdX_wall_dist, d_mdY_wall_dist, d_mdZ_wall_dist, d_wall_sign_mdX, d_wall_sign_mdY, d_wall_sign_mdZ );

                noslip_Sort_begin(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , d_mdX , d_mdY , d_mdZ ,
                    d_mdVx, d_mdVy, d_mdVz, d_mdIndex ,ux , d_L , d_r , N , Nmd , real_time, grid_size);

                noslip_MPCD_MD_collision(d_vx , d_vy , d_vz , d_index,
                    d_mdVx , d_mdVy , d_mdVz , d_mdIndex,
                    d_ux , d_uy , d_uz , d_e , d_scalefactor , d_n , d_m ,
                    d_rot , d_theta , d_phi , N , Nmd ,Nc ,devStates , grid_size, dn_tot, N_avg, sumblock_n, dm_tot, M_avg, sumblock_m,
                    a_x, a_y, a_z, d_variance, d_States);
            
                noslip_Sort_finish(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , 
                    d_mdX , d_mdY, d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, ux, 
                    d_L , d_r , N , Nmd , real_time, grid_size);
            
                real_time += h_mpcd;
                 

            }
            double *temperature;
            logging(basename + "_log.log" , (t+1)*swapsize , d_mdVx , d_mdVy , d_mdVz , d_vx, d_vy , d_vz, N , Nmd, grid_size , temperature);
            xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
            xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy , d_mdVz, Nmd);
       
        }

        md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
        mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    

    
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_rot); cudaFree(d_phi); cudaFree(d_theta);
        cudaFree(devStates); cudaFree(d_e); cudaFree(d_scalefactor);
        //Free memory MD particles:
        cudaFree(d_mdX);    cudaFree(d_mdY);    cudaFree(d_mdZ);
        cudaFree(d_mdVx);   cudaFree(d_mdVy);   cudaFree(d_mdVz);
        cudaFree(d_mdAx);   cudaFree(d_mdAy);   cudaFree(d_mdAz);
        cudaFree(md_Fx_holder); cudaFree(md_Fy_holder); cudaFree(md_Fz_holder);
        cudaFree(d_x_o); cudaFree(d_y_o); cudaFree(d_z_o);
        cudaFree(d_Vx_o); cudaFree(d_Vy_o); cudaFree(d_Vz_o);
        cudaFree(d_mdX_o); cudaFree(d_mdY_o); cudaFree(d_mdZ_o);
        cudaFree(d_mdVx_o); cudaFree(d_mdVy_o); cudaFree(d_mdVz_o);
        cudaFree(d_dt_min); cudaFree(d_md_dt_min);
        cudaFree(d_dt_x); cudaFree(d_md_dt_x);
        cudaFree(d_dt_y); cudaFree(d_md_dt_y);
        cudaFree(d_dt_z); cudaFree(d_md_dt_z);
        cudaFree(d_x_wall_dist); cudaFree(d_y_wall_dist); cudaFree(d_z_wall_dist);
        cudaFree(d_wall_sign_x); cudaFree(d_wall_sign_y); cudaFree(d_wall_sign_z);
        cudaFree(d_mdX_wall_dist); cudaFree(d_mdY_wall_dist); cudaFree(d_mdZ_wall_dist);
        cudaFree(d_wall_sign_mdX); cudaFree(d_wall_sign_mdY); cudaFree(d_wall_sign_mdZ);
        cudaFree(totalT);
        curandDestroyGenerator(gen);

        std::cout<<"The program has terminated succesffuly at time:"<<real_time<<std::endl;

    }
}

/*
    //this part is specific for when we have activity and periodic boundary condition. 
    else if (Activity==1 && BC==1){ 

        double real_time = TIME;

        double temper0 = 1;
        //temperature = temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size);
        
        //gama_T= temper0 / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; 
        *gama_T = 0.8;

      
        if (TIME ==0) Active_start_simulation(basename, simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, d_fa_kx,d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz,d_Ax_tot, d_Ay_tot, d_Az_tot, d_ex, d_ey,d_ez, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, gen , grid_size, real_time, gama_T, d_random_array, d_seed, d_flag_array, u_scale);
        else Active_restarting_simulation(basename , inputfile , simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz,d_Ax_tot,d_Ay_tot, d_Az_tot, d_ex, d_ey, d_ez, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, ux , N , Nmd , TIME , grid_size, real_time, gama_T, d_random_array, d_seed, d_flag_array, u_scale);
    
        
        //double real_time = TIME;
        int T =simuationtime/swapsize +TIME/swapsize;
        int delta = h_mpcd / h_md;
      
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
        //xyz_trj(basename + "_mpcdtraj.xyz", d_x, d_y , d_z, N);
        reducetraj(basename, d_x, d_y , d_z, d_xx, d_yy, d_zz, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, N, skipfactor, grid_size, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr, roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorrsumblock, blockSize_, grid_size_,
        d_xx_lim1, d_yy_lim1, d_zz_lim1, zerofactorr1,
        d_xx_lim2, d_yy_lim2, d_zz_lim2, zerofactorr2,
        d_xx_lim3, d_yy_lim3, d_zz_lim3, zerofactorr3,
        d_xx_lim4, d_yy_lim4, d_zz_lim4, zerofactorr4,
        d_xx_lim5, d_yy_lim5, d_zz_lim5, zerofactorr5,
        d_xx_lim6, d_yy_lim6, d_zz_lim6, zerofactorr6,
        d_xx_lim7, d_yy_lim7, d_zz_lim7, zerofactorr7,
        d_vxx_lim1, d_vyy_lim1, d_vzz_lim1,
        d_vxx_lim2, d_vyy_lim2, d_vzz_lim2, 
        d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, 
        d_vxx_lim4, d_vyy_lim4, d_vzz_lim4, 
        d_vxx_lim5, d_vyy_lim5, d_vzz_lim5,
        d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  
        d_vxx_lim7, d_vyy_lim7, d_vzz_lim7, 
        zerofactorrsumblock1,zerofactorrsumblock2,zerofactorrsumblock3,zerofactorrsumblock4,zerofactorrsumblock5,zerofactorrsumblock6,zerofactorrsumblock7);

 
        for(int t = TIME/swapsize ; t<T; t++) //T is TIME/swapsize + simulationtime/swapsize. it goes 10 steps from TIME/swapsize till end (in case swapsize is 100 and siumlationtime is 1000). 
        {
            
            for (int i =0;i<int(swapsize/h_mpcd); i++) //swapsize/h_mpcd when swapsize is 100 and h_mpcd is 0.1, is equal to 1000.
            {
                
                curandGenerateUniformDouble(gen, d_phi, Nc);
                curandGenerateUniformDouble(gen, d_theta, Nc);
                curandGenerateUniformDouble(gen, d_r, 3);
                



           

                //double temperature;
                //temperature = 0.0;
                //temperature = temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size);
                //printf("T=%lf\n",temperature);
                //double *gama_T=2.0;
                //*gama_T= temperature / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; 
                //printf("gama_T=%lf\n",*gama_T);

                //go to center of mass reference frame:
                CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, grid_size, shared_mem_size, blockSize_, grid_size_, density, 1, h_Xcm, h_Ycm, h_Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, topology );

                Active_MPCD_streaming(d_x , d_y , d_z , d_vx , d_vy , d_vz ,h_mpcd ,N ,grid_size ,
                 h_fa_x ,h_fa_y ,h_fa_z ,h_fb_x ,h_fb_y ,h_fb_z ,d_ex ,d_ey , d_ez, d_block_sum_ex ,d_block_sum_ey ,d_block_sum_ez ,
                 L ,Nmd ,ux , density ,1 ,real_time ,m_md , topology, shared_mem_size);
            

                Active_MD_streaming(d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz ,
                    d_mdAx , d_mdAy , d_mdAz ,md_Fx_holder, md_Fy_holder, md_Fz_holder, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fa_kz, 
                    d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_Ax_tot, d_Ay_tot, d_Az_tot, d_ex, d_ey, d_ez,
                    h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, 
                    h_md , Nmd ,density ,d_L ,ux ,grid_size ,delta ,real_time ,m_md ,N ,density ,1 , gama_T, d_random_array, d_seed, topology, h_Xcm, h_Ycm, h_Zcm, d_flag_array, u_scale);
                
                Sort_begin(d_x , d_y , d_z ,d_vx, d_vy, d_vz, d_index , d_mdX , d_mdY , d_mdZ ,
                    d_mdVx, d_mdVy, d_mdVz, d_mdIndex ,ux , d_L , d_r , N , Nmd , real_time, grid_size);

                MPCD_MD_collision(d_vx , d_vy , d_vz , d_index,
                    d_mdVx , d_mdVy , d_mdVz , d_mdIndex,
                    d_ux , d_uy , d_uz , d_e , d_scalefactor , d_n , d_m ,
                    d_rot , d_theta , d_phi , N , Nmd ,Nc ,devStates , grid_size);
            
                Sort_finish(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , 
                    d_mdX , d_mdY , d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, ux, 
                    d_L , d_r , N , Nmd , real_time, grid_size);
            
                real_time += h_mpcd;
                 

            }
            
            
            logging(basename + "_log.log" , (t+1)*swapsize , d_mdVx , d_mdVy , d_mdVz , d_vx, d_vy , d_vz, N , Nmd, grid_size, temperature );
           
            //printf("T=%f\n",*temperature);
            *gama_T= (*temperature) / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; /////problem is here?

            xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
            xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy , d_mdVz, Nmd);
            reducetraj(basename, d_x, d_y , d_z, d_xx, d_yy, d_zz, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, N, skipfactor, grid_size, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr,roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorrsumblock, blockSize_, grid_size_,
                d_xx_lim1,  d_yy_lim1,  d_zz_lim1, zerofactorr1,
                d_xx_lim2,  d_yy_lim2,  d_zz_lim2, zerofactorr2,
                d_xx_lim3,  d_yy_lim3,  d_zz_lim3, zerofactorr3,
                d_xx_lim4,  d_yy_lim4,  d_zz_lim4, zerofactorr4,
                d_xx_lim5,  d_yy_lim5,  d_zz_lim5, zerofactorr5,
                d_xx_lim6,  d_yy_lim6,  d_zz_lim6, zerofactorr6,
                d_xx_lim7,  d_yy_lim7,  d_zz_lim7, zerofactorr7,
                d_vxx_lim1, d_vyy_lim1, d_vzz_lim1,
                d_vxx_lim2, d_vyy_lim2, d_vzz_lim2, 
                d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, 
                d_vxx_lim4, d_vyy_lim4, d_vzz_lim4, 
                d_vxx_lim5, d_vyy_lim5, d_vzz_lim5,
                d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  
                d_vxx_lim7, d_vyy_lim7, d_vzz_lim7, 
                zerofactorrsumblock1,zerofactorrsumblock2,zerofactorrsumblock3,zerofactorrsumblock4,zerofactorrsumblock5,zerofactorrsumblock6,zerofactorrsumblock7);
            reducevel(basename, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, d_x, d_y, d_z, N, skipfactor, grid_size,roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorsumblock, blockSize_, grid_size_,
                d_vxx_lim1, d_vyy_lim1, d_vzz_lim1, zerofactor1,
                d_vxx_lim2, d_vyy_lim2, d_vzz_lim2,  zerofactor2,
                d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, zerofactor3,
                d_vxx_lim4, d_vyy_lim4, d_vzz_lim4,  zerofactor4,
                d_vxx_lim5, d_vyy_lim5, d_vzz_lim5, zerofactor5,
                d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  zerofactor6,
                d_vxx_lim7, d_vyy_lim7, d_vzz_lim7,  zerofactor7,
                zerofactorsumblock1 ,zerofactorsumblock2 ,zerofactorsumblock3 ,zerofactorsumblock4 ,zerofactorsumblock5 ,zerofactorsumblock6 ,zerofactorsumblock7);
            xyz_veltraj_both(basename, d_xx, d_yy, d_zz,d_vxx, d_vyy, d_vzz, NN, d_endp_x, d_endp_y, d_endp_z, scalefactor, grid_size);
            //xyz_trj(basename + "_mpcdtraj.xyz", d_x, d_y , d_z, N);
            //xyz_trj(basename + "_mpcdvel.xyz", d_vx, d_vy , d_vz, N);

            
       
        }

        md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
        mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    

    
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_rot); cudaFree(d_phi); cudaFree(d_theta);
        cudaFree(devStates); cudaFree(d_e); cudaFree(d_scalefactor);
        //Free memory MD particles:
        cudaFree(d_mdX);    cudaFree(d_mdY);    cudaFree(d_mdZ);
        cudaFree(d_mdVx);   cudaFree(d_mdVy);   cudaFree(d_mdVz);
        cudaFree(d_mdAx);   cudaFree(d_mdAy);   cudaFree(d_mdAz);
        cudaFree(md_Fx_holder); cudaFree(md_Fy_holder); cudaFree(md_Fz_holder);
        cudaFree(d_fa_kx); cudaFree(d_fa_ky); cudaFree(d_fa_kz);
        cudaFree(d_fb_kx); cudaFree(d_fb_ky); cudaFree(d_fb_kz);
        cudaFree(d_Aa_kx); cudaFree(d_Aa_ky); cudaFree(d_Aa_kz);
        cudaFree(d_Ab_kx); cudaFree(d_Ab_ky); cudaFree(d_Ab_kz);
        cudaFree(d_Ax_tot); cudaFree(d_Ay_tot); cudaFree(d_Az_tot);
        cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_ez);
        cudaFree(d_block_sum_ex); cudaFree(d_block_sum_ey); cudaFree(d_block_sum_ez);
        cudaFree(d_random_array);
        cudaFree(d_L); cudaFree(d_r); 
        cudaFree(d_m); cudaFree(d_n); 
        cudaFree(d_index); cudaFree(d_mdIndex);
        cudaFree(h_Xcm); cudaFree(h_Ycm); cudaFree(h_Zcm);
        cudaFree(CMsumblock_x); cudaFree(CMsumblock_y); cudaFree(CMsumblock_z);
        cudaFree(CMsumblock_mdx); cudaFree(CMsumblock_mdy); cudaFree(CMsumblock_mdz);
        //cudaFree(gama_T);
        cudaFree(d_flag_array);
        cudaFree(d_xx); cudaFree(d_yy); cudaFree(d_zz);
        cudaFree(d_endp_x); cudaFree(d_endp_y); cudaFree(d_endp_z);
        cudaFree(d_vxx); cudaFree(d_vyy); cudaFree(d_vzz);
        cudaFree(roundedNumber_x); cudaFree(roundedNumber_y); cudaFree(roundedNumber_z);
        cudaFree(roundedNumber_vx); cudaFree(roundedNumber_vy); cudaFree(roundedNumber_vz);
        cudaFree(zerofactor);  cudaFree(zerofactorr);
        cudaFree(zerofactorsumblock); cudaFree(zerofactorrsumblock);
        curandDestroyGenerator(gen);
        
        //reducefile_traj();
        //reducefile_vel();

        std::cout<<"The program has terminated succesffuly at time:"<<real_time<<std::endl;
    }



    else if (Activity==1 && BC==2){ 

        double real_time = TIME;

        double temper0 = 1;
        //temperature = temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size);
        
        //gama_T= temper0 / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; 
        *gama_T = 0.8;

       
        if (TIME ==0) Active_start_simulation(basename, simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, d_fa_kx,d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz,d_Ax_tot, d_Ay_tot, d_Az_tot, d_ex, d_ey,d_ez, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, gen , grid_size, real_time, gama_T, d_random_array, d_seed, d_flag_array, u_scale);
        else Active_restarting_simulation(basename , inputfile , simuationtime , swapsize ,d_L, d_mdX , d_mdY , d_mdZ,
        d_mdVx , d_mdVy , d_mdVz , d_mdAx , d_mdAy , d_mdAz , md_Fx_holder, md_Fy_holder,md_Fz_holder,
        d_x , d_y , d_z , d_vx , d_vy , d_vz, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fb_kz, d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz,d_Ax_tot,d_Ay_tot, d_Az_tot, d_ex, d_ey, d_ez, h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, ux , N , Nmd , TIME , grid_size, real_time, gama_T, d_random_array, d_seed, d_flag_array, u_scale);
    
        
        //double real_time = TIME;
        int T =simuationtime/swapsize +TIME/swapsize;
        int delta = h_mpcd / h_md;
      
        xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
        //xyz_trj(basename + "_mpcdtraj.xyz", d_x, d_y , d_z, N);
        reducetraj(basename, d_x, d_y , d_z, d_xx, d_yy, d_zz, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, N, skipfactor, grid_size, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr, roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorrsumblock, blockSize_, grid_size_,
        d_xx_lim1, d_yy_lim1, d_zz_lim1, zerofactorr1,
        d_xx_lim2, d_yy_lim2, d_zz_lim2, zerofactorr2,
        d_xx_lim3, d_yy_lim3, d_zz_lim3, zerofactorr3,
        d_xx_lim4, d_yy_lim4, d_zz_lim4, zerofactorr4,
        d_xx_lim5, d_yy_lim5, d_zz_lim5, zerofactorr5,
        d_xx_lim6, d_yy_lim6, d_zz_lim6, zerofactorr6,
        d_xx_lim7, d_yy_lim7, d_zz_lim7, zerofactorr7,
        d_vxx_lim1, d_vyy_lim1, d_vzz_lim1,
        d_vxx_lim2, d_vyy_lim2, d_vzz_lim2, 
        d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, 
        d_vxx_lim4, d_vyy_lim4, d_vzz_lim4, 
        d_vxx_lim5, d_vyy_lim5, d_vzz_lim5,
        d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  
        d_vxx_lim7, d_vyy_lim7, d_vzz_lim7, 
        zerofactorrsumblock1,zerofactorrsumblock2,zerofactorrsumblock3,zerofactorrsumblock4,zerofactorrsumblock5,zerofactorrsumblock6,zerofactorrsumblock7);

 
        for(int t = TIME/swapsize ; t<T; t++) //T is TIME/swapsize + simulationtime/swapsize. it goes 10 steps from TIME/swapsize till end (in case swapsize is 100 and siumlationtime is 1000). 
        {
            
            for (int i =0;i<int(swapsize/h_mpcd); i++) //swapsize/h_mpcd when swapsize is 100 and h_mpcd is 0.1, is equal to 1000.
            {
                
                curandGenerateUniformDouble(gen, d_phi, Nc);
                curandGenerateUniformDouble(gen, d_theta, Nc);
                curandGenerateUniformDouble(gen, d_r, 3);
                



         

                //double temperature;
                //temperature = 0.0;
                //temperature = temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size);
                //printf("T=%lf\n",temperature);
                //double *gama_T=2.0;
                //*gama_T= temperature / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; 
                //printf("gama_T=%lf\n",*gama_T);

                //go to center of mass reference frame:
                CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, dX_tot, dY_tot, dZ_tot, grid_size, shared_mem_size, blockSize_, grid_size_, density, 1, h_Xcm, h_Ycm, h_Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, topology );

                noslip_Active_MPCD_streaming(d_x , d_y , d_z , d_vx , d_vy , d_vz ,h_mpcd ,N ,grid_size ,
                 h_fa_x ,h_fa_y ,h_fa_z ,h_fb_x ,h_fb_y ,h_fb_z ,d_ex ,d_ey , d_ez, d_block_sum_ex ,d_block_sum_ey ,d_block_sum_ez ,
                 L ,Nmd ,ux , density ,1 ,real_time ,m_md , topology, shared_mem_size, d_L, d_x_latest, d_y_latest, d_z_latest, d_vx_latest, d_vy_latest, d_vz_latest,  d_dt1, d_x_o, d_y_o, d_z_o, d_x_wallDist_P, d_y_wallDist_P, d_z_wallDist_P, d_x_wallDist_N, d_y_wallDist_N, d_z_wallDist_N);
            

                noslip_Active_MD_streaming(d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz ,
                    d_mdAx , d_mdAy , d_mdAz ,md_Fx_holder, md_Fy_holder, md_Fz_holder, d_fa_kx, d_fa_ky, d_fa_kz, d_fb_kx, d_fb_ky, d_fa_kz, 
                    d_Aa_kx, d_Aa_ky, d_Aa_kz, d_Ab_kx, d_Ab_ky, d_Ab_kz, d_Ax_tot, d_Ay_tot, d_Az_tot, d_ex, d_ey, d_ez,
                    h_fa_x, h_fa_y, h_fa_z, h_fb_x, h_fb_y, h_fb_z, d_block_sum_ex, d_block_sum_ey, d_block_sum_ez, 
                    h_md , Nmd ,density ,d_L ,ux ,grid_size ,delta ,real_time ,m_md ,N ,density ,1 , gama_T, d_random_array, d_seed, topology, h_Xcm, h_Ycm, h_Zcm, d_flag_array, u_scale,
                    d_L, d_mdX_latest, d_mdY_latest, d_mdZ_latest,d_mdVx_latest, d_mdVy_latest, d_mdVz_latest, d_md_dt1, d_mdX_o, d_mdY_o, d_mdZ_o, d_mdX_wallDist_P, d_mdY_wallDist_P, d_mdZ_wallDist_P, d_mdX_wallDist_N, d_mdY_wallDist_N, d_mdZ_wallDist_N);
                
                noslip_Sort_begin(d_x , d_y , d_z ,d_vx, d_vy, d_vz, d_index , d_mdX , d_mdY , d_mdZ ,
                    d_mdVx, d_mdVy, d_mdVz, d_mdIndex ,ux , d_L , d_r , N , Nmd , real_time, grid_size);

                MPCD_MD_collision(d_vx , d_vy , d_vz , d_index,
                    d_mdVx , d_mdVy , d_mdVz , d_mdIndex,
                    d_ux , d_uy , d_uz , d_e , d_scalefactor , d_n , d_m ,
                    d_rot , d_theta , d_phi , N , Nmd ,Nc ,devStates , grid_size);
            
                noslip_Sort_finish(d_x , d_y , d_z , d_vx, d_vy, d_vz, d_index , 
                    d_mdX , d_mdY , d_mdZ, d_mdVx, d_mdVy, d_mdVz, d_mdIndex, ux, 
                    d_L , d_r , N , Nmd , real_time, grid_size);
            
                real_time += h_mpcd;
                 

            }
            
            
            logging(basename + "_log.log" , (t+1)*swapsize , d_mdVx , d_mdVy , d_mdVz , d_vx, d_vy , d_vz, N , Nmd, grid_size, temperature );
           
            //printf("T=%f\n",*temperature);
            *gama_T= (*temperature) / ((2*Rh)*(2*Rh)*DR*delta_ratio) ; /////problem is here?

            xyz_trj(basename + "_traj.xyz", d_mdX, d_mdY , d_mdZ, Nmd);
            xyz_trj(basename + "_vel.xyz", d_mdVx, d_mdVy , d_mdVz, Nmd);
            reducetraj(basename, d_x, d_y , d_z, d_xx, d_yy, d_zz, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, N, skipfactor, grid_size, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr,roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorrsumblock, blockSize_, grid_size_,
                d_xx_lim1,  d_yy_lim1,  d_zz_lim1, zerofactorr1,
                d_xx_lim2,  d_yy_lim2,  d_zz_lim2, zerofactorr2,
                d_xx_lim3,  d_yy_lim3,  d_zz_lim3, zerofactorr3,
                d_xx_lim4,  d_yy_lim4,  d_zz_lim4, zerofactorr4,
                d_xx_lim5,  d_yy_lim5,  d_zz_lim5, zerofactorr5,
                d_xx_lim6,  d_yy_lim6,  d_zz_lim6, zerofactorr6,
                d_xx_lim7,  d_yy_lim7,  d_zz_lim7, zerofactorr7,
                d_vxx_lim1, d_vyy_lim1, d_vzz_lim1,
                d_vxx_lim2, d_vyy_lim2, d_vzz_lim2, 
                d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, 
                d_vxx_lim4, d_vyy_lim4, d_vzz_lim4, 
                d_vxx_lim5, d_vyy_lim5, d_vzz_lim5,
                d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  
                d_vxx_lim7, d_vyy_lim7, d_vzz_lim7, 
                zerofactorrsumblock1,zerofactorrsumblock2,zerofactorrsumblock3,zerofactorrsumblock4,zerofactorrsumblock5,zerofactorrsumblock6,zerofactorrsumblock7);
            reducevel(basename, d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, d_x, d_y, d_z, N, skipfactor, grid_size,roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor, zerofactorsumblock, blockSize_, grid_size_,
                d_vxx_lim1, d_vyy_lim1, d_vzz_lim1, zerofactor1,
                d_vxx_lim2, d_vyy_lim2, d_vzz_lim2,  zerofactor2,
                d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, zerofactor3,
                d_vxx_lim4, d_vyy_lim4, d_vzz_lim4,  zerofactor4,
                d_vxx_lim5, d_vyy_lim5, d_vzz_lim5, zerofactor5,
                d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  zerofactor6,
                d_vxx_lim7, d_vyy_lim7, d_vzz_lim7,  zerofactor7,
                zerofactorsumblock1 ,zerofactorsumblock2 ,zerofactorsumblock3 ,zerofactorsumblock4 ,zerofactorsumblock5 ,zerofactorsumblock6 ,zerofactorsumblock7);
            xyz_veltraj_both(basename, d_xx, d_yy, d_zz,d_vxx, d_vyy, d_vzz, NN, d_endp_x, d_endp_y, d_endp_z, scalefactor, grid_size);
            //xyz_trj(basename + "_mpcdtraj.xyz", d_x, d_y , d_z, N);
            //xyz_trj(basename + "_mpcdvel.xyz", d_vx, d_vy , d_vz, N);

            
       
        }

        md_write_restart_file(basename, d_mdX , d_mdY , d_mdZ , d_mdVx , d_mdVy , d_mdVz , Nmd);
        mpcd_write_restart_file(basename ,d_x , d_y , d_z , d_vx , d_vy , d_vz , N);

    

    
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_rot); cudaFree(d_phi); cudaFree(d_theta);
        cudaFree(devStates); cudaFree(d_e); cudaFree(d_scalefactor);
        //Free memory MD particles:
        cudaFree(d_mdX);    cudaFree(d_mdY);    cudaFree(d_mdZ);
        cudaFree(d_mdVx);   cudaFree(d_mdVy);   cudaFree(d_mdVz);
        cudaFree(d_mdAx);   cudaFree(d_mdAy);   cudaFree(d_mdAz);
        cudaFree(md_Fx_holder); cudaFree(md_Fy_holder); cudaFree(md_Fz_holder);
        cudaFree(d_fa_kx); cudaFree(d_fa_ky); cudaFree(d_fa_kz);
        cudaFree(d_fb_kx); cudaFree(d_fb_ky); cudaFree(d_fb_kz);
        cudaFree(d_Aa_kx); cudaFree(d_Aa_ky); cudaFree(d_Aa_kz);
        cudaFree(d_Ab_kx); cudaFree(d_Ab_ky); cudaFree(d_Ab_kz);
        cudaFree(d_Ax_tot); cudaFree(d_Ay_tot); cudaFree(d_Az_tot);
        cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_ez);
        cudaFree(d_block_sum_ex); cudaFree(d_block_sum_ey); cudaFree(d_block_sum_ez);
        cudaFree(d_random_array);
        cudaFree(d_L); cudaFree(d_r); 
        cudaFree(d_m); cudaFree(d_n); 
        cudaFree(d_index); cudaFree(d_mdIndex);
        cudaFree(h_Xcm); cudaFree(h_Ycm); cudaFree(h_Zcm);
        cudaFree(CMsumblock_x); cudaFree(CMsumblock_y); cudaFree(CMsumblock_z);
        cudaFree(CMsumblock_mdx); cudaFree(CMsumblock_mdy); cudaFree(CMsumblock_mdz);
        //cudaFree(gama_T);
        cudaFree(d_flag_array);
        cudaFree(d_xx); cudaFree(d_yy); cudaFree(d_zz);
        cudaFree(d_endp_x); cudaFree(d_endp_y); cudaFree(d_endp_z);
        cudaFree(d_vxx); cudaFree(d_vyy); cudaFree(d_vzz);
        cudaFree(roundedNumber_x); cudaFree(roundedNumber_y); cudaFree(roundedNumber_z);
        cudaFree(roundedNumber_vx); cudaFree(roundedNumber_vy); cudaFree(roundedNumber_vz);
        cudaFree(zerofactor);  cudaFree(zerofactorr);
        cudaFree(zerofactorsumblock); cudaFree(zerofactorrsumblock);
        cudaFree(d_x_latest); cudaFree(d_y_latest); cudaFree(d_z_latest); 
        cudaFree(d_mdX_latest); cudaFree(d_mdY_latest); cudaFree(d_mdZ_latest);
        cudaFree(d_vx_latest); cudaFree(d_vy_latest); cudaFree(d_vz_latest); 
        cudaFree(d_mdVx_latest); cudaFree(d_mdVy_latest); cudaFree(d_mdVz_latest);
        cudaFree(d_x_o); cudaFree(d_y_o); cudaFree(d_z_o);
        cudaFree(d_mdX_o); cudaFree(d_mdY_o); cudaFree(d_mdZ_o);
        cudaFree(d_dt1); cudaFree(d_md_dt1);
        cudaFree(d_x_wallDist_P); cudaFree(d_y_wallDist_P); cudaFree(d_z_wallDist_P);
        cudaFree(d_x_wallDist_N); cudaFree(d_y_wallDist_N); cudaFree(d_z_wallDist_N);
        cudaFree(d_mdX_wallDist_P); cudaFree(d_mdY_wallDist_P); cudaFree(d_mdZ_wallDist_P);
        cudaFree(d_mdX_wallDist_N); cudaFree(d_mdY_wallDist_N); cudaFree(d_mdZ_wallDist_N);







        curandDestroyGenerator(gen);
        
        //reducefile_traj();
        //reducefile_vel();

        std::cout<<"The program has terminated succesffuly at time:"<<real_time<<std::endl;
    }

}

*/





