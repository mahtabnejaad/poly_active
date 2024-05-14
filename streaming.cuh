#include "gpu_md.h"
//streaming: the first function no force field is considered while calculating the new postion of the fluid 
__global__ void mpcd_streaming(double* x,double* y ,double* z,double* vx ,double* vy,double* vz ,double timestep, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N)
    {
        
        x[tid] += timestep * vx[tid];
        y[tid] += timestep * vy[tid];
        z[tid] += timestep * vz[tid];
        
    }
}

__host__ void MPCD_streaming(double* d_x,double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz , double h_mpcd, int N, int grid_size)
{
    mpcd_streaming<<<grid_size,blockSize>>>(d_x, d_y, d_z , d_vx, d_vy, d_vz, h_mpcd ,N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}


//Active MPCD:

__global__ void Active_mpcd_streaming(double* x,double* y ,double* z,double* vx ,double* vy,double* vz ,double timestep, int N, double fa_x, double fa_y, double fa_z, int size, double mass, double mass_fluid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double QQ=-(timestep*timestep)/(2*(size*mass+mass_fluid*N));
    double Q=-timestep/(size*mass+mass_fluid*N);
    
    if (tid<N)
    {
        
        x[tid] += timestep * vx[tid]+QQ * fa_x;
        y[tid] += timestep * vy[tid]+QQ * fa_y;
        z[tid] += timestep * vz[tid]+QQ * fa_z;
        vx[tid]=vx[tid]+Q * fa_x;
        vy[tid]=vy[tid]+Q * fa_y;
        vz[tid]=vz[tid]+Q * fa_z;
        
    }
}

__host__ void Active_MPCD_streaming(double* d_x,double* d_y ,double* d_z,double* d_vx ,double* d_vy,double* d_vz ,double h_mpcd, int N , int grid_size, double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L,int size , double ux, int mass, int mass_fluid, double real_time, int m, int topology, int shared_mem_size)
{
    


    Active_mpcd_streaming<<<grid_size,blockSize>>>(d_x, d_y, d_z , d_vx, d_vy, d_vz, h_mpcd, N, *fa_x, *fa_y, *fa_z, size, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}

