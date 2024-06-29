#include "gpu_md.h"

__global__ void gotoCMframe(double *X, double *Y, double *Z, double *Xcm,double *Ycm, double *Zcm, double *Vx, double *Vy, double *Vz, double *Vxcm,double *Vycm, double *Vzcm, int size){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid < size)
    {
        
        X[tid] = X[tid] - *Xcm;
        Y[tid] = Y[tid] - *Ycm;
        Z[tid] = Z[tid] - *Zcm;

        Vx[tid] = Vx[tid] - *Vxcm;
        Vy[tid] = Vy[tid] - *Vycm;
        Vz[tid] = Vz[tid] - *Vzcm;



    }
}

__global__ void backtoLabframe(double *X, double *Y, double *Z, double *Xcm,double *Ycm, double *Zcm, double *Vx, double *Vy, double *Vz, double *Vxcm,double *Vycm, double *Vzcm, int size){
    
        int tid = blockIdx.x * blockDim.x + threadIdx.x ;
        if (tid < size)
        {
            
            X[tid] = X[tid] + *Xcm;
            Y[tid] = Y[tid] + *Ycm;
            Z[tid] = Z[tid] + *Zcm;

            Vx[tid] = Vx[tid] + *Vxcm;
            Vy[tid] = Vy[tid] + *Vycm;
            Vz[tid] = Vz[tid] + *Vzcm;

        }
}

__global__ void gotoOUTBOXCMframe(double *X, double *Y, double *Z, double *Xcm, double *Ycm, double *Zcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vx, double *Vy, double *Vz, double *Vxcm, double *Vycm, double *Vzcm, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, int size, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid < size){
        if((X[tid]+ *Xcm) > L[0]/2 || (X[tid]+ *Xcm) < -L[0]/2 || (Y[tid] + *Ycm) > L[1]/2 || (Y[tid] + *Ycm) < -L[1]/2 || (Z[tid] + *Zcm) > L[2]/2 || (Z[tid] + *Zcm) < -L[2]/2){
        
            X[tid] = X[tid] - *Xcm_out;
            Y[tid] = Y[tid] - *Ycm_out;
            Z[tid] = Z[tid] - *Zcm_out;

            Vx[tid] = Vx[tid] - *Vxcm_out;
            Vy[tid] = Vy[tid] - *Vycm_out;
            Vz[tid] = Vz[tid] - *Vzcm_out;



        }
    }
}

__global__ void gobackOUTBOX_OLDCMframe(double *X, double *Y, double *Z, double *Xcm, double *Ycm, double *Zcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vx, double *Vy, double *Vz, double *Vxcm, double *Vycm, double *Vzcm, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, int size, double *L, int *n_outbox){

    int tid = blockIdx.x * blockDim.x + threadIdx.x ;
    if (tid < size){
       if(n_outbox[tid] == 1){
        
            X[tid] = X[tid] + *Xcm_out;
            Y[tid] = Y[tid] + *Ycm_out;
            Z[tid] = Z[tid] + *Zcm_out;

            Vx[tid] = Vx[tid] + *Vxcm_out;
            Vy[tid] = Vy[tid] + *Vycm_out;
            Vz[tid] = Vz[tid] + *Vzcm_out;



        }
    }
}





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

__host__ void Active_MPCD_streaming(double* d_x,double* d_y ,double* d_z,double* d_vx ,double* d_vy,double* d_vz, double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, 
double h_mpcd, int N, int grid_size, double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L,int size , double ux, double mass, double mass_fluid, double real_time, int m, int topology, int shared_mem_size, int shared_mem_size_)
{
    
    gotoCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    Active_mpcd_streaming<<<grid_size,blockSize>>>(d_x, d_y, d_z , d_vx, d_vy, d_vz, h_mpcd, N, *fa_x, *fa_y, *fa_z, size, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

     backtoLabframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
}

