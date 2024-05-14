//#include <stdio.h>

/*int reducefile_traj() {
    char inputFileName1[] = "0.1_mpcdtraj.xyz";
    char outputFileName1[] = "0.1_mpcdtraj_reduced.xyz";
    int skipFactor1 = 1000;  // Adjust this to control the level of reduction

    FILE *inputFile1, *outputFile;
    char line[70*60*60*10];  // Assuming a maximum line length of 256 characters

    inputFile1 = fopen(inputFileName1, "r");
    outputFile = fopen(outputFileName1, "w");

    if (inputFile1 == NULL || outputFile == NULL) {
        perror("Error opening files");
        return 1;
    }

    int lineCounter = 0;

    while (fgets(line, sizeof(line), inputFile1) != NULL) {
        if (lineCounter % skipFactor1 == 0) {
            // Write the line to the output file
            fprintf(outputFile, "%s", line);
        }
        lineCounter++;
    }

    fclose(inputFile1);
    fclose(outputFile);

    return 0;
}

int reducefile_vel() {
    char inputFileName2[] = "0.1_mpcdvel.xyz";
    char outputFileName2[] = "0.1_mpcdvel_reduced.xyz";
    int skipFactor2 = 1000;  // Adjust this to control the level of reduction

    FILE *inputFile2, *outputFile2;
    char line2[70*60*60*10];  // Assuming a maximum line length of 256 characters

    inputFile2 = fopen(inputFileName2, "r");
    outputFile2 = fopen(outputFileName2, "w");

    if (inputFile2 == NULL || outputFile2 == NULL) {
        perror("Error opening files");
        return 1;
    }

    int lineCounter2 = 0;

    while (fgets(line2, sizeof(line2), inputFile2) != NULL) {
        if (lineCounter2 % skipFactor2 == 0) {
            // Write the line to the output file
            fprintf(outputFile2, "%s", line2);
        }
        lineCounter2++;
    }

    fclose(inputFile2);
    fclose(outputFile2);

    return 0;
}*/
__global__ void spatial_limiting_kernel(double *d_xx,  double *d_yy, double *d_zz, double *d_vxx, double *d_vyy, double *d_vzz,
double *d_xx_lim1,  double *d_yy_lim1, double *d_zz_lim1, int *zerofactorr1,
double *d_xx_lim2,  double *d_yy_lim2, double *d_zz_lim2, int *zerofactorr2,
double *d_xx_lim3,  double *d_yy_lim3, double *d_zz_lim3, int *zerofactorr3,
double *d_xx_lim4,  double *d_yy_lim4, double *d_zz_lim4, int *zerofactorr4,
double *d_xx_lim5,  double *d_yy_lim5, double *d_zz_lim5, int *zerofactorr5,
double *d_xx_lim6,  double *d_yy_lim6, double *d_zz_lim6, int *zerofactorr6,
double *d_xx_lim7, double *d_yy_lim7, double *d_zz_lim7, int *zerofactorr7, 
 double *d_vxx_lim1, double *d_vyy_lim1, double *d_vzz_lim1, 
 double *d_vxx_lim2, double *d_vyy_lim2, double *d_vzz_lim2,
 double *d_vxx_lim3, double *d_vyy_lim3, double *d_vzz_lim3,
 double *d_vxx_lim4, double *d_vyy_lim4, double *d_vzz_lim4,
 double *d_vxx_lim5, double *d_vyy_lim5, double *d_vzz_lim5,
 double *d_vxx_lim6, double *d_vyy_lim6, double *d_vzz_lim6,
 double *d_vxx_lim7, double *d_vyy_lim7, double *d_vzz_lim7, int NN){

    int tidd= blockIdx.x*blockDim.x + threadIdx.x;
    //we should first make d_xx_lim1 and ... elements equal to 1000.000000
    if (tidd<NN){

        if(d_yy[tidd]>=0 && d_yy[tidd]<10){

            d_yy_lim1[tidd]=d_yy[tidd];
            d_xx_lim1[tidd]=d_xx[tidd];
            d_zz_lim1[tidd]=d_zz[tidd];
            d_vyy_lim1[tidd]=d_vyy[tidd];
            d_vxx_lim1[tidd]=d_vxx[tidd];
            d_vzz_lim1[tidd]=d_vzz[tidd];
            
            zerofactorr1[tidd] = 1;


        }
      

        
        else if(d_yy[tidd]>=10 && d_yy[tidd]<20){

            d_yy_lim2[tidd]=d_yy[tidd];
            d_xx_lim2[tidd]=d_xx[tidd];
            d_zz_lim2[tidd]=d_zz[tidd];
            d_vyy_lim2[tidd]=d_vyy[tidd];
            d_vxx_lim2[tidd]=d_vxx[tidd];
            d_vzz_lim2[tidd]=d_vzz[tidd];
          
            zerofactorr2[tidd] = 1;


        }
        

        else if(d_yy[tidd]>=20 && d_yy[tidd]<30){

            d_yy_lim3[tidd]=d_yy[tidd];
            d_xx_lim3[tidd]=d_xx[tidd];
            d_zz_lim3[tidd]=d_zz[tidd];
            d_vyy_lim3[tidd]=d_vyy[tidd];
            d_vxx_lim3[tidd]=d_vxx[tidd];
            d_vzz_lim3[tidd]=d_vzz[tidd];
      
            zerofactorr3[tidd] = 1;


        }


        else if(d_yy[tidd]>=-10 && d_yy[tidd]<0){

            d_yy_lim4[tidd]=d_yy[tidd];
            d_xx_lim4[tidd]=d_xx[tidd];
            d_zz_lim4[tidd]=d_zz[tidd];
            d_vyy_lim4[tidd]=d_vyy[tidd];
            d_vxx_lim4[tidd]=d_vxx[tidd];
            d_vzz_lim4[tidd]=d_vzz[tidd];

            zerofactorr4[tidd] = 1;


        }
   

        else if(d_yy[tidd]>=-20 && d_yy[tidd]<-10){

            d_yy_lim5[tidd]=d_yy[tidd];
            d_xx_lim5[tidd]=d_xx[tidd];
            d_zz_lim5[tidd]=d_zz[tidd];
            d_vyy_lim5[tidd]=d_vyy[tidd];
            d_vxx_lim5[tidd]=d_vxx[tidd];
            d_vzz_lim5[tidd]=d_vzz[tidd];
       
            zerofactorr5[tidd] = 1;


        }
       

        else if(d_yy[tidd]>=-30 && d_yy[tidd]<-20){

            d_yy_lim6[tidd]=d_yy[tidd];
            d_xx_lim6[tidd]=d_xx[tidd];
            d_zz_lim6[tidd]=d_zz[tidd];
            d_vyy_lim6[tidd]=d_vyy[tidd];
            d_vxx_lim6[tidd]=d_vxx[tidd];
            d_vzz_lim6[tidd]=d_vzz[tidd];
        
            zerofactorr6[tidd] = 1;


        }


        else {

            d_yy_lim7[tidd]=d_yy[tidd];
            d_xx_lim7[tidd]=d_xx[tidd];
            d_zz_lim7[tidd]=d_zz[tidd];
            d_vyy_lim7[tidd]=d_vyy[tidd];
            d_vxx_lim7[tidd]=d_vxx[tidd];
            d_vzz_lim7[tidd]=d_vzz[tidd];
          
            zerofactorr7[tidd] = 1;


        }
 
    }

}

__global__ void velocity_limiting_kernel(double *d_vxx, double *d_vyy, double *d_vzz, double *d_x, double *d_y, double *d_z,
double *d_vxx_lim1, double *d_vyy_lim1, double *d_vzz_lim1, 
int *zerofactor1,
double *d_vxx_lim2, double *d_vyy_lim2, double *d_vzz_lim2, 
int *zerofactor2,
double *d_vxx_lim3, double *d_vyy_lim3, double *d_vzz_lim3, 
int *zerofactor3,
double *d_vxx_lim4, double *d_vyy_lim4, double *d_vzz_lim4, 
int *zerofactor4,
double *d_vxx_lim5, double *d_vyy_lim5, double *d_vzz_lim5, 
int *zerofactor5,
double *d_vxx_lim6, double *d_vyy_lim6, double *d_vzz_lim6, 
int *zerofactor6,
double *d_vxx_lim7, double *d_vyy_lim7, double *d_vzz_lim7, 
int *zerofactor7, int NN){

        int tidd= blockIdx.x*blockDim.x + threadIdx.x;
        if (tidd<NN){

            if(d_y[tidd]>=0 && d_y[tidd]<10){
                d_vyy_lim1[tidd]=d_vyy[tidd];
                d_vxx_lim1[tidd]=d_vxx[tidd];
                d_vzz_lim1[tidd]=d_vzz[tidd];
                zerofactor1[tidd]=1;
            }
            else if(d_y[tidd]>=10 && d_y[tidd]<20){
                d_vyy_lim2[tidd]=d_vyy[tidd];
                d_vxx_lim2[tidd]=d_vxx[tidd];
                d_vzz_lim2[tidd]=d_vzz[tidd];
                zerofactor2[tidd]=1;
                
            }
            else if(d_y[tidd]>=20 && d_y[tidd]<30){
                d_vyy_lim3[tidd]=d_vyy[tidd];
                d_vxx_lim3[tidd]=d_vxx[tidd];
                d_vzz_lim3[tidd]=d_vzz[tidd];
                zerofactor3[tidd]=1;
            }
            else if(d_y[tidd]>=-10 && d_y[tidd]<0){
                d_vyy_lim4[tidd]=d_vyy[tidd];
                d_vxx_lim4[tidd]=d_vxx[tidd];
                d_vzz_lim4[tidd]=d_vzz[tidd];
                zerofactor4[tidd]=1;


            }
            else if(d_y[tidd]>=-20 && d_y[tidd]<-10){
                d_vyy_lim5[tidd]=d_vyy[tidd];
                d_vxx_lim5[tidd]=d_vxx[tidd];
                d_vzz_lim5[tidd]=d_vzz[tidd];
                zerofactor5[tidd]=1;
            }
            else if(d_y[tidd]>=-30 && d_y[tidd]<-20){
                d_vyy_lim6[tidd]=d_vyy[tidd];
                d_vxx_lim6[tidd]=d_vxx[tidd];
                d_vzz_lim6[tidd]=d_vzz[tidd];
                zerofactor6[tidd]=1;
            }
            else {
                d_vyy_lim7[tidd]=d_vyy[tidd];
                d_vxx_lim7[tidd]=d_vxx[tidd];
                d_vzz_lim7[tidd]=d_vzz[tidd];
                zerofactor7[tidd]=1;
            }
        }


}

__global__ void initializeArray(double *array, int size, double value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        array[index] = value;
    }
}

__global__ void initializeArrayint(int *array, int size,int value){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        array[index] = value;
    }
}

__global__ void reduceTraj(double *d_x,double *d_y, double *d_z, double *d_xx, double *d_yy, double *d_zz, int N, int skipfactor, double *roundedNumber_x,double *roundedNumber_y,double *roundedNumber_z, int *zerofactorr){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int tidd = int(tid/skipfactor);
    int decimalPlaces = 3; // Number of decimal places to keep
    
 
    if (tid<N)
    {

        if (tid%skipfactor == 0)
        {

           
            //if (d_x[tid] < 5.0 && d_y[tid] < 5.0 && d_z[tid] < 5.0 && d_x[tid] > -5.0 && d_y[tid] > -5.0 && d_z[tid] > -5.0) {
            //if (d_y[tid]<10 && d_y[tid]>=0){


               

                roundedNumber_x[tid] = roundf(d_x[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_x[tid]=d_x[tid];
                roundedNumber_y[tid] = roundf(d_y[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_y[tid]=d_y[tid];
                roundedNumber_z[tid]= roundf(d_z[tid] * pow(10, decimalPlaces)) / pow(10, decimalPlaces);
                //roundedNumber_z[tid]=d_z[tid];

                d_xx[tidd]=roundedNumber_x[tid];
                d_yy[tidd]=roundedNumber_y[tid];
                d_zz[tidd]=roundedNumber_z[tid];
            //}
            //}
            //else{
            //    zerofactorr[tid] = 1;
            //    d_xx[tidd]=1000.0000000;
            //    d_yy[tidd]=1000.0000000;
            //    d_zz[tidd]=1000.0000000;
            //}

                
        }
        
    }
  
}


__global__ void reduceVel( double *d_vx,double *d_vy, double *d_vz, double *d_vxx, double *d_vyy, double *d_vzz, double *d_x, double *d_y, double *d_z, int N, int skipfactor, double *roundedNumber_vx,double *roundedNumber_vy,double *roundedNumber_vz, int *zero_factor){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int tidd = int(tid/skipfactor);
    int decimalPlacess = 3; // Number of decimal places to keep
    

    if (tid<N)
    {

        if (tid%skipfactor == 0)
        {
            












            //if (d_x[tid] < 5 && d_y[tid] < 5 && d_z[tid] < 5 && d_x[tid] > -5 && d_y[tid] > -5 && d_z[tid] > -5 ){
            //if (d_y[tid]<10 && d_y[tid]>=0){

                roundedNumber_vx[tid] = roundf(d_vx[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vx[tid]=d_vx[tid];
           
                roundedNumber_vy[tid] = roundf(d_vy[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vy[tid]=d_vy[tid];
            
                roundedNumber_vz[tid]= roundf(d_vz[tid] * pow(10, decimalPlacess)) / pow(10, decimalPlacess);
                //roundedNumber_vz[tid]=d_vz[tid];

                d_vxx[tidd]=roundedNumber_vx[tid];
                d_vyy[tidd]=roundedNumber_vy[tid];
                d_vzz[tidd]=roundedNumber_vz[tid];
            //}
            //}
            //else{
            //      zero_factor[tid] = 1;
                 
            //      d_vxx[tidd]=1000.0000000;
            //      d_vyy[tidd]=1000.0000000;
            //      d_vzz[tidd]=1000.0000000;
            //    }
        }

     
    }

}

__host__ void reducetraj(std::string basename, double *d_x,double *d_y, double *d_z,double *d_xx, double *d_yy, double *d_zz,double *d_vx, double *d_vy, double *d_vz, double *d_vxx, double *d_vyy, double *d_vzz,
int N, int skipfactor,int grid_size, double *roundedNumber_x,double *roundedNumber_y,double *roundedNumber_z, int *zerofactorr,double *roundedNumber_vx,double *roundedNumber_vy,double *roundedNumber_vz, int *zerofactor, int *zerofactorrsumblock, int blockSize_ ,int grid_size_,
double *d_xx_lim1,  double *d_yy_lim1, double *d_zz_lim1, int *zerofactorr1,
double *d_xx_lim2,  double *d_yy_lim2, double *d_zz_lim2, int *zerofactorr2,
double *d_xx_lim3,  double *d_yy_lim3, double *d_zz_lim3, int *zerofactorr3,
double *d_xx_lim4,  double *d_yy_lim4, double *d_zz_lim4, int *zerofactorr4,
double *d_xx_lim5,  double *d_yy_lim5, double *d_zz_lim5, int *zerofactorr5,
double *d_xx_lim6,  double *d_yy_lim6, double *d_zz_lim6, int *zerofactorr6,
double *d_xx_lim7,  double *d_yy_lim7, double *d_zz_lim7, int *zerofactorr7,
double *d_vxx_lim1,  double *d_vyy_lim1, double *d_vzz_lim1,
double *d_vxx_lim2,  double *d_vyy_lim2, double *d_vzz_lim2, 
double *d_vxx_lim3,  double *d_vyy_lim3, double *d_vzz_lim3, 
double *d_vxx_lim4,  double *d_vyy_lim4, double *d_vzz_lim4,
double *d_vxx_lim5,  double *d_vyy_lim5, double *d_vzz_lim5, 
double *d_vxx_lim6,  double *d_vyy_lim6, double *d_vzz_lim6, 
double *d_vxx_lim7,  double *d_vyy_lim7, double *d_vzz_lim7,
int *zerofactorrsumblock1,int *zerofactorrsumblock2,int *zerofactorrsumblock3,int *zerofactorrsumblock4,int *zerofactorrsumblock5,int *zerofactorrsumblock6,int *zerofactorrsumblock7 ){


    int NN = int(N/skipfactor);
    int shared_mem_size_ = 3 * blockSize_ * sizeof(int);
    int block_sum_zerofactorr[grid_size_];

    int block_sum_zerofactorr1[grid_size_];
    int block_sum_zerofactorr2[grid_size_];
    int block_sum_zerofactorr3[grid_size_];
    int block_sum_zerofactorr4[grid_size_];
    int block_sum_zerofactorr5[grid_size_];
    int block_sum_zerofactorr6[grid_size_];
    int block_sum_zerofactorr7[grid_size_];

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr1, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr2, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr3, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr4, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr5, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr6, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactorr7, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    reduceTraj<<<grid_size, blockSize>>>(d_x, d_y, d_z, d_xx, d_yy, d_zz, N, skipfactor, roundedNumber_x, roundedNumber_y, roundedNumber_z, zerofactorr);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    reduceVel<<<grid_size, blockSize>>>(d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, d_x, d_y, d_z, N, skipfactor, roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim1, NN, 1000.000000);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim2, NN, 1000.000000);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_xx_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_yy_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_yy_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_zz_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_zz_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    spatial_limiting_kernel<<<grid_size, blockSize>>>(d_xx, d_yy, d_zz, d_vxx, d_vyy, d_vzz,
        d_xx_lim1, d_yy_lim1, d_zz_lim1, zerofactorr1,
        d_xx_lim2,  d_yy_lim2, d_zz_lim2, zerofactorr2,
        d_xx_lim3,  d_yy_lim3, d_zz_lim3, zerofactorr3,
        d_xx_lim4,  d_yy_lim4, d_zz_lim4, zerofactorr4,
        d_xx_lim5,  d_yy_lim5, d_zz_lim5, zerofactorr5,
        d_xx_lim6,  d_yy_lim6, d_zz_lim6, zerofactorr6,
        d_xx_lim7, d_yy_lim7, d_zz_lim7, zerofactorr7,
        d_vxx_lim1, d_vyy_lim1, d_vzz_lim1, 
        d_vxx_lim2, d_vyy_lim2, d_vzz_lim2, 
        d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, 
        d_vxx_lim4, d_vyy_lim4, d_vzz_lim4, 
        d_vxx_lim5, d_vyy_lim5, d_vzz_lim5,
        d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  
        d_vxx_lim7, d_vyy_lim7, d_vzz_lim7, NN);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    //in this line we should sum over all zerofactorr elements to calculate zerofactorr_sum
    //intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr, zerofactorrsumblock, N);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr1, zerofactorrsumblock1, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr2, zerofactorrsumblock2, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr3, zerofactorrsumblock3, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr4, zerofactorrsumblock4, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr5, zerofactorrsumblock5, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr6, zerofactorrsumblock6, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactorr7, zerofactorrsumblock7, NN);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaMemcpy(block_sum_zerofactorr, zerofactorrsumblock, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(block_sum_zerofactorr1, zerofactorrsumblock1, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr2, zerofactorrsumblock2, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr3, zerofactorrsumblock3, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr4, zerofactorrsumblock4, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr5, zerofactorrsumblock5, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr6, zerofactorrsumblock6, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactorr7, zerofactorrsumblock7, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);

    int d_zerofactorr_sum = 0;

    int d_zerofactorr1_sum = 0;
    int d_zerofactorr2_sum = 0;
    int d_zerofactorr3_sum = 0;
    int d_zerofactorr4_sum = 0;
    int d_zerofactorr5_sum = 0;
    int d_zerofactorr6_sum = 0;
    int d_zerofactorr7_sum = 0;

    for (int j = 0; j < grid_size; j++)
        {
            d_zerofactorr_sum += block_sum_zerofactorr[j];

            d_zerofactorr1_sum += block_sum_zerofactorr1[j];
            d_zerofactorr2_sum += block_sum_zerofactorr2[j];
            d_zerofactorr3_sum += block_sum_zerofactorr3[j];
            d_zerofactorr4_sum += block_sum_zerofactorr4[j];
            d_zerofactorr5_sum += block_sum_zerofactorr5[j];
            d_zerofactorr6_sum += block_sum_zerofactorr6[j];
            d_zerofactorr7_sum += block_sum_zerofactorr7[j];



        }

    d_zerofactorr1_sum = NN-d_zerofactorr1_sum; 
    d_zerofactorr2_sum = NN-d_zerofactorr2_sum;
    d_zerofactorr3_sum = NN-d_zerofactorr3_sum;
    d_zerofactorr4_sum = NN-d_zerofactorr4_sum;
    d_zerofactorr5_sum = NN-d_zerofactorr5_sum;
    d_zerofactorr6_sum = NN-d_zerofactorr6_sum;
    d_zerofactorr7_sum = NN-d_zerofactorr7_sum;
    
    
    //printf("number of zeros is = %i\n", d_zerofactorr_sum);
    xyz_trj_mpcd(basename + "_mpcdtraj___reduced.xyz", d_xx, d_yy , d_zz, NN, d_zerofactorr_sum);

    xyz_trj_mpcd(basename + "y0mpcdtraj10.xyz", d_xx_lim1, d_yy_lim1 , d_zz_lim1, NN, d_zerofactorr1_sum);
    zero_factor(basename + "y0count10.xyz", NN, d_zerofactorr1_sum);

    xyz_trj_mpcd(basename + "y10mpcdtraj20.xyz", d_xx_lim2, d_yy_lim2 , d_zz_lim2, NN, d_zerofactorr2_sum);
    zero_factor(basename + "y10count20.xyz", NN, d_zerofactorr2_sum);

    xyz_trj_mpcd(basename + "y20mpcdtraj30.xyz", d_xx_lim3, d_yy_lim3 , d_zz_lim3, NN, d_zerofactorr3_sum);
    zero_factor(basename + "y20count30.xyz", NN, d_zerofactorr3_sum); 

    xyz_trj_mpcd(basename + "y-10mpcdtraj0.xyz", d_xx_lim4, d_yy_lim4 , d_zz_lim4, NN, d_zerofactorr4_sum);
    zero_factor(basename + "y-10count0.xyz", NN, d_zerofactorr4_sum);

    xyz_trj_mpcd(basename + "y-20mpcdtraj-10.xyz", d_xx_lim5, d_yy_lim5 , d_zz_lim5, NN, d_zerofactorr5_sum);
    zero_factor(basename + "y-20count-10.xyz", NN, d_zerofactorr5_sum);

    xyz_trj_mpcd(basename + "y-30mpcdtraj-20.xyz", d_xx_lim6, d_yy_lim6 , d_zz_lim6, NN, d_zerofactorr6_sum);
    zero_factor(basename + "y-30count-20.xyz", NN, d_zerofactorr6_sum);

    xyz_trj_mpcd(basename + "_7mpcdtraj___reduced.xyz", d_xx_lim7, d_yy_lim7 , d_zz_lim7, NN, d_zerofactorr7_sum);
    zero_factor(basename + "_7mpcdcount__reduced.xyz", NN, d_zerofactorr7_sum);



    xyz_trj_mpcd(basename + "y0mpcdvel10.xyz", d_vxx_lim1, d_vyy_lim1 , d_vzz_lim1, NN, d_zerofactorr1_sum);
    xyz_trj_mpcd(basename + "y10mpcdvel20.xyz", d_vxx_lim2, d_vyy_lim2 , d_vzz_lim2, NN, d_zerofactorr2_sum);
    xyz_trj_mpcd(basename + "y20mpcdvel30.xyz", d_vxx_lim3, d_vyy_lim3 , d_vzz_lim3, NN, d_zerofactorr3_sum);
    xyz_trj_mpcd(basename + "y-10mpcdvel0.xyz", d_vxx_lim4, d_vyy_lim4 , d_vzz_lim4, NN, d_zerofactorr4_sum);
    xyz_trj_mpcd(basename + "y-20mpcdvel-10.xyz", d_vxx_lim5, d_vyy_lim5 , d_vzz_lim5, NN, d_zerofactorr5_sum);
    xyz_trj_mpcd(basename + "y-30mpcdvel-20.xyz", d_vxx_lim6, d_vyy_lim6 , d_vzz_lim6, NN, d_zerofactorr6_sum);
    xyz_trj_mpcd(basename + "_7mpcdvel___reduced.xyz", d_vxx_lim7, d_vyy_lim7 , d_vzz_lim7, NN, d_zerofactorr7_sum);




}




__global__ void startend_points(double *d_xx, double *d_yy, double *d_zz, double *d_vxx, double *d_vyy, double *d_vzz, double *endp_x, double *endp_y, double *endp_z, int NN, double *scalefactor){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid<NN){

        endp_x[tid]=d_xx[tid]+d_vxx[tid]* *scalefactor;
        endp_y[tid]=d_yy[tid]+d_vyy[tid]* *scalefactor;
        endp_z[tid]=d_zz[tid]+d_vzz[tid]* *scalefactor;

    }

}
//only for mpcd to reduce the data
__host__ void reducevel(std::string basename, double *d_vx,double *d_vy, double *d_vz,
    double *d_vxx, double *d_vyy, double *d_vzz, double *d_x, double *d_y, double *d_z, 
    int N, int skipfactor,int grid_size, double *roundedNumber_vx,double *roundedNumber_vy,double *roundedNumber_vz, int *zerofactor,
    int *zerofactorsumblock, int blockSize_ , int grid_size_,
    double *d_vxx_lim1,  double *d_vyy_lim1, double *d_vzz_lim1, int *zerofactor1,
    double *d_vxx_lim2,  double *d_vyy_lim2, double *d_vzz_lim2, int *zerofactor2,
    double *d_vxx_lim3,  double *d_vyy_lim3, double *d_vzz_lim3, int *zerofactor3,
    double *d_vxx_lim4,  double *d_vyy_lim4, double *d_vzz_lim4, int *zerofactor4,
    double *d_vxx_lim5,  double *d_vyy_lim5, double *d_vzz_lim5, int *zerofactor5,
    double *d_vxx_lim6,  double *d_vyy_lim6, double *d_vzz_lim6, int *zerofactor6,
    double *d_vxx_lim7,  double *d_vyy_lim7, double *d_vzz_lim7, int *zerofactor7,
    int *zerofactorsumblock1,int *zerofactorsumblock2,int *zerofactorsumblock3,int *zerofactorsumblock4,
    int *zerofactorsumblock5, int *zerofactorsumblock6, int *zerofactorsumblock7){


    int NN = int(N/skipfactor);
    int shared_mem_size_ = 3 * blockSize_ * sizeof(int);
    int block_sum_zerofactor[grid_size_];

    /*int block_sum_zerofactor1[grid_size_];
    int block_sum_zerofactor2[grid_size_];
    int block_sum_zerofactor3[grid_size_];
    int block_sum_zerofactor4[grid_size_];
    int block_sum_zerofactor5[grid_size_];
    int block_sum_zerofactor6[grid_size_];
    int block_sum_zerofactor7[grid_size_];*/
    
    /*initializeArrayint<<<grid_size,blockSize>>>(zerofactor1, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor2, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor3, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor4, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor5, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor6, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArrayint<<<grid_size,blockSize>>>(zerofactor7, NN, 0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/


    /*reduceVel<<<grid_size, blockSize>>>(d_vx, d_vy, d_vz, d_vxx, d_vyy, d_vzz, d_x, d_y, d_z, N, skipfactor, roundedNumber_vx, roundedNumber_vy, roundedNumber_vz, zerofactor);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    /*initializeArray<<<grid_size, blockSize>>>(d_vxx_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vxx_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vyy_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim1, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim2, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim3, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim4, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim5, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim6, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    initializeArray<<<grid_size, blockSize>>>(d_vzz_lim7, NN, 1000.000000);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/


    /*velocity_limiting_kernel<<<grid_size, blockSize>>>(d_vxx, d_vyy, d_vzz,d_x, d_y, d_z,
        d_vxx_lim1, d_vyy_lim1, d_vzz_lim1, zerofactor1,
        d_vxx_lim2, d_vyy_lim2, d_vzz_lim2,  zerofactor2,
        d_vxx_lim3, d_vyy_lim3, d_vzz_lim3, zerofactor3,
        d_vxx_lim4, d_vyy_lim4, d_vzz_lim4,  zerofactor4,
        d_vxx_lim5, d_vyy_lim5, d_vzz_lim5, zerofactor5,
        d_vxx_lim6, d_vyy_lim6, d_vzz_lim6,  zerofactor6,
        d_vxx_lim7, d_vyy_lim7, d_vzz_lim7,  zerofactor7, NN);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    //in this line we should sum over all zerofactor elements to calculate zerofactor_sum
    /*intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor, zerofactorsumblock, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    /*intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor1, zerofactorsumblock1, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor2, zerofactorsumblock2, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor3, zerofactorsumblock3, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor4, zerofactorsumblock4, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor5, zerofactorsumblock5, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor6, zerofactorsumblock6, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    intreduceKernel_<<<grid_size_,blockSize_,shared_mem_size_>>>(zerofactor7, zerofactorsumblock7, NN);
     gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() ); */

    //cudaMemcpy(block_sum_zerofactor, zerofactorsumblock, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);

    /*cudaMemcpy(block_sum_zerofactor1, zerofactorsumblock1, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor2, zerofactorsumblock2, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor3, zerofactorsumblock3, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor4, zerofactorsumblock4, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor5, zerofactorsumblock5, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor6, zerofactorsumblock6, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_sum_zerofactor7, zerofactorsumblock7, grid_size_*sizeof(int), cudaMemcpyDeviceToHost);*/

    //int d_zerofactor_sum = 0;

    /*int d_zerofactor1_sum = 0;
    int d_zerofactor2_sum = 0;
    int d_zerofactor3_sum = 0;
    int d_zerofactor4_sum = 0;
    int d_zerofactor5_sum = 0;
    int d_zerofactor6_sum = 0;
    int d_zerofactor7_sum = 0;*/

    /*for (int j = 0; j < grid_size; j++)
        {
            d_zerofactor_sum += block_sum_zerofactor[j];

            d_zerofactor1_sum += block_sum_zerofactor1[j];
            d_zerofactor2_sum += block_sum_zerofactor2[j];
            d_zerofactor3_sum += block_sum_zerofactor3[j];
            d_zerofactor4_sum += block_sum_zerofactor4[j];
            d_zerofactor5_sum += block_sum_zerofactor5[j];
            d_zerofactor6_sum += block_sum_zerofactor6[j];
            d_zerofactor7_sum += block_sum_zerofactor7[j];







           
        }*/

    //printf("zerofactor = %i", d_zerofactor_sum);

    /*d_zerofactor1_sum = NN-d_zerofactor1_sum;
    d_zerofactor2_sum = NN-d_zerofactor2_sum;
    d_zerofactor3_sum = NN-d_zerofactor3_sum;
    d_zerofactor4_sum = NN-d_zerofactor4_sum;
    d_zerofactor5_sum = NN-d_zerofactor5_sum;
    d_zerofactor6_sum = NN-d_zerofactor6_sum;
    d_zerofactor7_sum = NN-d_zerofactor7_sum;*/


    //xyz_trj_mpcd(basename + "_mpcdvel___reduced.xyz", d_vxx, d_vyy , d_vzz, NN, d_zerofactor_sum);

    /*xyz_trj_mpcd(basename + "_1mpcdvel___reduced.xyz", d_vxx_lim1, d_vyy_lim1 , d_vzz_lim1, NN, d_zerofactor1_sum);
    xyz_trj_mpcd(basename + "_2mpcdvel___reduced.xyz", d_vxx_lim2, d_vyy_lim2 , d_vzz_lim2, NN, d_zerofactor2_sum);
    xyz_trj_mpcd(basename + "_3mpcdvel___reduced.xyz", d_vxx_lim3, d_vyy_lim3 , d_vzz_lim3, NN, d_zerofactor3_sum);
    xyz_trj_mpcd(basename + "_4mpcdvel___reduced.xyz", d_vxx_lim4, d_vyy_lim4 , d_vzz_lim4, NN, d_zerofactor4_sum);
    xyz_trj_mpcd(basename + "_5mpcdvel___reduced.xyz", d_vxx_lim5, d_vyy_lim5 , d_vzz_lim5, NN, d_zerofactor5_sum);
    xyz_trj_mpcd(basename + "_6mpcdvel___reduced.xyz", d_vxx_lim6, d_vyy_lim6 , d_vzz_lim6, NN, d_zerofactor6_sum);
    xyz_trj_mpcd(basename + "_7mpcdvel___reduced.xyz", d_vxx_lim7, d_vyy_lim7 , d_vzz_lim7, NN, d_zerofactor7_sum);*/



}
 
__host__ void xyz_veltraj_both(std::string basename, double *d_xx, double *d_yy, double *d_zz, double *d_vxx, double *d_vyy, double *d_vzz, int NN, double *endp_x, double *endp_y, double *endp_z, double *scalefactor, int grid_size){

    xyz_trjvel(basename + "_mpcdtrajvel_both.xyz", d_xx, d_yy , d_zz,d_vxx, d_vyy, d_vzz, NN);
    double *scale__factor;
    cudaMalloc((void**)&scale__factor, sizeof(double));
    cudaMemcpy(scale__factor, scalefactor, sizeof(double) , cudaMemcpyHostToDevice);
    startend_points<<<grid_size, blockSize>>>(d_xx, d_yy, d_zz, d_vxx, d_vyy, d_vzz, endp_x, endp_y, endp_z, NN, scale__factor);
    xyz_trjvel(basename + "_startend.xyz", d_xx, d_yy , d_zz,endp_x ,endp_y , endp_z, NN);
    xyz_trj(basename + "_start.xyz", d_xx, d_yy , d_zz, NN);
    xyz_trj(basename + "_end.xyz", endp_x, endp_y , endp_z, NN);
}


