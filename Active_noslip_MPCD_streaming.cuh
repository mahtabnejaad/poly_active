
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
__global__ void mpcd_distance_from_walls(double *x, double *y, double *z, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *L, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        if (wall_sign_x[tid] == 1)   x_wall_dist[tid] = L[0]/2-(x[tid]);
        else if (wall_sign_x[tid] == -1)  x_wall_dist[tid] = -(L[0]/2+(x[tid]));
        else if(wall_sign_x[tid] == 0)  x_wall_dist[tid] = L[0]/2 -(x[tid]);//we can change it as we like . it doesn't matter.


        if (wall_sign_y[tid] == 1)   y_wall_dist[tid] = L[1]/2-(y[tid]);
        else if (wall_sign_y[tid] == -1)  y_wall_dist[tid] = -(L[1]/2+(y[tid]));
        else if(wall_sign_y[tid] == 0)  y_wall_dist[tid] = L[1]/2 -(y[tid]);//we can change it as we like . it doesn't matter.


        if (wall_sign_z[tid] == 1)   z_wall_dist[tid] = L[2]/2-(z[tid]);
        else if (wall_sign_z[tid] == -1)  z_wall_dist[tid] = -(L[2]/2+(z[tid]));
        else if(wall_sign_z[tid] == 0)  z_wall_dist[tid] = L[2]/2 -(z[tid]);//we can change it as we like . it doesn't matter.



        //printf("***dist_x[%i]=%f, dist_y[%i]=%f, dist_z[%i]=%f\n", tid, x_wall_dist[tid], tid, y_wall_dist[tid], tid, z_wall_dist[tid]);
        int idxx;
        idxx = (int(x[tid] + L[0] / 2 + 2) + (L[0] + 4) * int(y[tid] + L[1] / 2 + 2) + (L[0] + 4) * (L[1] + 4) * int(z[tid] + L[2] / 2 + 2));
        //printf("index[%i]=%i, x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, idxx, tid, x[tid], tid, y[tid], tid, z[tid]);//checking

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
__global__ void Active_noslip_mpcd_deltaT(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x, double *dt_y, double *dt_z, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf("---fa_x=%f, fa_y=%f, fa_z=%f\n", *fa_x, *fa_y, *fa_z);
    if (tid<N){
        
        

        if(wall_sign_x[tid] == 0 ){

            dt_x[tid] = 10000;//a big number because next step is to consider the minimum of dt .
           
        }
        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
           dt_x[tid] = abs(x_wall_dist[tid]/vx[tid]);

            
        }  

        if(wall_sign_y[tid] == 0 ){

            dt_y[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            
        }
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            dt_y[tid] = abs(y_wall_dist[tid]/vy[tid]);
            
            
        }  

        if(wall_sign_z[tid] == 0 ){

            dt_z[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            
        }
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            dt_z[tid] = abs(z_wall_dist[tid]/vz[tid]);

            
        }  



    }


}


//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_noslip_mpcd_deltaT_opposite(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x_opp, double *dt_y_opp, double *dt_z_opp, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf("---fa_x=%f, fa_y=%f, fa_z=%f\n", *fa_x, *fa_y, *fa_z);
    if (tid<N){
        
        

        if(wall_sign_x[tid] == 0 ){

            dt_x_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
           
        }
        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
           if(vx[tid]>0) dt_x_opp[tid] = abs((x_wall_dist[tid]-L[0])/(-vx[tid]));
           else if(vx[tid]<0) dt_x_opp[tid] = abs((x_wall_dist[tid]+L[0])/(-vx[tid]));

            
        }  

        if(wall_sign_y[tid] == 0 ){

            dt_y_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            
        }
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(vy[tid]>0)  dt_y_opp[tid] = abs((y_wall_dist[tid]-L[1])/(-vy[tid]));
            else if(vy[tid]<0)  dt_y_opp[tid] = abs((y_wall_dist[tid]+L[1])/(-vy[tid]));
            
            
        }  

        if(wall_sign_z[tid] == 0 ){

            dt_z_opp[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            
        }
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(vz[tid]>0)  dt_z_opp[tid] = abs((z_wall_dist[tid]-L[2])/(-vz[tid]));
            else if(vz[tid]<0)  dt_z_opp[tid] = abs((z_wall_dist[tid]+L[2])/(-vz[tid]));

            
        }  



    }


}





//a function to calculate dt1 dt2 and dt3 which are dts calculated with the help of particle's velocities and distances from corresponding walls 
__global__ void Active_CM_noslip_mpcd_deltaT(double *vx, double *vy, double *vz, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *dt_x, double *dt_y, double *dt_z, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid, double *L){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double delta_x;
    double delta_y;
    double delta_z;
    double delta_x_p; double delta_x_n; double delta_y_p; double delta_y_n; double delta_z_p; double delta_z_n;
    //printf("---fa_x=%f, fa_y=%f, fa_z=%f\n", *fa_x, *fa_y, *fa_z);
    if (tid<N){
        
        double mm = (Nmd*mass+mass_fluid*N);

        if(wall_sign_x[tid] == 0 ){

            if(*fa_x/mm  == 0) dt_x[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(*fa_x/mm  > 0.0)  dt_x[tid] = sqrt(2*x_wall_dist[tid]/(*fa_x/mm));
            else if(*fa_x/mm < 0.0)  dt_x[tid] = sqrt(2*(x_wall_dist[tid]-L[0])/(*fa_x/mm));
        }
        else if(wall_sign_x[tid] == 1 || wall_sign_x[tid] == -1){
            
           if(*fa_x == 0.0)   dt_x[tid] = abs(x_wall_dist[tid]/vx[tid]);

            else if (*fa_x != 0.0){

                delta_x = ((vx[tid]*vx[tid])+(2*x_wall_dist[tid]*(*fa_x/mm)));

                if(delta_x >= 0.0){
                        if(vx[tid] > 0.0)         dt_x[tid] = ((-vx[tid] + sqrt(delta_x))/(*fa_x/mm));
                        else if(vx[tid] < 0.0)    dt_x[tid] = ((-vx[tid] - sqrt(delta_x))/(*fa_x/mm));
                        
                } 
                else if (delta_x < 0.0){
                        delta_x_p = ((vx[tid]*vx[tid])+(2*(x_wall_dist[tid]-L[0])*(*fa_x/mm)));
                        delta_x_n = ((vx[tid]*vx[tid])+(2*(x_wall_dist[tid]+L[0])*(*fa_x/mm)));

                        if(vx[tid] > 0.0)        dt_x[tid] = ((-vx[tid] - sqrt(delta_x_p))/(*fa_x/mm));
                        else if(vx[tid] < 0.0)   dt_x[tid] = ((-vx[tid] + sqrt(delta_x_n))/(*fa_x/mm));
            
                    }  
            }  
        }  

        if(wall_sign_y[tid] == 0 ){
            if(*fa_y/mm== 0) dt_y[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(*fa_y/mm > 0.0)  dt_y[tid] = sqrt(2*y_wall_dist[tid]/(*fa_y/mm));
            else if(*fa_y/mm < 0.0)  dt_y[tid] = sqrt(2*(y_wall_dist[tid]-L[1])/(*fa_y/mm));
        }
        else if(wall_sign_y[tid] == 1 || wall_sign_y[tid] == -1){
            
            if(*fa_y/mm == 0.0)   dt_y[tid] = abs(y_wall_dist[tid]/vy[tid]);
            
            else if (*fa_y/mm != 0.0){

                delta_y = (vy[tid]*vy[tid])+(2*y_wall_dist[tid]*(*fa_y/mm));

                if (delta_y >= 0){

                    if(vy[tid] > 0.0)              dt_y[tid] = ((-vy[tid] + sqrt(delta_y))/(*fa_y/mm));
                    else if (vy[tid] < 0.0)        dt_y[tid] = ((-vy[tid] - sqrt(delta_y))/(*fa_y/mm));
                }
                else if(delta_y < 0){

                    delta_y_p = ((vy[tid]*vy[tid])+(2*(y_wall_dist[tid]-L[1])*(*fa_y/mm)));
                    delta_y_n = ((vy[tid]*vy[tid])+(2*(y_wall_dist[tid]+L[1])*(*fa_y/mm)));

                    if(vy[tid] > 0.0)        dt_y[tid] = ((-vy[tid] - sqrt(delta_y_p))/(*fa_y/mm));
                    else if(vy[tid] < 0.0)   dt_y[tid] = ((-vy[tid] + sqrt(delta_y_n))/(*fa_y/mm));

                }        
            }

        }  

        if(wall_sign_z[tid] == 0 ){

            if(*fa_z/mm == 0)        dt_z[tid] = 10000;//a big number because next step is to consider the minimum of dt .
            else if(*fa_z/mm > 0.0)  dt_z[tid] = sqrt(2*z_wall_dist[tid]/(*fa_z/mm));
            else if(*fa_z/mm < 0.0)  dt_z[tid] = sqrt(2*(z_wall_dist[tid]-L[2])/(*fa_z/mm));
        }
        else if(wall_sign_z[tid] == 1 || wall_sign_z[tid] == -1){
            
            if(*fa_z/mm == 0.0)   dt_z[tid] = abs(z_wall_dist[tid]/vz[tid]);

            else if (*fa_z/mm != 0.0){

                delta_z = (vz[tid]*vz[tid])+(2*z_wall_dist[tid]*(*fa_z/mm));

                if (delta_z >= 0.0){
                    
                    if(vz[tid] > 0.0)             dt_z[tid] = ((-vz[tid] + sqrt(delta_z))/(*fa_z/mm));
                    else if(vz[tid] < 0.0)        dt_z[tid] = ((-vz[tid] - sqrt(delta_z))/(*fa_z/mm));  
                }

                else if (delta_z < 0.0){
                
                    delta_z_p = ((vz[tid]*vz[tid])+(2*(z_wall_dist[tid]-L[2])*(*fa_z/mm)));
                    delta_z_n = ((vz[tid]*vz[tid])+(2*(z_wall_dist[tid]+L[2])*(*fa_z/mm)));

                    if(vz[tid] > 0.0)        dt_z[tid] = ((-vz[tid] - sqrt(delta_z_p))/(*fa_z/mm));
                    else if(vz[tid] < 0.0)   dt_z[tid] = ((-vz[tid] + sqrt(delta_z_n))/(*fa_z/mm));
                    
                }
                
            }
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
__global__ void Active_CM_mpcd_crossing_location(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *x_o, double *y_o, double *z_o, double *dt_min, double dt, double *L, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid){

    double mm = (Nmd*mass+mass_fluid*N);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        //if( ((x[tid] + dt * vx[tid]) >L[0]/2 || (x[tid] + dt * vx[tid])<-L[0]/2 || (y[tid] + dt * vy[tid])>L[1]/2 || (y[tid] + dt * vy[tid])<-L[1]/2 || (z[tid]+dt * vz[tid])>L[2]/2 || (z[tid] + dt * vz[tid])<-L[2]/2) && dt_min[tid]>0.1) printf("dt_min[%i] = %f\n", tid, dt_min[tid]);
        x_o[tid] = x[tid] + vx[tid]*dt_min[tid] + 0.5 * *fa_x * dt_min[tid] * dt_min[tid] / mm;
        y_o[tid] = y[tid] + vy[tid]*dt_min[tid] + 0.5 * *fa_y * dt_min[tid] * dt_min[tid] / mm;
        z_o[tid] = z[tid] + vz[tid]*dt_min[tid] + 0.5 * *fa_z * dt_min[tid] * dt_min[tid] / mm;
    }

}



__global__ void Active_CM_mpcd_crossing_velocity(double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, int N, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid){

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


__global__ void Active_mpcd_velocityverlet(double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt, int N, double *L, double *T, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        
        //if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2) printf("********** x[%i]=%f, y[%i]=%f, z[%i]=%f\n", tid, x[tid], tid, y[tid], tid, z[tid]);
        x[tid] += dt * vx[tid];
        y[tid] += dt * vy[tid];
        z[tid] += dt * vz[tid];
        

        T[tid]+=dt;
        /*if(tid == 0) {
            printf("T[0] = %f", T[0]);
        }*/
    }
}





__global__ void Active_CM_mpcd_velocityverlet(double *x, double *y, double *z, double *vx, double *vy, double *vz, double dt, int N, double *L, double *T, double *fa_x, double *fa_y, double *fa_z, int Nmd, double mass, double mass_fluid){

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

__global__ void Take_o_to_CM_system(double *x_o, double *y_o, double *z_o, double *vx_o, double *vy_o, double *vz_o, double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        x_o[tid] = x_o[tid] - *Xcm;
        y_o[tid] = y_o[tid] - *Ycm;
        z_o[tid] = z_o[tid] - *Zcm;

        vx_o[tid] = vx_o[tid] - *Vxcm;
        vy_o[tid] = vy_o[tid] - *Vycm;
        vz_o[tid] = vz_o[tid] - *Vzcm;
    }
}

__global__ void Take_o_to_Lab_system(double *x_o, double *y_o, double *z_o, double *vx_o, double *vy_o, double *vz_o, double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        x_o[tid] = x_o[tid] + *Xcm;
        y_o[tid] = y_o[tid] + *Ycm;
        z_o[tid] = z_o[tid] + *Zcm;

        vx_o[tid] = vx_o[tid] + *Vxcm;
        vy_o[tid] = vy_o[tid] + *Vycm;
        vz_o[tid] = vz_o[tid] + *Vzcm;
    }
}

__global__ void Take_o_to_outerCM_system(double *x_o, double *y_o, double *z_o, double *vx_o, double *vy_o, double *vz_o, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){
        x_o[tid] = x_o[tid] - *Xcm_out;
        y_o[tid] = y_o[tid] - *Ycm_out;
        z_o[tid] = z_o[tid] - *Zcm_out;

        vx_o[tid] = vx_o[tid] - *Vxcm_out;
        vy_o[tid] = vy_o[tid] - *Vycm_out;
        vz_o[tid] = vz_o[tid] - *Vzcm_out;
    }
}

__global__ void mpcd_particles_on_crossing_points(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N, int *n_out_flag){



    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        //if(dt_min[tid] < dt){
        if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            x[tid] = x_o[tid];
            y[tid] = y_o[tid];
            z[tid] = z_o[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            vx[tid] = -vx_o[tid];
            vy[tid] = -vy_o[tid];
            vz[tid] = -vz_o[tid];
            n_out_flag[tid] = 1;
        }
        else  n_out_flag[tid]=0;
    }

}

__global__ void mpcd_particles_on_opposite_crossing_points(double *x, double *y, double *z, double *x_o_opp, double *y_o_opp, double *z_o_opp, double *vx, double *vy, double *vz, double *vx_o_opp, double *vy_o_opp, double *vz_o_opp, double *dt_min, double *dt_min_opp, double dt, double *L, int N, int *n_out_flag_opp){



    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        //if(dt_min_opp[tid] < (dt - 2*dt_min[tid]) ){
        if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2){
            //make the position of particle equal to (xo, yo, zo):
            x[tid] = x_o_opp[tid];
            y[tid] = y_o_opp[tid];
            z[tid] = z_o_opp[tid];
            //make the velocity equal to the reverse of the velocity in crossing point.
            vx[tid] = -vx_o_opp[tid];
            vy[tid] = -vy_o_opp[tid];
            vz[tid] = -vz_o_opp[tid];
            n_out_flag_opp[tid] = 1;
        }
        else  n_out_flag_opp[tid]=0;
    }

}

//Active_CM_particle_on_box_and_reverse_velocity_and_md_bounceback_velocityverlet1
__global__ void Active_CM_mpcd_bounceback_velocityverlet1(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *fa_x, double *fa_y, double *fa_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *dt_min, double dt, double *L, int N, double *Xcm, double *Ycm, double *Zcm, int *errorFlag, int *n_out_flag, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<N){

   
    double Mtot = (N * mass_fluid + Nmd * mass); 
    //double QQ2=-((dt - (dt_min[tid]))*(dt - (dt_min[tid])))/(2*(Nmd*mass+mass_fluid*N));
    //double Q2=-(dt - (dt_min[tid]))/(Nmd*mass+mass_fluid*N);

    


    //if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2){
    if((x[tid]+*Xcm)>L[0]/2 || (x[tid]+*Xcm)<-L[0]/2 || (y[tid]+*Ycm)>L[1]/2 || (y[tid]+*Ycm)<-L[1]/2 || (z[tid]+*Zcm)>L[2]/2 || (z[tid]+*Zcm)<-L[2]/2){
        
        if(n_out_flag[tid] == 1){
            
            if (dt_min[tid] > dt) {
                printf("*********************dt_min[%i]=%f\n", tid, dt_min[tid]);
                dt_min[tid]=dt;
                
                //*errorFlag = 1;  // Set the error flag
                //return;  // Early exit
            }
            //let the particle move during dt-dt1 with the reversed velocity:
            x[tid] += (dt - (dt_min[tid])) * vx[tid] + 0.5 * ((dt - (dt_min[tid]))*(dt - (dt_min[tid]))) * (-*Ax_cm);// QQ2 * *fa_x in CM or 0 in lab;
            y[tid] += (dt - (dt_min[tid])) * vy[tid] + 0.5 * ((dt - (dt_min[tid]))*(dt - (dt_min[tid]))) * (-*Ay_cm);// QQ2 * *fa_y in CM or 0 in lab;
            z[tid] += (dt - (dt_min[tid])) * vz[tid] + 0.5 * ((dt - (dt_min[tid]))*(dt - (dt_min[tid]))) * (-*Az_cm);// QQ2 * *fa_z in CM or 0 in lab;
            vx[tid]= vx[tid] +   (dt - (dt_min[tid])) * (-*Ax_cm);// Q2 * *fa_x in CM or 0 in lab;// * 0.5;
            vy[tid]= vy[tid] +   (dt - (dt_min[tid])) * (-*Ay_cm);// Q2 * *fa_y in CM or 0 in lab;// * 0.5;
            vz[tid]= vz[tid] +   (dt - (dt_min[tid])) * (-*Az_cm);// Q2 * *fa_z in CM or 0 in lab;// * 0.5;
        
            if((x_o[tid] + *Xcm )>L[0]/2 || (x_o[tid] + *Xcm)<-L[0]/2 || (y_o[tid] + *Ycm )>L[1]/2 || (y_o[tid] + *Ycm )<-L[1]/2 || (z_o[tid] + *Zcm )>L[2]/2 || (z_o[tid] + *Zcm )<-L[2]/2)  printf("wrong x_o[%i]=%f, y_o[%i]=%f, z_o[%i]=%f\n", tid, (x_o[tid] + *Xcm), tid, (y_o[tid] + *Ycm), tid, (z_o[tid] + *Zcm));

            printf("location after bounceback in lab x[%i]=%f, y[%i]=%f, z[%i]=%f\n ", tid, (x[tid] + *Xcm), tid, (y[tid] + *Ycm), tid, (z[tid] + *Zcm));
            printf("velocity after bounceback in lab vx[%i]=%f, vy[%i]=%f, vz[%i]=%f\n ", tid, (vx[tid] ), tid, (vy[tid] ), tid, (vz[tid] ));
        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        /*if((x[tid] + *Xcm )>L[0]/2 || (x[tid] + *Xcm)<-L[0]/2 || (y[tid] + *Ycm )>L[1]/2 || (y[tid] + *Ycm )<-L[1]/2 || (z[tid] + *Zcm )>L[2]/2 || (z[tid] + *Zcm )<-L[2]/2){



            *errorFlag = 1;  // Set the error flag
            return;  // Early exit
        }*/
        
    }

}

}


__global__ void Active_CM_mpcd_opposite_bounceback_velocityverlet1(double *x, double *y, double *z, double *x_o_opp, double *y_o_opp, double *z_o_opp, double *vx, double *vy, double *vz, double *vx_o_opp, double *vy_o_opp, double *vz_o_opp, double *fa_x, double *fa_y, double *fa_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *dt_min, double *dt_min_opp, double dt, double *L, int N, double *Xcm, double *Ycm, double *Zcm, int *errorFlag, int *n_out_flag_opp, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid<N){

  

    //double QQ3=-((dt - 2*(dt_min[tid])-dt_min_opp[tid])*(dt - 2*(dt_min[tid])-dt_min_opp[tid])/(2*(Nmd*mass+mass_fluid*N)));
    //double Q3=-((dt - 2*(dt_min[tid])-dt_min_opp[tid])/(Nmd*mass+mass_fluid*N));


    //if(x[tid]>L[0]/2 || x[tid]<-L[0]/2 || y[tid]>L[1]/2 || y[tid]<-L[1]/2 || z[tid]>L[2]/2 || z[tid]<-L[2]/2){
    if((x[tid]+*Xcm)>L[0]/2 || (x[tid]+*Xcm)<-L[0]/2 || (y[tid]+*Ycm)>L[1]/2 || (y[tid]+*Ycm)<-L[1]/2 || (z[tid]+*Zcm)>L[2]/2 || (z[tid]+*Zcm)<-L[2]/2){
        
        if(n_out_flag_opp[tid] == 1){
            
            if (dt_min_opp[tid] > (dt - 2* dt_min[tid])) {
                printf("*********************dt_min[%i]=%f\n", tid, dt_min_opp[tid]);
                dt_min_opp[tid]=dt-2*dt_min[tid];
                
                *errorFlag = 1;  // Set the error flag
                return;  // Early exit
            }
            //let the particle move during dt-dt1 with the reversed velocity:
            x[tid] += (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * vx[tid] + 0.5 * ((dt - 2*(dt_min[tid])-dt_min_opp[tid])*(dt - 2*(dt_min[tid])-dt_min_opp[tid])) * (-*Ax_cm);// QQ3 * *fa_x in CM or 0 in lab;
            y[tid] += (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * vy[tid] + 0.5 * ((dt - 2*(dt_min[tid])-dt_min_opp[tid])*(dt - 2*(dt_min[tid])-dt_min_opp[tid])) * (-*Ay_cm);// QQ3 * *fa_y in CM or 0 in lab;
            z[tid] += (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * vz[tid] + 0.5 * ((dt - 2*(dt_min[tid])-dt_min_opp[tid])*(dt - 2*(dt_min[tid])-dt_min_opp[tid])) * (-*Az_cm);// QQ3 * *fa_z in CM or 0 in lab;
            vx[tid]= vx[tid] +   (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * (-*Ax_cm);// Q3 * *fa_x in CM or 0 in lab;// * 0.5;
            vy[tid]= vy[tid] +   (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * (-*Ay_cm);// Q3 * *fa_y in CM or 0 in lab;// * 0.5;
            vz[tid]= vz[tid] +   (dt - 2*(dt_min[tid])-dt_min_opp[tid]) * (-*Az_cm);// Q3 * *fa_z in CM or 0 in lab;// * 0.5;
        
            if((x_o_opp[tid] + *Xcm )>L[0]/2 || (x_o_opp[tid] + *Xcm)<-L[0]/2 || (y_o_opp[tid] + *Ycm )>L[1]/2 || (y_o_opp[tid] + *Ycm )<-L[1]/2 || (z_o_opp[tid] + *Zcm )>L[2]/2 || (z_o_opp[tid] + *Zcm )<-L[2]/2)  printf("wrong x_o_opp[%i]=%f, y_o_opp[%i]=%f, z_o_opp[%i]=%f\n", tid, (x_o_opp[tid] + *Xcm), tid, (y_o_opp[tid] + *Ycm), tid, (z_o_opp[tid] + *Zcm));

            printf("location after the second bounceback in lab x[%i]=%f, y[%i]=%f, z[%i]=%f\n ", tid, (x[tid] + *Xcm), tid, (y[tid] + *Ycm), tid, (z[tid] + *Zcm));
            printf("velocity after the second bounceback in lab vx[%i]=%f, vy[%i]=%f, vz[%i]=%f\n ", tid, (vx[tid] ), tid, (vy[tid] ), tid, (vz[tid] ));
        }
        //printf("** dt_min[%i]=%f, x[%i]=%f, y[%i]=%f, z[%i]=%f \n", tid, dt_min[tid], tid, x[tid], tid, y[tid], tid, z[tid]);//checking
        if((x[tid] + *Xcm )>L[0]/2 || (x[tid] + *Xcm)<-L[0]/2 || (y[tid] + *Ycm )>L[1]/2 || (y[tid] + *Ycm )<-L[1]/2 || (z[tid] + *Zcm )>L[2]/2 || (z[tid] + *Zcm )<-L[2]/2){

            

            *errorFlag = 1;  // Set the error flag
            return;  // Early exit
        }
        
    }

}

}



__global__ void Active_CM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N, double *fa_x, double *fa_y, double *fa_z, double *Xcm, double *Ycm, double *Zcm, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        double QQ2=-((dt - (dt_min[tid]))*(dt - (dt_min[tid])))/(2*(Nmd*mass+mass_fluid*N));
        double Q2=-(dt - (dt_min[tid]))/(Nmd*mass+mass_fluid*N);

        //if((x[tid] + *Xcm )>L[0]/2 || (x[tid] + *Xcm)<-L[0]/2 || (y[tid] + *Ycm )>L[1]/2 || (y[tid] + *Ycm )<-L[1]/2 || (z[tid] + *Zcm )>L[2]/2 || (z[tid] + *Zcm )<-L[2]/2){
          if(dt_min[tid] < dt){
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
        //if((x[tid] + *Xcm )>L[0]/2 || (x[tid] + *Xcm)<-L[0]/2 || (y[tid] + *Ycm )>L[1]/2 || (y[tid] + *Ycm )<-L[1]/2 || (z[tid] + *Zcm )>L[2]/2 || (z[tid] + *Zcm )<-L[2]/2)  printf("*************************goes out %i\n", tid);
        
    }

}



__global__ void Active_outboxCM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet(double *x, double *y, double *z, double *x_o, double *y_o, double *z_o, double *vx, double *vy, double *vz, double *vx_o, double *vy_o, double *vz_o, double *dt_min, double dt, double *L, int N, double *fa_x, double *fa_y, double *fa_z, double *Xcm, double *Ycm, double *Zcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, int Nmd, double mass, double mass_fluid){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid<N){

        double QQ2=-((dt - (dt_min[tid]))*(dt - (dt_min[tid])))/(2*(Nmd*mass+mass_fluid*N));
        double Q2=-(dt - (dt_min[tid]))/(Nmd*mass+mass_fluid*N);

        if((x[tid] + *Xcm + *Xcm_out)>L[0]/2 || (x[tid] + *Xcm + *Xcm_out)<-L[0]/2 || (y[tid] + *Ycm + *Ycm_out)>L[1]/2 || (y[tid] + *Ycm + *Ycm_out)<-L[1]/2 || (z[tid] + *Zcm + *Zcm_out)>L[2]/2 || (z[tid] + *Zcm + *Zcm_out)<-L[2]/2){
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
double *L, int Nmd , double ux, double mass, double mass_fluid, double real_time, int m, int topology, double *dt_x, double *dt_y, double *dt_z, double *dt_min, 
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

    Active_CM_noslip_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, fax, fay, faz, Nmd, mass, mass_fluid, L);
    gpuErrchk( cudaPeekAtLastError() );               
    gpuErrchk( cudaDeviceSynchronize() );

    /*Active_Lab_noslip_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/


    Active_deltaT_min<<<grid_size,blockSize>>>(dt_x, dt_y, dt_z, dt_min, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o, y_o, z_o, dt_min, h_mpcd, L, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    Active_CM_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //CM_outside_particles
    /*outerParticles_CM_system(d_mdX, d_mdY, d_mdZ, d_x, d_y, d_z,  d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, n_outbox_md, n_outbox_mpcd,
    mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, dn_mpcd_tot, dn_md_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, 
    Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz,
    CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, CMsumblock_n_outbox_mpcd, CMsumblock_n_outbox_md, topology, L);
    */
    
    //gotoOUTBOXCMframe  go to out of box cm frame for mpcd particles:
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //go to out of box cm frame for md particles:(should I???)
    gotoOUTBOXCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //take o point (crossing point) components to the outer particles' CM system:
    Take_o_to_outerCM_system<<<grid_size,blockSize>>>(x_o, y_o, z_o, vx_o, vy_o, vz_o, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    //put the particles that had traveled outside of the box , on box boundaries.
    Active_outboxCM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, Nmd, mass, mass_fluid);
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



__host__ void Active_noslip_MPCD_streaming2(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double* d_mdX, double* d_mdY, double* d_mdZ, double* d_mdVx , double* d_mdVy, double* d_mdVz,
double *X_tot, double *Y_tot, double *Z_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot,
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, double h_mpcd, int N, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L, int Nmd , double ux, double mass, double mass_fluid, double real_time, int m, int topology, double *dt_x, double *dt_y, double *dt_z, double *dt_min, 
double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *T, int *n_outbox_mpcd, int *n_outbox_md, int *dn_mpcd_tot, int *dn_md_tot, int *CMsumblock_n_outbox_mpcd, int *CMsumblock_n_outbox_md)

{

    double *fax, *fay, *faz;
    cudaMalloc((void**)&fax, sizeof(double)); cudaMalloc((void**)&fay, sizeof(double)); cudaMalloc((void**)&faz, sizeof(double));
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice); 
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fay, fa_y, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(faz, fa_z, sizeof(double) , cudaMemcpyHostToDevice);


    CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    

    wall_sign<<<grid_size,blockSize>>>(d_vx , d_vy , d_vz, wall_sign_x, wall_sign_y, wall_sign_z, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    distance_from_walls<<<grid_size,blockSize>>>(d_x , d_y , d_z, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_noslip_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, fax, fay, faz, Nmd, mass, mass_fluid, L);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    Active_deltaT_min<<<grid_size,blockSize>>>(dt_x, dt_y, dt_z, dt_min, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o, y_o, z_o, dt_min, h_mpcd, L, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, N, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    //take all MPCD particles to CM reference frame:
    gotoCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Take_o_to_CM_system<<<grid_size,blockSize>>>(x_o, y_o, z_o, vx_o, vy_o, vz_o, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    //gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    /*
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

    //take o point (crossing point) components to the outer particles' CM system:
    Take_o_to_outerCM_system<<<grid_size,blockSize>>>(x_o, y_o, z_o, vx_o, vy_o, vz_o, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/
    
    //put the particles that had traveled outside of the box , on box boundaries.
    /*Active_CM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_CM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Xcm, Ycm, Zcm, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


    /*
    //go back to the old CM frame:
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    */

    //gotoLabFrame for mpcd particles:
    backtoLabframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    //backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


}


__host__ void Active_noslip_MPCD_streaming3(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double* d_mdX, double* d_mdY, double* d_mdZ, double* d_mdVx , double* d_mdVy, double* d_mdVz,
double *X_tot, double *Y_tot, double *Z_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot,
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, double h_mpcd, int N, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z ,double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L, int Nmd , double ux, double mass, double mass_fluid, double real_time, int m, int topology, double *dt_x, double *dt_y, double *dt_z, double *dt_min, 
double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z, double *T, int *n_outbox_mpcd, int *n_outbox_md, int *dn_mpcd_tot, int *dn_md_tot, int *CMsumblock_n_outbox_mpcd, int *CMsumblock_n_outbox_md)

{

    double *fax, *fay, *faz;
    cudaMalloc((void**)&fax, sizeof(double)); cudaMalloc((void**)&fay, sizeof(double)); cudaMalloc((void**)&faz, sizeof(double));
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice); 
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fay, fa_y, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(faz, fa_z, sizeof(double) , cudaMemcpyHostToDevice);


    CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
    Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    

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
    
    
    //take all MPCD particles to CM reference frame:
    gotoCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Take_o_to_CM_system<<<grid_size,blockSize>>>(x_o, y_o, z_o, vx_o, vy_o, vz_o, Xcm, Ycm, Zcm, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    //??with this function call MD particles go to box's center of mass frame:(should I???)
    //gotoCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_CM_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, fax, fay, faz, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    /*
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

    //take o point (crossing point) components to the outer particles' CM system:
    Take_o_to_outerCM_system<<<grid_size,blockSize>>>(x_o, y_o, z_o, vx_o, vy_o, vz_o, Xcm_out, Ycm_out, Zcm_out, Vxcm_out, Vycm_out, Vzcm_out, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/
    
    //put the particles that had traveled outside of the box , on box boundaries.
    /*Active_outboxCM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );*/

    //put the particles that had traveled outside of the box , on box boundaries.
    Active_CM_particle_on_box_and_reverse_velocity_and_mpcd_bounceback_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z, x_o, y_o, z_o, d_vx ,d_vy ,d_vz , vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, fax, fay, faz, Xcm, Ycm, Zcm, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );



    /*
    //go back to the old CM frame:
    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, N, L, n_outbox_mpcd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gobackOUTBOX_OLDCMframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, Xcm_out, Ycm_out, Zcm_out, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Vxcm_out, Vycm_out, Vzcm_out, Nmd, L, n_outbox_md);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    */

    //gotoLabFrame for mpcd particles:
    backtoLabframe<<<grid_size,blockSize>>>(d_x, d_y, d_z, Xcm, Ycm, Zcm, d_vx, d_vy, d_vz, Vxcm, Vycm, Vzcm, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //gotoLabFrame for md particles:
    //backtoLabframe<<<grid_size,blockSize>>>(d_mdX, d_mdY, d_mdZ, Xcm, Ycm, Zcm, d_mdVx, d_mdVy, d_mdVz, Vxcm, Vycm, Vzcm, Nmd);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );


}





__host__ void Active_noslip_MPCD_streaming4(double* d_x, double* d_y , double* d_z, double* d_vx , double* d_vy, double* d_vz, double* d_mdX, double* d_mdY, double* d_mdZ, double* d_mdVx , double* d_mdVy, double* d_mdVz,
double *X_tot, double *Y_tot, double *Z_tot, double *Vx_tot, double *Vy_tot, double *Vz_tot, double *mdX_tot, double *mdY_tot, double *mdZ_tot, double *mdVx_tot, double *mdVy_tot, double *mdVz_tot,
double *CMsumblock_x, double *CMsumblock_y, double *CMsumblock_z, double *CMsumblock_mdx, double *CMsumblock_mdy, double *CMsumblock_mdz,
double *CMsumblock_Vx, double *CMsumblock_Vy, double *CMsumblock_Vz, double *CMsumblock_mdVx, double *CMsumblock_mdVy, double *CMsumblock_mdVz,
double *Xcm, double *Ycm, double *Zcm, double *Vxcm, double *Vycm, double *Vzcm, double *Xcm_out, double *Ycm_out, double *Zcm_out, double *Vxcm_out, double *Vycm_out, double *Vzcm_out, double h_mpcd, int N, int grid_size, int shared_mem_size, int shared_mem_size_, int blockSize_, int grid_size_,
double *fa_x, double *fa_y, double *fa_z, double *fb_x, double *fb_y, double *fb_z, double *Ax_cm, double *Ay_cm, double *Az_cm, double *ex, double *ey, double *ez,double *block_sum_ex, double *block_sum_ey, double *block_sum_ez,
double *L, int Nmd , double ux, double mass, double mass_fluid, double real_time, int m, int topology, double *dt_x, double *dt_y, double *dt_z, double *dt_min, double *dt_x_opp, double *dt_y_opp, double *dt_z_opp, double *dt_min_opp,
double *x_o, double *y_o ,double *z_o, double *vx_o, double *vy_o, double *vz_o, double *x_o_opp, double *y_o_opp, double *z_o_opp, double *vx_o_opp, double *vy_o_opp, double *vz_o_opp, double *x_wall_dist, double *y_wall_dist, double *z_wall_dist, double *wall_sign_x, double *wall_sign_y, double *wall_sign_z,
double *T, int *n_outbox_mpcd, int *n_outbox_md, int *dn_mpcd_tot, int *dn_md_tot, int *CMsumblock_n_outbox_mpcd, int *CMsumblock_n_outbox_md, int *hostErrorFlag, int *hostErrorFlag_opp, int *n_out_flag, int *n_out_flag_opp, double *d_zero)

{

    double *fax, *fay, *faz;
    cudaMalloc((void**)&fax, sizeof(double)); cudaMalloc((void**)&fay, sizeof(double)); cudaMalloc((void**)&faz, sizeof(double));
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice); 
    cudaMemcpy(fax, fa_x, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(fay, fa_y, sizeof(double) , cudaMemcpyHostToDevice);  cudaMemcpy(faz, fa_z, sizeof(double) , cudaMemcpyHostToDevice);


    //CM_system(d_mdX, d_mdY, d_mdZ,d_x, d_y, d_z, d_mdVx, d_mdVy, d_mdVz, d_vx, d_vy, d_vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, density, 1,
    //Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);

    

    wall_sign<<<grid_size,blockSize>>>(d_vx , d_vy , d_vz, wall_sign_x, wall_sign_y, wall_sign_z, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //calculate particle's distance from walls if the particle is inside the box:
    mpcd_distance_from_walls<<<grid_size,blockSize>>>(d_x , d_y , d_z, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Active_noslip_mpcd_deltaT<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x, dt_y, dt_z, N, fax, fay, faz, Nmd, mass, mass_fluid, L);
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
    

    Active_noslip_mpcd_deltaT_opposite<<<grid_size,blockSize>>>(d_vx, d_vy, d_vz, wall_sign_x, wall_sign_y, wall_sign_z, x_wall_dist, y_wall_dist, z_wall_dist, dt_x_opp, dt_y_opp, dt_z_opp, N, fax, fay, faz, Nmd, mass, mass_fluid, L);
    gpuErrchk( cudaPeekAtLastError() );                
    gpuErrchk( cudaDeviceSynchronize() );

    deltaT_min<<<grid_size,blockSize>>>(dt_x_opp, dt_y_opp, dt_z_opp, dt_min_opp, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    mpcd_opposite_crossing_location<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, x_o_opp, y_o_opp, z_o_opp, dt_min_opp, h_mpcd, L, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    mpcd_opposite_crossing_velocity<<<grid_size,blockSize>>>(d_vx ,d_vy ,d_vz , vx_o_opp, vy_o_opp, vz_o_opp, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    Active_mpcd_velocityverlet<<<grid_size,blockSize>>>(d_x , d_y , d_z , d_vx , d_vy , d_vz, h_mpcd, N, L, T, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    
    
    //we put the particles that had gone outside the box, on the box's boundaries and set its velocity equal to the negative of the crossing velocity in Lab system.
    mpcd_particles_on_crossing_points<<<grid_size,blockSize>>>(d_x, d_y, d_z, x_o, y_o, z_o, d_vx, d_vy, d_vz, vx_o, vy_o, vz_o, dt_min, h_mpcd, L, N, n_out_flag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
  
    //CM_system : we call CM system to calculate the new CM after streaming. we should check if this is alright or not. we could also use the former CM system for bounceback part.
    //CM_system(mdX, mdY, mdZ, x, y, z, mdvx, mdvy, mdvz, vx, vy, vz, Nmd, N, mdX_tot, mdY_tot, mdZ_tot, X_tot, Y_tot, Z_tot, mdVx_tot, mdVy_tot, mdVz_tot, Vx_tot, Vy_tot, Vz_tot, grid_size, shared_mem_size, shared_mem_size_, blockSize_, grid_size_, mass, mass_fluid,
    //Xcm, Ycm, Zcm, CMsumblock_x, CMsumblock_y, CMsumblock_z, CMsumblock_mdx, CMsumblock_mdy, CMsumblock_mdz, Vxcm, Vycm, Vzcm, CMsumblock_Vx, CMsumblock_Vy, CMsumblock_Vz, CMsumblock_mdVx, CMsumblock_mdVy, CMsumblock_mdVz, topology);


    

    int *d_errorFlag_mpcd;
    *hostErrorFlag = 0;
    cudaMalloc(&d_errorFlag_mpcd, sizeof(int));
    cudaMemcpy(d_errorFlag_mpcd, hostErrorFlag, sizeof(int), cudaMemcpyHostToDevice);


    double *Axcm, *Aycm, *Azcm;
    cudaMalloc(&Axcm, sizeof(double));
    cudaMalloc(&Aycm, sizeof(double));
    cudaMalloc(&Azcm, sizeof(double));
    cudaMemcpy(Axcm, Ax_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Aycm, Ay_cm, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Azcm, Az_cm, sizeof(double), cudaMemcpyHostToDevice);

    double *zeroo;
    cudaMalloc(&zeroo, sizeof(double));
    *d_zero = 0.0;
    cudaMemcpy(zeroo, d_zero, sizeof(double), cudaMemcpyHostToDevice);

    //after putting the particles that had traveled outside of the box on its boundaries, we let them stream in the opposite direction for the time they had spent outside the box. 
    Active_CM_mpcd_bounceback_velocityverlet1<<<grid_size,blockSize>>>(d_x , d_y, d_z, x_o, y_o, z_o, d_vx, d_vy, d_vz, vx_o, vy_o, vz_o, fax, fay, faz, zeroo, zeroo, zeroo, dt_min, h_mpcd, L, N, zeroo, zeroo, zeroo, d_errorFlag_mpcd, n_out_flag, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Check for kernel errors and sync
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_errorFlag_mpcd);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }

    // Check the error flag
    cudaMemcpy(hostErrorFlag, d_errorFlag_mpcd, sizeof(int), cudaMemcpyDeviceToHost);
    if (*hostErrorFlag) {
        printf("Error condition met in kernel (first bounceback). Exiting.\n");
        // Clean up and exit
        cudaFree(d_errorFlag_mpcd);
        *hostErrorFlag = -1;  // Set error flag
        return;
    }


    int *d_errorFlag_mpcd_opp;
    *hostErrorFlag_opp = 0;
    cudaMalloc(&d_errorFlag_mpcd_opp, sizeof(int));
    cudaMemcpy(d_errorFlag_mpcd_opp, hostErrorFlag_opp, sizeof(int), cudaMemcpyHostToDevice);


    
    mpcd_particles_on_opposite_crossing_points<<<grid_size,blockSize>>>(d_x, d_y, d_z, x_o_opp, y_o_opp, z_o_opp, d_vx, d_vy, d_vz, vx_o_opp, vy_o_opp, vz_o_opp, dt_min, dt_min_opp, h_mpcd, L, N, n_out_flag_opp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double *zerooo;
    cudaMalloc(&zerooo, sizeof(double));
    *d_zero = 0.0;
    cudaMemcpy(zerooo, d_zero, sizeof(double), cudaMemcpyHostToDevice);

    Active_CM_mpcd_opposite_bounceback_velocityverlet1<<<grid_size,blockSize>>>(d_x , d_y, d_z, x_o_opp, y_o_opp, z_o_opp, d_vx, d_vy, d_vz, vx_o_opp, vy_o_opp, vz_o_opp, fax, fay, faz, zerooo, zerooo, zerooo, dt_min, dt_min_opp, h_mpcd, L, N, zerooo, zerooo, zerooo, d_errorFlag_mpcd_opp, n_out_flag_opp, Nmd, mass, mass_fluid);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Check for kernel errors and sync
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_errorFlag_mpcd_opp);
        *hostErrorFlag_opp = -1;  // Set error flag
        return;
    }

    // Check the error flag
    cudaMemcpy(hostErrorFlag_opp, d_errorFlag_mpcd_opp, sizeof(int), cudaMemcpyDeviceToHost);
    if (*hostErrorFlag_opp) {
        printf("Error condition met in kernel (second bounceback). Exiting.\n");
        // Clean up and exit
        cudaFree(d_errorFlag_mpcd_opp);
        *hostErrorFlag_opp = -1;  // Set error flag
        return;
    }


    
    

    //cudaFree(d_errorFlag_mpcd);
    cudaFree(Axcm); cudaFree(Aycm); cudaFree(Azcm);
    cudaFree(zeroo);
    cudaFree(zerooo);

    
   




}

