// Rendering:
// 1) set the lens and precalculate constants - CPU and update to GPU
// 2) paint the screen white and reset the depth feild - Screen and Depth on GPU, done on GPU
// 3) for each triangle paint onto a pixel array - Done on GPU using GPU triangle data going to GPU 
// 4) copy that pixel array to the buffer frame
// 5) push the buffer array to the window
#include "gpuController.h"


void create_render_data(struct cpu_data **cpu_data, struct gpu_data **h_gpu_data, struct gpu_data **d_gpu_data, SDL_Surface *image, int nTrigs, const float *trigs, const Uint8 *clrs, float *v, float *o){
       // Holds all relevent dat about the objects, render constants, and imagescreen info
       //     Used to keep track of these things
       *cpu_data = (struct cpu_data *)malloc(sizeof(struct cpu_data));
       init_cpu_dat(*cpu_data, image, nTrigs, trigs, clrs, v, o);

       // Initalize relevent data onto the GPU -> stays for the duration of the program
       //                                      -> some parts can be eddited as we go
       //                                      -> Creates a memory space on the GPU
       // Allocate the pointer structure on the GPU and CPU -> list of pointers to GPU and other imediate data
       cudaMalloc((void **)d_gpu_data, sizeof(struct gpu_data));
       *h_gpu_data = (struct gpu_data *)malloc(sizeof(struct gpu_data));
       init_gpu_dat(*d_gpu_data, *h_gpu_data, *cpu_data);
}

void init_cpu_dat(struct cpu_data *dat, SDL_Surface *image, int nTrigs, const float *trigs, const Uint8 *clrs, float *v, float *o){
       // Copy over the amount of trigs, and allocate space for personal use
       dat->nTrigs = nTrigs;
       dat->trigs = (float *)malloc(sizeof(float)*9*nTrigs);
       dat->colors = (Uint8 *)malloc(nTrigs*3);
       memcpy(dat->trigs, trigs, sizeof(float)*9*nTrigs);
       memcpy(dat->colors, clrs, 3*nTrigs);

       // Extract data about screen size and pixel organization, also dest pixel location
       dat->pixels_arr = (Uint8 *)image->pixels;
       dat->BPP = image->format->BytesPerPixel;
       dat->pitch = image->pitch;
       dat->w = image->w;
       dat->h = image->h;
       dat->pix_arr_size = (dat->h)*(dat->pitch);
       dat->depth_arr_size = (dat->h)*(dat->w)*sizeof(Uint32);

       // Allocate space for all equation constants -> keeping track of where we are and info to make projection easy
       dat->v = (float *)malloc(sizeof(float) * 15);
       dat->hx = dat->v + 3;
       dat->hy = dat->hx + 3;
       dat->c1 = dat->hy + 3;
       dat->mag = dat->c1 + 1;
       dat->offset = dat->mag + 1;
       dat->a = dat->offset + 3;


       // Update the equation constants based on the inital position
       update_lens(dat, v, o);
}

void update_lens(struct cpu_data *dat, float *v, float *o){
       cpyVec(v, dat->v);
       float temp[3] = {0,0,1};
       cpyVec(o, dat->offset);

       cross(dat->v, temp, dat->hx);
       normalize(dat->hx);
       //constMult(1, dat->hx, dat->hx)

       cross(dat->hx, dat->v, dat->hy);
       normalize(dat->hy);
       //constMult(1, dat->hy, dat->hy)

       *(dat->a) = 0.9;
       *(dat->mag) = vecMag(v);
       *(dat->mag) = (*(dat->mag)) * (*(dat->mag));
       *(dat->c1) = (1-*(dat->a)) * *(dat->mag);
       *(dat->c1) = *(dat->c1)/(2*vecMag(dat->hx));
}

void kill_cpu_data(struct cpu_data *dat){
       free(dat->trigs);
       free(dat->colors);
       free(dat->v);
       free(dat);
}

void init_gpu_dat(struct gpu_data *d_dat, struct gpu_data *h_dat, struct cpu_data *cpu_dat){
       
       // Fill all direct values
       struct gpu_data *temp = h_dat;
       temp->d_nTrigs = cpu_dat->nTrigs;
       temp->d_BPP = cpu_dat->BPP;
       temp->d_pitch = cpu_dat->pitch;
       temp->d_w = cpu_dat->w;
       temp->d_h = cpu_dat->h;

       // Allocate space for the pixel array
       int size = (cpu_dat->pitch)*(cpu_dat->h)*sizeof(Uint8);
       cudaMalloc((void **)&(temp->d_pixels_arr), size);

       // Allocate space for the trig color data
       size = 3*cpu_dat->nTrigs;
       cudaMalloc((void **)&(temp->d_colors), size);
       
       // Allocate space for depthscreen
       cudaMalloc((void **)&(temp->d_depthScreen), temp->d_w*temp->d_h*sizeof(Uint32));

       // Allocate space for the rest of the floats
       //       = trigs and cords   constants
       size = (cpu_dat->nTrigs * 18 + 11 + 3)*sizeof(float);
       cudaMalloc((void **)&(temp->d_trigs), size);

       // Set the rest of the pointers
       temp->d_cords_arr    = temp->d_trigs + cpu_dat->nTrigs * 9;
       temp->d_v            = temp->d_cords_arr + cpu_dat->nTrigs * 9;
       temp->d_hx           = temp->d_v + 3;
       temp->d_hy           = temp->d_hx + 3;
       temp->d_c1           = temp->d_hy + 3;
       temp->d_mag          = temp->d_c1 + 1;
       temp->d_offset       = temp->d_mag + 1;

       // copy over the triangles and the pointer list to the device
       cudaMemcpy(temp->d_trigs, cpu_dat->trigs, sizeof(float) * cpu_dat->nTrigs * 9, cudaMemcpyHostToDevice);
       cudaMemcpy((void *)d_dat, (void *)temp, sizeof(struct gpu_data), cudaMemcpyHostToDevice);
       cudaMemcpy(temp->d_colors, cpu_dat->colors, 3*cpu_dat->nTrigs, cudaMemcpyHostToDevice);
}

void kill_gpu_data(struct gpu_data *h_gDat){
       cudaFree(h_gDat->d_pixels_arr);
       cudaFree(h_gDat->d_depthScreen);
       cudaFree(h_gDat->d_colors);
       cudaFree(h_gDat->d_trigs);
       free(h_gDat);
}

void update_GPU_lens(struct gpu_data *h_dat, struct cpu_data *cdat){
       cudaMemcpy(h_dat->d_v, cdat->v, 14*sizeof(float), cudaMemcpyHostToDevice);
}

__device__ float dot(float *v, float *w, float *o){
       return (*v - *o)*(*w) + (*(v + 1) - *(o + 1))*(*(w + 1)) + (*(v + 2) - *(o + 2))*(*(w + 2));
}

__device__ void crossG(float *v, float *w, float *ans){
       ans[0] = v[1] * w[2] - w[1] * v[2];
       ans[1] = v[2] * w[0] - v[0] * w[2];
       ans[2] = v[0] * w[1] - w[0] * v[1];
}

__device__ void subG(float *v, float *w, float *ans){
       for(int i = 0; i < 3; i++){
              ans[i] = v[i] - w[i];
       }
}

__device__ float dist(float *w, float *v, float * o){
       float a = w[0] - o[0] - v[0];
       float b = w[1] - o[1] - v[1];
       float c = w[2] - o[2] - v[2];
       return sqrt(a*a + b*b + c*c);
}

__global__ void cordify(struct gpu_data *dat, int num_floats){ // Run for each vector j = float j*3 = i{
       
       // i is starting index of a vector
       int i = 3*(threadIdx.x + blockIdx.x * blockDim.x); //0, 3, 6

       float topx = *(float *)(dat->d_c1);
       float topy = topx;
       float bot =  *(float *)(dat->d_mag);
       float mag;

       

       // float *w = dat->d_trigs + i*3
       if(i < num_floats){
              // 1) calculate the x component

              topx = dot(dat->d_trigs + i, dat->d_hx, dat->d_offset) * topx;
              topy = dot(dat->d_trigs + i, dat->d_hy, dat->d_offset) * topy;
              bot = bot - dot(dat->d_trigs + i, dat->d_v, dat->d_offset);
              mag = dist(dat->d_trigs + i, dat->d_v, dat->d_offset);

              if(bot <= 0.1* *(float *)(dat->d_mag))
              { // behind
                     *(float *)(dat->d_cords_arr + i) = -1.0f;
                     *(float *)(dat->d_cords_arr + i + 1) = -1.0f;
                     *(float *)(dat->d_cords_arr + i + 2) = -1.0f;
              }
              else
              {

                     *(float *)(dat->d_cords_arr + i) = (topx/bot + 0.5)*dat->d_w;
                     *(float *)(dat->d_cords_arr + i + 1) = (topy/bot + 0.5)*dat->d_h;
                     *(float *)(dat->d_cords_arr + i + 2) = mag;
              }

       }
}

__device__ float max(float a, float b, float c, float d){
       return (((a > b) ? a:b) > ((c > d) ? c:d) ? ((a > b) ? a:b):((c > d) ? c:d));
}

__device__ float min(float a, float b, float c, float d){
       return (((a < b) ? a:b) < ((c < d) ? c:d) ? ((a < b) ? a:b):((c < d) ? c:d));
}


// Want to paint the area of the triangle given the cords in dat.
// First load in the triangle cordanates
// make a vector formula for each line joining the points
// P1P2, P2P3, P3P1
// Start at the minimium of y values -> to the maximum y value of the triangle
// for a given y0, find the range of x's you need to paint
// at x, y check the depth of the point and check if the points in range of the screen and then paint it

// Form is    t = a*y_0 + b
//            x = c*t   + d
// Side 0: P0->P1: v = P0 + t*(P1 - P0) --> (y_0 - P0_y)/(P1_y - P0_y) = t, x = P0_x + t*(P1_x - P0_x)
// Side 1: P1->P2:
 // Side 3: P2->P0
 
__global__ void draw(struct gpu_data *dat, int num_floats){
       // Run for each triangle

       int i = 9*(threadIdx.x + blockDim.x*blockIdx.x);



       if(i < num_floats){

              float points[3][3];
              float yMin, yMax, xMin, xMax;
              float as[3];
              float bs[3];
              float cs[3];
              float ds[3];
              float t;
              float xT;
              int Ix;
              int Iy;

              // Copy over the three points of the triangle into points from the coordanate data computed
              for(int j = 0; j < 3; j ++){
                     points[j][0] = dat->d_cords_arr[i + j*3];
                     points[j][1] = dat->d_cords_arr[i + j*3 + 1];
                     points[j][2] = dat->d_cords_arr[i + j*3 + 2];
              }
              if(points[0][2] != -1 && points[1][2] != -1 && points[2][2] != -1){

                     // Assign the begining and end of y values to traverse
                     yMin = min(points[0][1], points[1][1], points[2][1], 0);
                     yMax = max(points[0][1], points[1][1], points[2][1], dat->d_h);

                     // Determine the coeffcients to the Lines between the points to find x(y)
                     for(int j = 0; j < 3; j ++){
                            if(points[(j+1)%3][1] == points[j][1]){
                                   as[j] = 0;
                                   bs[j] = -1;
                            }
                            else{
                                   as[j] = 1/(points[(j+1)%3][1] - points[j][1]);
                                   bs[j] = -1 * points[j][1] * as[j];
                            }
                            cs[j] = points[(j+1)%3][0] - points[j][0];
                            ds[j] = points[j][0];
                     }
                     // What is the depth given the x, y cords -> will be in form az + bx + cy = d or a(z-z0) + b(x-x0) + x(y-y0) = 0
                     // Solve for z: z = d - bx - cy

                     // find depth array
                     float depth;
                     float c = points[0][1]*(points[2][0] - points[1][0]) - points[1][1]*(points[2][0] - points[0][0]) + points[2][1]*(points[1][0] - points[0][0]);
                     float b = points[2][0] - points[0][0];
                     float a;
                     if(c != 0 && b != 0){
                            c = (points[0][2]*(points[2][0] - points[1][0]) - points[1][2]*(points[2][0] - points[0][0]) + points[2][2]*(points[1][0] - points[0][0]))/c;
                            b = ((points[2][2] - points[0][2]) - c*(points[2][1] - points[0][1]))/b;
                            a = points[0][2] - b*points[0][0] - c*points[0][1];
                     }
                     else{
                            c = 0;
                            b = 0;
                            a = min(points[0][2], points[1][2], points[2][2], 10);// (points[0][2] + points[1][2] + points[2][2])/3;
                     }

                     // Traverse the pixels within the triangle
                     for(float y = yMin; y <= yMax; y+=0.9){
                            xMin =  99999999;
                            xMax = -99999999;
                            
                            // find the range of x values to use given the equations of the lines
                            for(int j = 0; j < 3; j++)
                            {
                                   t = as[j]*y + bs[j];
                                   xT = cs[j]*t + ds[j];
                                   if(t <= 1 && t >= 0)
                                   {
                                          if(xT < xMin){xMin = xT;}
                                          if(xT > xMax){xMax = xT;}
                                   }
                            }
                            xMin = (xMin < 0) ? 0: xMin;
                            xMax = (xMax > dat->d_w) ? dat->d_w:xMax;
                            
                            for(float x = xMin; x <= xMax; x+=0.9){
                                   Iy = dat->d_h - (int)y;
                                   Ix = (int)x;
                                   depth = (a + b*x + c*y)*1000;
                                   // Make sure that the pixel is within range of the screen and compare the depth.
                                   if(Iy >= 0 && Iy <= dat->d_h && Ix >= 0 && Ix <= dat->d_w){
                                          atomicMin(dat->d_depthScreen + Iy*dat->d_w + Ix, (int)depth);
                                          
                                          if(dat->d_depthScreen[Iy*dat->d_w + Ix] == (int)depth){
                                                 // Paint the pixel and set new depth
                                                 //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                                 dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP] = dat->d_colors[(int)(i/3)];
                                                 dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP + 1] = dat->d_colors[(int)(i/3) + 1];
                                                 dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP + 2] = dat->d_colors[(int)(i/3) + 2];

                                          }
                                   }
                            }
                            
                     }
              }

       }

}
void print_cords(int nTrigs, float *trigs){
       printf("\nPrinting object data for %d triangles...\n", nTrigs);
       for(int i = 0; i < nTrigs; i++){
              printf("Triangle %d:\n\t P1: (%f, %f, %f) \n\t P2: (%f, %f, %f) \n\t P3:(%f, %f, %f) \n\n",
                     i, trigs[9*i], trigs[9*i + 1], trigs[9*i + 2], trigs[9*i + 3],
                     trigs[9*i + 4], trigs[9*i + 5], trigs[9*i + 6], trigs[9*i + 7],
                     trigs[9*i + 8]);
       }
}
void render_and_buffer(struct gpu_data *d_gDat, struct gpu_data *h_gDat, struct cpu_data *cDat){
       
       // 1. Paint pixels white and reset depth array to max
       cudaMemset(h_gDat->d_pixels_arr, (Uint8)(0), cDat->pix_arr_size);
       cudaMemset(h_gDat->d_depthScreen, (Uint8)(0xFF), cDat->depth_arr_size);

       // Turn all triangle vectors into cords
       int n = cDat->nTrigs;
       dim3 grid_size( (int)(n*3 / 512) + 1); // 5000
       dim3 block_size(512);

       cordify<<<grid_size, block_size>>>(d_gDat, n*9);//<<<1,1>>>(h_gDat->d_cords_arr);
       
       float *temp = (float *)malloc(sizeof(float)*n*9);
       cudaMemcpy(temp, h_gDat->d_cords_arr, sizeof(float)*9*n, cudaMemcpyDeviceToHost);
       //print_cords(n, temp);
       free(temp);

       // draw each triangle
       dim3 gridsize2((int)(n/512) + 1);
       draw<<<gridsize2, 512>>>(d_gDat, n*9);

       // Copy the pixel array back to the window
       cudaMemcpy(cDat->pixels_arr, h_gDat->d_pixels_arr, (cDat->pix_arr_size)*sizeof(Uint8), cudaMemcpyDeviceToHost);


}



