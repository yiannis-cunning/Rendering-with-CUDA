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
              //bot = dot(dat->d_trigs + i, dat->d_v, dat->d_offset);
              mag = dist(dat->d_trigs + i, dat->d_v, dat->d_offset);

              if(bot <= 0.1* *(float *)(dat->d_mag))
              //if(bot <= 0) //-0.1* *(float *)(dat->d_mag))
              //if(bot/(*(float *)(dat->d_mag)) <= 0.1 || bot >= (*(float *)(dat->d_mag)))
              //if(bot == 0)
              { // behind
                     *(float *)(dat->d_cords_arr + i) = -1.0f;
                     *(float *)(dat->d_cords_arr + i + 1) = -1.0f;
                     *(float *)(dat->d_cords_arr + i + 2) = -1.0f;
              }
              else
              {
                     //
                     //bot = *(float *)(dat->d_mag) - bot;
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
       // Run for each triangle: have x and y dimensions on the threads
       // y dim traverses the y direction, x values traverse the x direction


       int n = 9*blockIdx.x;
       int thread_num = threadIdx.x + threadIdx.y*blockDim.x; // goes from 0->512

       if(n < num_floats){
              __shared__ float points[3][3];                   // points[j] = p1 = {x, y ,z} (cords that define the trianlge)
              __shared__ float yMin, yMax, xMin, xMax;         // constants to traverse the x values of the triangle
              __shared__ float as[3];
              __shared__ float bs[3];
              __shared__ float cs[3];
              __shared__ float ds[3];
              __shared__ float c, b, a;                        // depth(x,y) = a+bx+cy (returns float) -> depth array is in int (mult by 1000)

              float depth, t, xT, xMint, xMaxt, y, x;        // Temporary local variables
              int Ix, Iy;
              // First copy over triangle cords to shared memory
              if(thread_num < 9 ){
                     points[(int)(thread_num/3)][thread_num % 3] = dat->d_cords_arr[n + thread_num];
              }

              if(points[0][2] != -1 && points[1][2] != -1 && points[2][2] != -1){

                     // Next create depth arry stuff and bounding square on trianlge - all a function of points
                     if(thread_num == 0){
                            yMin = min(points[0][1], points[1][1], points[2][1], dat->d_h);
                            yMin = (yMin < 0) ? 0: yMin;
                            yMax = max(points[0][1], points[1][1], points[2][1], 0);
                            yMax = (yMax > dat->d_h) ? dat->d_h : yMax;
                            xMin = min(points[0][0], points[1][0], points[2][0], dat->d_w);
                            xMax = max(points[0][0], points[1][0], points[2][0], 0);

                            c = points[0][1]*(points[2][0] - points[1][0]) - points[1][1]*(points[2][0] - points[0][0]) + points[2][1]*(points[1][0] - points[0][0]);
                            b = points[2][0] - points[0][0];
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
                     }

                     // Then spilt up the creation of the x range function for the triangle (determines the pixels to traverse given a y value)
                     if(thread_num < 3){
                            if(points[(thread_num+1)%3][1] == points[thread_num][1]){ // for vertical lines
                                   as[thread_num] = 0;
                                   bs[thread_num] = -1;
                            }
                            else{
                                   as[thread_num] = 1/(points[(thread_num+1)%3][1] - points[thread_num][1]);
                                   bs[thread_num] = -1 * points[thread_num][1] * as[thread_num];
                            }
                            cs[thread_num] = points[(thread_num+1)%3][0] - points[thread_num][0];
                            ds[thread_num] = points[thread_num][0];
                     }
                     __syncthreads();
                     // 0, 0.9, 1.8, 2.7 --> bd = 2
                     // Then traverse the triangle. -> want to run from yMin to yMax inc 0.9
                     for(float yi = yMin; yi <= yMax; yi += ((float)blockDim.y)*0.8){
                            y = yi + 0.8*((float)threadIdx.y);
                            //y = yi;

                            if(y <= yMax){
                                   // Find the x range
                                   xMint = xMax;
                                   xMaxt = xMin;
                                   
                                   for(int j = 0; j < 3; j++){
                                          t = as[j]*y + bs[j];
                                          xT = cs[j]*t + ds[j];
                                          if(t <= 1 && t >= 0)
                                          {
                                                 if(xT < xMint){xMint = xT;}
                                                 if(xT > xMaxt){xMaxt = xT;}
                                          }
                                   }
                                   xMint = (xMint < 0) ? 0: xMint;
                                   xMaxt = (xMaxt > dat->d_w) ? dat->d_w : xMaxt;
                                   // Want to traverse the x range
                                   for(float xi = xMint; xi <= xMaxt; xi += blockDim.x*0.8){
                                          x = xi + 0.8*threadIdx.x;
                                          if(x <= xMaxt){
                                                 Iy = dat->d_h - (int)y;
                                                 Ix = (int)x;
                                                 depth = (a + b*x + c*y)*1000;
                                                 // Make sure that the pixel is within range of the screen and compare the depth.
                                                 if(Iy >= 0 && Iy < dat->d_h && Ix >= 0 && Ix < dat->d_w){
                                                        atomicMin(dat->d_depthScreen + Iy*dat->d_w + Ix, (int)depth);
                                                        
                                                        if(dat->d_depthScreen[Iy*dat->d_w + Ix] == (int)depth){
                                                               // Paint the pixel and set new depth
                                                               //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                                               dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP] = dat->d_colors[(int)(n/3)]*!(Ix == (int)xMaxt || Ix == xMint);
                                                               dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP + 1] = dat->d_colors[(int)(n/3) + 1]*!(Ix == (int)xMaxt || Ix == xMint);
                                                               dat->d_pixels_arr[Iy*dat->d_pitch + Ix*dat->d_BPP + 2] = dat->d_colors[(int)(n/3) + 2]*!(Ix == (int)xMaxt || Ix == xMint);

                                                        }
                                                 }
                                          }
                                   }
                            }
                     }
              }
       }
}


float maxh(float a, float b, float c, float d){
       return (((a > b) ? a:b) > ((c > d) ? c:d) ? ((a > b) ? a:b):((c > d) ? c:d));
}

float minh(float a, float b, float c, float d){
       return (((a < b) ? a:b) < ((c < d) ? c:d) ? ((a < b) ? a:b):((c < d) ? c:d));
}
void print_cords(int nTrigs, float *trigs){
       printf("\nPrinting object data for %d triangles...\n", nTrigs);
       for(int i = 0; i < nTrigs; i++){
              printf("Triangle %d:\n\t P1: (%f, %f, %f) \n\t P2: (%f, %f, %f) \n\t P3:(%f, %f, %f) \n\n",
                     i, trigs[9*i], trigs[9*i + 1], trigs[9*i + 2], trigs[9*i + 3],
                     trigs[9*i + 4], trigs[9*i + 5], trigs[9*i + 6], trigs[9*i + 7],
                     trigs[9*i + 8]);
              printf("Bounding box: xs: %f, xf: %f ;; ys: %f, yf: %f\n", 
                     minh(trigs[9*i], trigs[9*i + 3], trigs[9*i + 6], trigs[9*i + 6]), 
                     maxh(trigs[9*i], trigs[9*i + 3], trigs[9*i + 6], trigs[9*i + 6]),
                     minh(trigs[9*i + 1], trigs[9*i + 4], trigs[9*i + 7], trigs[9*i + 8]),
                     maxh(trigs[9*i + 1], trigs[9*i + 4], trigs[9*i + 7], trigs[9*i + 8]));
                     
       }
}
void render_and_buffer(struct gpu_data *d_gDat, struct gpu_data *h_gDat, struct cpu_data *cDat, int a, int b){
       
       // 1. Paint pixels white and reset depth array to max
       cudaMemset(h_gDat->d_pixels_arr, (Uint8)(0), cDat->pix_arr_size);
       cudaMemset(h_gDat->d_depthScreen, (Uint8)(0xFF), cDat->depth_arr_size);

       // Turn all triangle vectors into cords
       int n = cDat->nTrigs;
       dim3 grid_size( (int)(n*3 / 512) + 1);
       dim3 block_size(512);

       cordify<<<grid_size, block_size>>>(d_gDat, n*9);

       dim3 blck(16,16);
       draw<<<n, blck>>>(d_gDat, n*9);//(nt-ns)


       // Copy the pixel array back to the window
       cudaMemcpy(cDat->pixels_arr, h_gDat->d_pixels_arr, (cDat->pix_arr_size)*sizeof(Uint8), cudaMemcpyDeviceToHost);

}


