
#include "renderer_cuda.h"
#include "Windowsnumerics.h"

void passert_cuda(bool cond, const char *msg){
       if(!cond){
              printf("CUDA ERROR: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError()));
              exit(1);
       }
}


void safe_cudaAlloc(void **dest, uint32_t size, const char *msg){
       cudaMalloc(dest, size);
       if(*dest == NULL){
              printf("CUDA ALLOCATION ERROR: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError()));
              exit(1);
       }
}




typedef struct asset_record_t {    // Linked list struct
       int type;

       uint32_t             nTrigs;
       float                *d_trigs;

       uint32_t             nLines;
       float                *d_lines;

       uint8_t              *d_colors;

       asset_record_t       *next;
} asset_record_t;



typedef struct gpu_allocations_t{

       // Screen info
       uint32_t                    *d_depthScreen_arr;
       uint8_t                     *d_pixels_arr;
       uint8_t                     BPP;
       uint32_t                    pitch;
       uint32_t                    w;
       uint32_t                    h;

       uint32_t                    pixel_arr_size;
       uint32_t                    depth_arr_size;

       uint8_t                     *dest_pixel_arr;

       // Assets allocations
       int                         n_assets;
       asset_record_t              *asset_record_head;         // linked list of allocations - index = id

       // Buffer info
       uint32_t                    sz_buffer;
       float                       *d_cord_arr_buffer;

} gpu_allocations_t;



static gpu_allocations_t *allocs;


void add_record(asset_record_t *head, asset_record_t *new_record){
       if(head == NULL){
              return ;
       }
       while(head->next != NULL){
              head = head->next;
       }
       head->next = new_record;
}

asset_record_t *get_nth_record(asset_record_t *head, int n){
       while(head != NULL && n != 0){
              n = n - 1;
              head = head->next;
       }
       return head;
}



int init(SDL_Surface *image){
       int count;
       cudaError_t ret;
       cudaDeviceProp prop;
       ret = cudaGetDevice(&count);
       passert_cuda(ret == cudaSuccess, "Getting current device");

       ret = cudaGetDeviceProperties(&prop, count);
       passert_cuda(ret == cudaSuccess, "Getting device properties");
       printf("\nUsing device: %s\n", prop.name);





       allocs = (gpu_allocations_t *)calloc(sizeof(gpu_allocations_t), 1);
       passert(allocs != NULL, "Initing cuda renderer");

       allocs->BPP              = image->format->BytesPerPixel;
       allocs->pitch            = image->pitch;
       allocs->w                = image->w;
       allocs->h                = image->h;
       allocs->dest_pixel_arr   = (Uint8 *)image->pixels;

       int pixel_arr_size = (allocs->pitch)*(allocs->h)*sizeof(Uint8);
       int depth_arr_size = (allocs->w)*(allocs->h)*sizeof(uint32_t);

       allocs->pixel_arr_size = pixel_arr_size;
       allocs->depth_arr_size = depth_arr_size;


       safe_cudaAlloc((void **)&(allocs->d_depthScreen_arr), depth_arr_size, "Depth array");
       safe_cudaAlloc((void **)&(allocs->d_pixels_arr), pixel_arr_size, "Pixel array");

       return 0;
}

int alloc_asset(float *vectors, uint8_t *colors, uint32_t nElems, int type){
       if(type != 0 && type != 1){return -1;}
       
       allocs->n_assets += 1;

       // 1) make a record for this asset allocation
       asset_record_t *new_asset_record = (asset_record_t *)calloc(sizeof(asset_record_t), 1);
       new_asset_record->next      = NULL;
       new_asset_record->type      = type;

       if(allocs->asset_record_head == NULL){
              allocs->asset_record_head = new_asset_record;
       } else{
              add_record(allocs->asset_record_head, new_asset_record);
       }

       printf("Added record\n");
       // 2) Allocate space for vector array
       int elem_arr_size = 0;
       int color_arr_size = nElems * sizeof(uint8_t) * 3;
       if(type == 0){
              elem_arr_size = nElems * sizeof(float) * 9;
              new_asset_record->nTrigs = nElems;
              safe_cudaAlloc((void **)&(new_asset_record->d_trigs), elem_arr_size, "New asset trianlge array");
              cudaMemcpy(new_asset_record->d_trigs, vectors, elem_arr_size, cudaMemcpyHostToDevice);
       } else if(type == 1){
              elem_arr_size = nElems * sizeof(float) * 6;
              new_asset_record->nLines = nElems;
              safe_cudaAlloc((void **)&(new_asset_record->d_lines), elem_arr_size, "New asset line array");
              cudaMemcpy(new_asset_record->d_lines, vectors, elem_arr_size, cudaMemcpyHostToDevice);
       }

       // 3) copy over the color array
       safe_cudaAlloc((void **)&(new_asset_record->d_colors), color_arr_size, "New color array");
       cudaMemcpy(new_asset_record->d_colors, colors, color_arr_size, cudaMemcpyHostToDevice);

       printf("Allocated and copied over data\n");


       if(elem_arr_size > allocs->sz_buffer){
              // 4) reseize buffer
              float *new_allocation_cord = NULL;
              safe_cudaAlloc((void **)&(new_allocation_cord), elem_arr_size, "Resizing cord/buffer array");
              if(allocs->d_cord_arr_buffer != NULL){cudaFree(allocs->d_cord_arr_buffer);}
              
              allocs->sz_buffer           = elem_arr_size;
              allocs->d_cord_arr_buffer   = new_allocation_cord;
       }

       printf("DONE\n");




       // return asset id
       return allocs->n_assets - 1;
}





int kill(){
       cudaFree(allocs->d_depthScreen_arr);
       cudaFree(allocs->d_pixels_arr);

       asset_record_t *cur_asset_rec = allocs->asset_record_head;
       asset_record_t *prev_asset_rec;

       while(cur_asset_rec != NULL){
              if(cur_asset_rec->type == 0){
                     cudaFree(cur_asset_rec->d_trigs);    // Free the trig array
              } else if(cur_asset_rec->type == 1){
                     cudaFree(cur_asset_rec->d_lines);    // Free the trig array
              }
              cudaFree(cur_asset_rec->d_colors);   // Free color array
              prev_asset_rec = cur_asset_rec;
              cur_asset_rec = cur_asset_rec->next;
              free(prev_asset_rec);              // Free this record
       }
       free(allocs);
       return 0;
}

__device__ float dist(float *w, float3 v, float3 o){
       float a = w[0] - o.x ;//- v.x;
       float b = w[1] - o.y ;//- v.y;
       float c = w[2] - o.z ;//- v.z;
       return sqrt(a*a + b*b + c*c);
}

__device__ float dot_off(float *a, float3 b, float3 off){
       return (*a - off.x)*b.x + (*(a+1) - off.y)*b.y + (*(a+2) - off.z)*b.z;
}

__global__ void cordify2(float *cord_arr, float *trig_arr, uint32_t num_floats, float3 view, float3 offset, float3 hx, float3 hy, float magv, int w, int h){

       int i = 3*(threadIdx.x + blockIdx.x * blockDim.x);

       float px;
       float py;
       float pz;

       if(i < num_floats){

              px = dot_off(trig_arr + i, view, offset);
              py = -1*dot_off(trig_arr + i, hx, offset);
              pz = dot_off(trig_arr + i, hy, offset);
              if(px < 0){
                     *(float *)(cord_arr + i) = -1.0f;
                     *(float *)(cord_arr + i + 1) = -1.0f;
                     *(float *)(cord_arr + i + 2) = -1.0f;  
              }
              else{
                     *(float *)(cord_arr + i) = ((py*magv*0.5)/(px) + 0.5)*w;
                     *(float *)(cord_arr + i + 1) = ((pz*magv*0.5)/(px) + 0.5)*h;
                     *(float *)(cord_arr + i + 2) = dist(trig_arr + i, view, offset);
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
 
__global__ void draw(int num_floats, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch){
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
                     points[(int)(thread_num/3)][thread_num % 3] = cord_arr[n + thread_num];
              }

              if(points[0][2] != -1 && points[1][2] != -1 && points[2][2] != -1){

                     // Next create depth arry stuff and bounding square on trianlge - all a function of points
                     if(thread_num == 0){
                            yMin = min(points[0][1], points[1][1], points[2][1], h);
                            yMin = (yMin < 0) ? 0: yMin;
                            yMax = max(points[0][1], points[1][1], points[2][1], 0);
                            yMax = (yMax > h) ? h : yMax;
                            xMin = min(points[0][0], points[1][0], points[2][0], w);
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
                     for(float yi = yMin - 1; yi <= yMax + 1; yi += ((float)blockDim.y)*0.3){
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
                                   xMaxt = (xMaxt > w) ? w : xMaxt;
                                   // Want to traverse the x range
                                   for(float xi = xMint - 1; xi <= xMaxt + 1; xi += blockDim.x*0.3){
                                          x = xi + 0.8*threadIdx.x;
                                          if(x <= xMaxt){
                                                 Iy = h - round(y);
                                                 Ix = round(x);
                                                 depth = (a + b*x + c*y)*1000;
                                                 // Make sure that the pixel is within range of the screen and compare the depth.
                                                 if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                                                        atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                                                        
                                                        if(depth_arr[Iy*w + Ix] == (int)depth){
                                                               // Paint the pixel and set new depth
                                                               //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                                               pixel_arr[Iy*pitch + Ix*BPP] = color_arr[(int)(n/3)];//*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[(int)(n/3) + 1];//*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[(int)(n/3) + 2];//*!(Ix == (int)xMaxt || Ix == xMint);

                                                        }
                                                 }
                                          }
                                   }
                            }
                     }
              }
       }
}

/*
__global__ void draw2(int nTrigs, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch){
       // Run for each triangle: have x and y dimensions on the threads
       // y dim traverses the y direction, x values traverse the x direction

       int trig_number = blockIdx.x;
       int thread_num = threadIdx.x + threadIdx.y*blockDim.x; // goes from 0->512

       if(trig_number < nTrigs){
              __shared__ float p1[3];
              __shared__ float p2[3];
              __shared__ float p3[3];

              // 1) Get the cords of triangle (float x1, float y1, float depth) * 3
              if(thread_num < 3){
                     p1[thread_num] = cord_arr[trig_number*9 + thread_num];
                     p2[thread_num] = cord_arr[trig_number*9 + 3 + thread_num];
                     p3[thread_num] = cord_arr[trig_number*9 + 6 + thread_num];
              }
              
              // 2) Find minimum bounding box on the trianlge while remain inside the screen
              int yMin = floor(min(p1[1], p2[1], p3[1]));
              yMin = (yMin < 0) ? 0: yMin;
              int yMax = ceil(max(p1[1], p2[1], p3[1]));
              yMax = (yMax > h - 1) ? h-1 : yMax;
              int xMin = floor(min(p1[0], p2[0], p3[0]));
              xMin = (xMin < 0) ? 0: xMin;
              int xMax = ceil(max(p1[0], p2[0], p3[0]));
              xMax = (xMax > w-1) ? w-1 : xMax;

              if(yMin > h-1 || yMax < 0 || xMin > w-1 || xMax < 0){
                     return;
              }

              // 3) Loop through the box
              for(int x = xMin + threadIdx.x; x <= xMax; x += blockDim.x){
                     for(int y = yMin + threadIdx.y; y <= yMax; y += blockDim.y){
                            // is this inside or outside the triangle
                     }
              }


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
                     points[(int)(thread_num/3)][thread_num % 3] = cord_arr[n + thread_num];
              }

              if(points[0][2] != -1 && points[1][2] != -1 && points[2][2] != -1){

                     // Next create depth arry stuff and bounding square on trianlge - all a function of points
                     if(thread_num == 0){
                            yMin = min(points[0][1], points[1][1], points[2][1], h);
                            yMin = (yMin < 0) ? 0: yMin;
                            yMax = max(points[0][1], points[1][1], points[2][1], 0);
                            yMax = (yMax > h) ? h : yMax;
                            xMin = min(points[0][0], points[1][0], points[2][0], w);
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
                     for(float yi = yMin - 1; yi <= yMax + 1; yi += ((float)blockDim.y)*0.8){
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
                                   xMaxt = (xMaxt > w) ? w : xMaxt;
                                   // Want to traverse the x range
                                   for(float xi = xMint - 1; xi <= xMaxt + 1; xi += blockDim.x*0.8){
                                          x = xi + 0.8*threadIdx.x;
                                          if(x <= xMaxt){
                                                 Iy = h - round(y);
                                                 Ix = round(x);
                                                 depth = (a + b*x + c*y)*1000;
                                                 // Make sure that the pixel is within range of the screen and compare the depth.
                                                 if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                                                        atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                                                        
                                                        if(depth_arr[Iy*w + Ix] == (int)depth){
                                                               // Paint the pixel and set new depth
                                                               //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                                               pixel_arr[Iy*pitch + Ix*BPP] = color_arr[(int)(n/3)];//*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[(int)(n/3) + 1];//*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[(int)(n/3) + 2];//*!(Ix == (int)xMaxt || Ix == xMint);

                                                        }
                                                 }
                                          }
                                   }
                            }
                     }
              }
       }
}

*/

__device__ void draw_pixel(uint8_t *pixel_array, uint8_t *clr, int Ix, int Iy, uint32_t depth, int w, int h, uint8_t BPP, int pitch, uint32_t *depth_arr){
       
       if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
              atomicMin(depth_arr + Iy*w + Ix, depth);
              
              if(depth_arr[Iy*w + Ix] == depth){
                     // Paint the pixel and set new depth
                     //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                     pixel_array[Iy*pitch + Ix*BPP] = clr[0];//*!(Ix == (int)xMaxt || Ix == xMint);
                     pixel_array[Iy*pitch + Ix*BPP + 1] = clr[1];//*!(Ix == (int)xMaxt || Ix == xMint);
                     pixel_array[Iy*pitch + Ix*BPP + 2] = clr[2];//*!(Ix == (int)xMaxt || Ix == xMint);

              }
       }  
}


// 1 thread for every 1 line / 2 cords
__global__ void draw_lines(int num_lines, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch){


       int thread_num = threadIdx.x + blockIdx.x*blockDim.x; // goes from 0->512
       int line_num = thread_num;

       if(thread_num < num_lines){


              float p_start[3];
              float p_stop[3];
              p_start[0] = cord_arr[thread_num*6];
              p_start[1] = cord_arr[thread_num*6 + 1];
              p_start[2] = cord_arr[thread_num*6 + 2];

              p_stop[0] = cord_arr[thread_num*6 + 3];
              p_stop[1] = cord_arr[thread_num*6 + 4];
              p_stop[2] = cord_arr[thread_num*6 + 5];
              
              float dx = p_stop[0] - p_start[0];
              float dy = p_stop[1] - p_start[1];
              float dd = p_stop[2] - p_start[2];
              float dist = sqrt(dx*dx + dy*dy);
              dx = dx/dist;
              dy = dy/dist;
              
              if(p_start[0] == -1 || p_stop[0] == -1){
                     return;
              }
              if(p_start[0] < 0 || p_start[0] > w || p_stop[0] < 0 || p_stop[0] > w){
                     return;
              }
              if(p_start[1] < 0 || p_start[1] > h || p_stop[1] < 0 || p_stop[1] > h){
                     return;
              }

              //draw_pixel(pixel_arr, color_arr, 10 + thread_num, 10, 0, w, h, BPP, pitch, depth_arr);
              //draw_pixel(pixel_arr, color_arr, (int)p_start[0], h - (int)p_start[1], 0, w, h, BPP, pitch, depth_arr);

              float x = p_start[0];
              float y = p_start[1];
              for(int i = 0; i < dist; i += 1){
                     x += dx;
                     y += dy;
                     draw_pixel(pixel_arr, color_arr + thread_num*3, round(x), h - round(y), 0, w, h, BPP, pitch, depth_arr);
              }
              
              return;
              int Iy, Ix;
              int depth = 0;


              for(int j = 0; j < 16; j+= 1){
                     Iy = h - round(p_start[1] + j%4);
                     Ix = round(p_start[0] + (int)j/4);
                     // Make sure that the pixel is within range of the screen and compare the depth.
                     if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                            atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                            
                            if(depth_arr[Iy*w + Ix] == (int)depth){
                                   // Paint the pixel and set new depth
                                   //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                   pixel_arr[Iy*pitch + Ix*BPP] = color_arr[thread_num*3];//*!(Ix == (int)xMaxt || Ix == xMint);
                                   pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[thread_num*3 + 1];//*!(Ix == (int)xMaxt || Ix == xMint);
                                   pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[thread_num*3 + 2];//*!(Ix == (int)xMaxt || Ix == xMint);

                            }
                     }  
              }
              for(int j = 0; j < 16; j+=1){
                     Iy = h - round(p_stop[1] + j%4);
                     Ix = round(p_stop[0] + (int)j/4);
                     // Make sure that the pixel is within range of the screen and compare the depth.
                     if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                            atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                            
                            if(depth_arr[Iy*w + Ix] == (int)depth){
                                   // Paint the pixel and set new depth
                                   //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                   pixel_arr[Iy*pitch + Ix*BPP] = color_arr[thread_num*3];//*!(Ix == (int)xMaxt || Ix == xMint);
                                   pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[thread_num*3 + 1];//*!(Ix == (int)xMaxt || Ix == xMint);
                                   pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[thread_num*3 + 2];//*!(Ix == (int)xMaxt || Ix == xMint);

                            }
                     }  
              }

              /*
              float dx = p_stop[0] - p_start[0];
              float dy = p_stop[1] - p_start[1];
              float dd = p_stop[2] - p_start[2];
              float dist = sqrt(dx*dx + dy*dy);
              
              dx = dx/dist;
              dy = dy/dist;
              dd = dd/dist;

              float x, y, depth;
              int Ix, Iy;
              for(float i = 0; i < dist; i += 0.75){
                     x += dx;
                     y += dy;
                     depth += dd;
                     for(int j = 0; j < 9; j+=1){

                            Iy = h - round(y + j%3);
                            Ix = round(x + (int)j/3);
                            // Make sure that the pixel is within range of the screen and compare the depth.
                            if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                                   atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                                   
                                   if(depth_arr[Iy*w + Ix] == (int)depth){
                                          // Paint the pixel and set new depth
                                          //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                          pixel_arr[Iy*pitch + Ix*BPP] = color_arr[thread_num*3];//*!(Ix == (int)xMaxt || Ix == xMint);
                                          pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[thread_num*3 + 1];//*!(Ix == (int)xMaxt || Ix == xMint);
                                          pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[thread_num*3 + 2];//*!(Ix == (int)xMaxt || Ix == xMint);

                                   }
                            }
                     }
              }*/
       }
}

__device__ int inrange(float3 point, int w, int h){
       if(point.x < 0 || point.x >= w || point.y < 0 || point.y >= h){
              return 0;
       }
       return 1;
}



__global__ void correct_lines(int num_cords, float *cord_arr, int h, int w){


       int cord_number = threadIdx.x + blockIdx.x*blockDim.x;
       if(cord_number < num_cords){
              int ref;
              if(cord_number%2 == 0){ref = cord_number + 1;}
              else{ref = cord_number - 1;}

              float3 point;
              float3 other;
              point.x = cord_arr[cord_number*3];
              point.y = cord_arr[cord_number*3 + 1];
              point.z = cord_arr[cord_number*3 + 2];
              other.x = cord_arr[ref*3];
              other.y = cord_arr[ref*3 + 1];
              other.z = cord_arr[ref*3 + 2];

              if(inrange(point, w, h) || point.x == -1 || other.x == -1){
                     return;
              }
              __syncthreads();


              float3 ans;
              float inter;
              // Do left wall
              if(point.x < 0 && other.x > 0){
                     inter = point.y + -1*point.x *(other.y - point.y)/(other.x - point.x);
                     
                     if(inter > 0 && inter < h){
                            ans.x = 0;
                            ans.y = inter;
                            cord_arr[cord_number*3] = ans.x;
                            cord_arr[cord_number*3 + 1] = ans.y;
                            return;
                     }
              }
              // Do top wall
              if(point.y < 0 && other.y > 0){
                     inter = point.x - point.y*(other.x - point.x)/(other.y - point.y);
                     if(inter >= 0 && inter < w){
                            ans.x = inter;
                            ans.y = 0;
                            cord_arr[cord_number*3] = ans.x;
                            cord_arr[cord_number*3 + 1] = ans.y;
                            return;
                     }
              }
              point.x = point.x - w;
              point.y = point.y - h;
              other.x = other.x - w;
              other.y = other.y - h;
              // Do right wall
              if(point.x > 0 && other.x < 0){
                     inter = point.y + -1*point.x *(other.y - point.y)/(other.x - point.x);
                     
                     if(inter < 0 && inter > -1*h){
                            ans.x = 0 + w;
                            ans.y = inter + h;
                            cord_arr[cord_number*3] = ans.x;
                            cord_arr[cord_number*3 + 1] = ans.y;
                            // set ans
                            return;
                     }
              }
              // Do bottom wall
              if(point.y > 0 && other.y < 0){
                     inter = point.x - point.y*(other.x - point.x)/(other.y - point.y);
                     
                     if(inter <= 0 && inter >= -1*w){
                            ans.x = inter + w;
                            ans.y = 0 + h;
                            cord_arr[cord_number*3] = ans.x;
                            cord_arr[cord_number*3 + 1] = ans.y;
                            return;
                     }
              }
              cord_arr[cord_number*3] = -1;
              cord_arr[cord_number*3 + 1] = -1;
              cord_arr[cord_number*3 + 2] = -1;


       }

}








float3 vec_to_float3(float *v){
       float3 ans;
       ans.x = v[0];
       ans.y = v[1];
       ans.z = v[2];
       return ans;
}


int render_and_buffer_2(float *view, float *offset, instance_t *instances_arr){
       // Launch 1 kernel per instances
       
       printf("Starting to render...\n");
       // 1. Paint pixels white and reset depth array to max
       cudaMemset(allocs->d_pixels_arr, (Uint8)(0), allocs->pixel_arr_size);
       cudaMemset(allocs->d_depthScreen_arr, (Uint8)(0xFF), allocs->depth_arr_size);

       instance_t *inst            = instances_arr;
       asset_record_t *asset       = NULL;
       
       float eff_offset[3] = {0, 0, 0};
       float hx[3] = {0, 0, 0};
       float hy[3] = {0, 0, 0};
       float az[3] = {0, 0, 1};
       float magv = 0;

       while(inst != NULL){
              printf("Getting next instance...\n");
              if(inst->is_visible == false){continue;}  // NON-VISIBLE
              asset = get_nth_record(allocs->asset_record_head, inst->asset_id);
              if(asset == NULL){continue;}             // UNKOWN ASSET
              
              //printf("Drawing instance\n");

              subVec(offset, inst->offset, eff_offset);
              
              cross(view, az, hx);
              cross(hx, view, hy);
              normalize(hx);
              normalize(hy);
              magv = vecMag(view);
              printf("Have first instance: type = %d, asset_id = %d, asset_type = %d, pos: %f. %f. %f\n", inst->type, inst->asset_id, asset->type, inst->offset[0], inst->offset[1], inst->offset[2]);
              if(inst->type == 0 && asset->type == 0){
                     dim3 grid_size( (int)(asset->nTrigs*3 / 512) + 1);
                     dim3 block_size(512);
                     // Kernel launch - cordify
                     cordify2<<<grid_size, block_size>>>(allocs->d_cord_arr_buffer, 
                                                        asset->d_trigs, asset->nTrigs*9,
                                                        vec_to_float3(view), vec_to_float3(eff_offset),
                                                        vec_to_float3(hx), vec_to_float3(hy),
                                                        magv, allocs->w, allocs->h);

                     // Kernel launch - paint
                     dim3 blck(16,16);
                     int n = asset->nTrigs;

                     //int num_floats, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch
                     draw<<<n, blck>>>(n*9, allocs->d_cord_arr_buffer, allocs->d_depthScreen_arr,
                                                 allocs->d_pixels_arr, asset->d_colors, allocs->h, 
                                                 allocs->w, allocs->BPP, allocs->pitch);

              } else if(inst->type = 1 && asset->type == 1){
                     printf("Sending kernel lines %d", asset->nLines);
                     dim3 grid_size( (int)(asset->nLines*2 / 512) + 1);
                     dim3 block_size(512);
                     // Kernel launch - cordify
                     cordify2<<<grid_size, block_size>>>(allocs->d_cord_arr_buffer, 
                                                        asset->d_lines, asset->nLines*6,
                                                        vec_to_float3(view), vec_to_float3(eff_offset),
                                                        vec_to_float3(hx), vec_to_float3(hy),
                                                        magv, allocs->w, allocs->h);


                     correct_lines<<<grid_size, block_size>>>(asset->nLines*2, allocs->d_cord_arr_buffer, allocs->h, allocs->w);

                     // Kernel launch - paint
                     dim3 grid_size2( (int)(asset->nLines / 512) + 1);
                     //int num_lines, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch
                     //int num_floats, float *cord_arr, uint32_t *depth_arr, uint8_t *pixel_arr, uint8_t *color_arr, int h, int w, uint8_t BPP, int pitch
                     draw_lines<<<grid_size2, block_size>>>(asset->nLines, allocs->d_cord_arr_buffer, allocs->d_depthScreen_arr,
                                                 allocs->d_pixels_arr, asset->d_colors, allocs->h, 
                                                 allocs->w, allocs->BPP, allocs->pitch);
                     
              }
              inst = inst->next;
       }


       // 3. Copy the pixel array back to the window
       cudaMemcpy(allocs->dest_pixel_arr, allocs->d_pixels_arr, allocs->pixel_arr_size, cudaMemcpyDeviceToHost);
       return 1;
}


/*
A = [ v1 ]
    [ v2 ]
    [ v3 ]

*/
__global__ void rotate(int npoints, float3 v1, float3 v2, float3 v3, float *vecs){
       int i = (threadIdx.x + blockIdx.x * blockDim.x);

       float inx;
       float iny;
       float inz;

       if(i < npoints){
              inx = vecs[i*3];
              iny = vecs[i*3 + 1];
              inz = vecs[i*3 + 2];

              vecs[3*i] = v1.x*inx + v1.y*iny + v1.z*inz;
              vecs[3*i + 1] = v2.x*inx + v2.y*iny + v2.z*inz;
              vecs[3*i + 2] = v3.x*inx + v3.y*iny + v3.z*inz;

       }
}
 



int rotate_asset(int asset_id, float *matrix){
       
       asset_record_t *head = allocs->asset_record_head;

       int i = 0;
       while(head != NULL && i != asset_id){
              head = head->next;
              i += 1;
       }
       if(head == NULL){
              return -1;
       }
       rotate<<<(int)(head->nTrigs*3/512 + 1), 512>>>(head->nTrigs*3, 
                                   vec_to_float3(matrix), 
                                   vec_to_float3(matrix + 3),
                                   vec_to_float3(matrix + 6),
                                   head->d_trigs);
       return 1;


}