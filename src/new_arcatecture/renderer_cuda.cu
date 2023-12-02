
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
       uint32_t             nTrigs;
       float                *d_trigs;
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

int alloc_asset(float *trigs, uint8_t *colors, uint32_t nTrigs){
       allocs->n_assets += 1;
       int trig_arr_size = nTrigs * sizeof(float) * 9;
       int color_arr_size = nTrigs * sizeof(uint8_t) * 3;

       // 1) make a record for this asset allocation
       asset_record_t *new_asset_record = (asset_record_t *)calloc(sizeof(asset_record_t), 1);
       new_asset_record->nTrigs    = nTrigs;
       new_asset_record->next      = NULL;

       if(allocs->asset_record_head == NULL){
              allocs->asset_record_head = new_asset_record;
       } else{
              add_record(allocs->asset_record_head, new_asset_record);
       }

       // 2) Make the acutal allocation for colors and trigs
       safe_cudaAlloc((void **)&(new_asset_record->d_trigs), trig_arr_size, "New asset trianlge array");
       safe_cudaAlloc((void **)&(new_asset_record->d_colors), color_arr_size, "New asset color array");

       // 3) copy over the data 
       cudaMemcpy(new_asset_record->d_trigs, trigs, trig_arr_size, cudaMemcpyHostToDevice);
       cudaMemcpy(new_asset_record->d_colors, colors, color_arr_size, cudaMemcpyHostToDevice);
       
       // 4) Resize cord array to fit biggest asset
       if(nTrigs*9*sizeof(float) > allocs->sz_buffer){
              // reseize buffer
              float *new_allocation_cord = NULL;
              safe_cudaAlloc((void **)&(new_allocation_cord), nTrigs*9*sizeof(float), "Resizing cord/buffer array");
              if(allocs->d_cord_arr_buffer != NULL){cudaFree(allocs->d_cord_arr_buffer);}
              
              allocs->sz_buffer           = nTrigs*9*sizeof(float);
              allocs->d_cord_arr_buffer   = new_allocation_cord;
       }


       // return asset id
       return allocs->n_assets - 1;
}





int kill(){
       cudaFree(allocs->d_depthScreen_arr);
       cudaFree(allocs->d_pixels_arr);

       asset_record_t *cur_asset_rec = allocs->asset_record_head;
       asset_record_t *prev_asset_rec;

       while(cur_asset_rec != NULL){
              cudaFree(cur_asset_rec->d_trigs);    // Free the trig array
              cudaFree(cur_asset_rec->d_colors);   // Free color array
              prev_asset_rec = cur_asset_rec;
              cur_asset_rec = cur_asset_rec->next;
              free(prev_asset_rec);              // Free this record
       }
       free(allocs);
       return 0;
}

__device__ float dist(float *w, float3 v, float3 o){
       float a = w[0] - o.x - v.x;
       float b = w[1] - o.y - v.y;
       float c = w[2] - o.z - v.z;
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
              if(px < magv){
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
                                   xMaxt = (xMaxt > w) ? w : xMaxt;
                                   // Want to traverse the x range
                                   for(float xi = xMint; xi <= xMaxt; xi += blockDim.x*0.8){
                                          x = xi + 0.8*threadIdx.x;
                                          if(x <= xMaxt){
                                                 Iy = h - (int)y;
                                                 Ix = (int)x;
                                                 depth = (a + b*x + c*y)*1000;
                                                 // Make sure that the pixel is within range of the screen and compare the depth.
                                                 if(Iy >= 0 && Iy < h && Ix >= 0 && Ix < w){
                                                        atomicMin(depth_arr + Iy*w + Ix, (int)depth);
                                                        
                                                        if(depth_arr[Iy*w + Ix] == (int)depth){
                                                               // Paint the pixel and set new depth
                                                               //dat->d_depthScreen[Iy*dat->d_w + Ix] = (int)depth;
                                                               pixel_arr[Iy*pitch + Ix*BPP] = color_arr[(int)(n/3)]*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 1] = color_arr[(int)(n/3) + 1]*!(Ix == (int)xMaxt || Ix == xMint);
                                                               pixel_arr[Iy*pitch + Ix*BPP + 2] = color_arr[(int)(n/3) + 2]*!(Ix == (int)xMaxt || Ix == xMint);

                                                        }
                                                 }
                                          }
                                   }
                            }
                     }
              }
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
              inst = inst->next;
       }


       // 3. Copy the pixel array back to the window
       cudaMemcpy(allocs->dest_pixel_arr, allocs->d_pixels_arr, allocs->pixel_arr_size, cudaMemcpyDeviceToHost);
       return 1;
}