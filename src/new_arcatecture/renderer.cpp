#include "renderer.h"
#include "common.h"

#include "../gpuController.h"
#include "../File_Conversion/readSTL.h"


typedef struct asset_t{
       int nTrigs;
       float *trigs;
       uint8_t *colors;
} asset_t;

typedef struct static_render_data_t{
       // Stored assets on gpu - most dynamic
       int nAssets;
       asset_t *head;

       // output frame buffer + info
       uint8_t *pixels_arr;
       uint32_t *depthScreen_arr;
       uint8_t BPP;
       uint32_t pitch;
       uint32_t w;
       uint32_t h;

       // temporary cords storage -> needed for every asset...
       uint32_t cord_arr_len;
       float *d_cords_arr;

} static_render_data_t;





static int nTrigs = 0;
static float *trigs = NULL;
static Uint8 *clrs = NULL;
static struct cpu_data *render_data_cpu = NULL;
static struct gpu_data *h_render_data_gpu = NULL;
static struct gpu_data *d_render_data_gpu = NULL;
static dynamic_render_data_t *dynamic_render_data = NULL;

void kill_all();
int cuda_sender_loop(LPVOID param);


DWORD WINAPI cuda_sender_init(LPVOID param){

       thread_args_t *args = (thread_args_t *)param;
       HANDLE sendkernel_sem = args->sem1;
       HANDLE finish_use_sem = args->sem2;
       dynamic_render_data = (dynamic_render_data_t *)args->p1;


       printf("Waiting for FIRST command to start init\n");
       DWORD ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command
       printf("Got signal, starting init\n");


       // Create object data
       char *filename = (char *)"blaarkop.stl";
       if(!load_triangles_stl(filename, &trigs, (char **)(&clrs), &nTrigs)){return 0;};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); return 0;}


       // Create controller data
       float pos[3] = {10, 10, 0};
       float view[3] = {-10, -10, -10};

       // Create render data and links to window
       if(create_render_data(&render_data_cpu, &h_render_data_gpu, &d_render_data_gpu, dynamic_render_data->imageSurface, nTrigs, trigs, clrs, dynamic_render_data->view, dynamic_render_data->offset) == 0){printf("Error init render data. \n");return 0;}
       
       ReleaseSemaphore( finish_use_sem,  // handle to semaphore
                                   1,            // increase count by one
                                   NULL);       // not interested in previous count
       free(trigs);
       free(clrs);

       cuda_sender_loop(param);

       return 1;
}



int cuda_sender_loop(LPVOID param){
       thread_args_t *args = (thread_args_t *)param;
       HANDLE sendkernel_sem = args->sem1;
       HANDLE finish_use_sem = args->sem2;
       void *shared_p = args->p1;
       DWORD ret;


       while(1){
              // 1) wait for request to send render frame

              printf("Waiting for command to start making the frame\n");
              ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command
              printf("Got signal, loading from the dynamic data\n");


              update_lens(render_data_cpu, dynamic_render_data->view, dynamic_render_data->offset);
              update_GPU_lens(h_render_data_gpu, render_data_cpu);
              render_and_buffer(d_render_data_gpu, h_render_data_gpu, render_data_cpu, 2, 2);
              
              ReleaseSemaphore( finish_use_sem,  // handle to semaphore
                                   1,            // increase count by one
                                   NULL);       // not interested in previous count
              // 2) get the dynamic data and load it 



              // 3) launch all the kernels


              // 4) wait for kernel competion



              // 5) check for static data updates

       }
       return 0;
}



void kill_all(){
       kill_cpu_data(render_data_cpu);
       kill_gpu_data(h_render_data_gpu);
       //cudaFree(d_render_data_gpu);
}