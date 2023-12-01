#include "renderer.h"
#include "common.h"

#include "gpuController.h"
#include "../File_Conversion/readSTL.h"
#include "renderer_cuda.h"





static int nTrigs = 0;
static float *trigs = NULL;
static Uint8 *clrs = NULL;
static struct cpu_data *render_data_cpu = NULL;
static struct gpu_data *h_render_data_gpu = NULL;
static struct gpu_data *d_render_data_gpu = NULL;
static dynamic_render_data_t *dynamic_render_data = NULL;

HANDLE sendkernel_sem;
HANDLE finish_use_sem;



void kill_all();
int cuda_sender_loop(LPVOID param);



DWORD WINAPI cuda_sender_init(LPVOID param){

       thread_args_t *args = (thread_args_t *)param;
       sendkernel_sem = args->sem1;
       finish_use_sem = args->sem2;
       dynamic_render_data = (dynamic_render_data_t *)args->p1;


       printf("Waiting for FIRST command to start init\n");
       DWORD ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command
       printf("Got signal, starting init\n");



       // 1) Init static render data on device

       // Create object data
       char *filename = (char *)"blaarkop.stl";
       if(!load_triangles_stl(filename, &trigs, (char **)(&clrs), &nTrigs)){return 0;};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); return 0;}
       



       // Create controller data
       float pos[3] = {10, 10, 0};
       float view[3] = {-10, -10, -10};
       float vreal[3] = {-10, -10, -10};
       float oreal[3] = {10, 10, 10};

       // Create render data and links to window
       if(create_render_data(&render_data_cpu, &h_render_data_gpu, &d_render_data_gpu, dynamic_render_data->imageSurface, nTrigs, trigs, clrs, dynamic_render_data->view, dynamic_render_data->offset, vreal, oreal) == 0){printf("Error init render data. \n");return 0;}
       
       init(dynamic_render_data->imageSurface);
       alloc_asset(trigs, clrs, nTrigs);

       //ReleaseSemaphore( finish_use_sem,  // handle to semaphore
       //                            1,            // increase count by one
       //                            NULL);       // not interested in previous count
       free(trigs);
       free(clrs);

       cuda_sender_loop(param);

       return 1;
}



int cuda_sender_loop(LPVOID param){
       DWORD ret;
       int reti;
       void *static_data = NULL;
       void *dyna_data = NULL;



       while(1){
              // 1) wait for request to send render frame


              //reti = update_dynamic_data(dynamic_render_data->view, dynamic_render_data->offset, dynamic_render_data->inst_head, dynamic_render_data->nInstances);
              passert(reti == 0, "Updating dynamic render data");
              
              static_data = get_static_render_data_p();
              dyna_data = get_dynamic_render_data_p();
              passert(static_data != NULL && dyna_data != NULL, "static and dynamic data check");


              update_lens(render_data_cpu, dynamic_render_data->view, dynamic_render_data->offset, dynamic_render_data->view_real, dynamic_render_data->offset_real);
              update_GPU_lens(h_render_data_gpu, render_data_cpu, d_render_data_gpu);
              
              
              render_and_buffer(d_render_data_gpu, h_render_data_gpu, render_data_cpu, 2, 2, static_data, dyna_data);
              
              ReleaseSemaphore( finish_use_sem,  // handle to semaphore
                                   1,            // increase count by one
                                   NULL);       // not interested in previous count
              
              ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command


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
       kill();
       //cudaFree(d_render_data_gpu);
}



/*

on GPU:

static data
       - frame buffer + info
       - asset 1 arr
       - asset 2 arr
       ...

dynamic data
       - offset
       - view
       - instance buffer 1
       - instance buffer 2
       - instance buffer 3
       ...




*/