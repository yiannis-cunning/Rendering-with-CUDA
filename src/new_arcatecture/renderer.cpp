#include "renderer.h"
#include "common.h"


#include "../File_Conversion/readSTL.h"
#include "renderer_cuda.h"





static int nTrigs = 0;
static float *trigs = NULL;
static Uint8 *clrs = NULL;
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



       // 1) Wait for access to dynamic render data - signal to start a frame
       printf("Waiting for FIRST command to start init\n");
       DWORD ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command
       printf("Got signal, starting init\n");



       // 1) Init a asset onto the device

       // Create object data
       char *filename = (char *)"blaarkop.stl";
       if(!load_triangles_stl(filename, &trigs, (char **)(&clrs), &nTrigs)){return 0;};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); return 0;}
       
  
       init(dynamic_render_data->imageSurface);
       alloc_asset(trigs, clrs, nTrigs);


       free(trigs);
       free(clrs);

       cuda_sender_loop(param);

       return 1;
}



int cuda_sender_loop(LPVOID param){
       DWORD ret;
       int reti;

       render_and_buffer_2(dynamic_render_data->view_real, dynamic_render_data->offset_real, dynamic_render_data->inst_head);
       ReleaseSemaphore( finish_use_sem, 1, NULL);


       while(1){

              // 1) wait for request to send render frame
              ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command

              // 2) get the dynamic data and load it  + launch all the kernels
              render_and_buffer_2(dynamic_render_data->view_real, dynamic_render_data->offset_real, dynamic_render_data->inst_head);

              ReleaseSemaphore( finish_use_sem, 1, NULL);
              


              // 5) check for static data updates

       }
       return 0;
}



void kill_all(){
       kill();
}
