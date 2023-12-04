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


       init(dynamic_render_data->imageSurface);

       // 1) Init a asset onto the device
       
       
       // Create object data
       char *filename = (char *)"blaarkop.stl";
       if(!load_triangles_stl(filename, &trigs, (char **)(&clrs), &nTrigs)){exit(1);};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); exit(1);}
       
       alloc_asset(trigs, clrs, nTrigs, 0);

       float mat[9] = {1, 0, 0, 0, 0, -1, 0, 1, 0};       // 90 deg, y away from x ccw
       int i = rotate_asset(0, mat);
       passert(i != -1, "ERROR with rotattion");

       free(trigs);
       free(clrs);

       char *filename2 = (char *)"this.stl";
       if(!load_triangles_stl(filename2, &trigs, (char **)(&clrs), &nTrigs)){exit(1);};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); exit(1);}

       alloc_asset(trigs, clrs, nTrigs, 0);

       float mat2[9] = {1, 0, 0, 0, 0, -1, 0, 1, 0};       // 90 deg, y away from x ccw
       int i2 = rotate_asset(1, mat2);
       passert(i2 != -1, "ERROR with rotattion");

       free(trigs);
       free(clrs);

       // Creating lines
       printf("Making lines...\n");
       int nLines = 60;
       float *lines = (float *)malloc(sizeof(float)*60*6);
       uint8_t *clrs_l = (uint8_t *)malloc(sizeof(uint8_t)*60*3);

       for(int i = 0; i < 20; i += 1){
              setVector(lines + i*6, i, 0, 0);
              setVector(lines + i*6 + 3, (i + 1), 0, 0);
              clrs_l[i*3] = 0x00;
              clrs_l[i*3 + 1] = 0x00;
              clrs_l[i*3 + 2] = 0xFF;

              setVector(lines + (i + 20)*6, 0, i, 0);
              setVector(lines + (i + 20)*6 + 3, 0, (i + 1), 0);
              clrs_l[(i + 20)*3] = 0x00;
              clrs_l[(i + 20)*3 + 1] = 0xFF;
              clrs_l[(i + 20)*3 + 2] = 0x00;
       
              setVector(lines + (i + 40)*6, 0, 0, i);
              setVector(lines + (i + 40)*6 + 3, 0, 0, (i + 1));
              clrs_l[(i + 40)*3] = 0xFF;
              clrs_l[(i + 40)*3 + 1] = 0x00;
              clrs_l[(i + 40)*3 + 2] = 0x00;

       }
       printf("Allocating space for asset...\n");
       alloc_asset(lines, clrs_l, nLines, 1);

       printf("Done alocs\n");
       free(lines);
       free(clrs_l);

       printf("Starting redner loop\n");
       cuda_sender_loop(param);

       return 1;
}



int cuda_sender_loop(LPVOID param){
       DWORD ret;
       int reti;

       render_and_buffer_2(dynamic_render_data->view_real, dynamic_render_data->offset_real, dynamic_render_data->inst_head);
       ReleaseSemaphore( finish_use_sem, 1, NULL);

       printf("Starting render loop");
       while(1){

              // 1) wait for request to send render frame
              ret = WaitForSingleObject(sendkernel_sem, INFINITE); // decrement command
              printf("Got signal - Rendering...\n");
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
