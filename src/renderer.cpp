#include "renderer.h"
#include "common.h"


#include "File_Conversion/readSTL.h"
#include "renderer_cuda.h"





static int nTrigs = 0;
static float *trigs = NULL;
static Uint8 *clrs = NULL;
static dynamic_render_data_t *dynamic_render_data = NULL;

HANDLE sendkernel_sem;
HANDLE finish_use_sem;


static HANDLE command_pend_sem, command_done_sem;
static command_t *cmd = NULL;


void kill_all();

int cuda_sender_loop(LPVOID param);



DWORD WINAPI cuda_sender_init(LPVOID param){

       thread_args_t *args = (thread_args_t *)param;
       sendkernel_sem = args->sem1;
       finish_use_sem = args->sem2;

       cmd = args->command;
       command_pend_sem = args->pending_cmd;
       command_done_sem = args->done_cmd;

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
       int nLines = 12;
       float *lines = (float *)malloc(sizeof(float)*12*6);
       uint8_t *clrs_l = (uint8_t *)malloc(sizeof(uint8_t)*12*3);
       
       int ln = 0;
       for(int z = 0; z < 20; z += 10){   // z = 0, 10
              for(int y = 0; y < 20; y += 10){ // y = 0, 10
                     setVector(lines + ln*6, 0, y, z);
                     setVector(lines + ln*6 + 3, 10, y, z);
                     clrs_l[ln*3] = 0x00;
                     clrs_l[ln*3 + 1] = 0x00;
                     clrs_l[ln*3 + 2] = 0xFF;
                     ln += 1;
              }
       }
       for(int x = 0; x < 20; x += 10){   // x = 0, 10
              for(int y = 0; y < 20; y += 10){ // y = 0, 10
                     setVector(lines + ln*6, x, y, 0);
                     setVector(lines + ln*6 + 3, x, y, 10);
                     clrs_l[ln*3] = 0x00;
                     clrs_l[ln*3 + 1] = 0xFF;
                     clrs_l[ln*3 + 2] = 0x00;
                     ln += 1;
              }
       }
       for(int x = 0; x < 20; x += 10){   // x = 0, 10
              for(int z = 0; z < 20; z += 10){ // z = 0, 10
                     setVector(lines + ln*6, x, 0, z);
                     setVector(lines + ln*6 + 3, x, 10, z);
                     clrs_l[ln*3] = 0xFF;
                     clrs_l[ln*3 + 1] = 0x00;
                     clrs_l[ln*3 + 2] = 0x00;
                     ln += 1;
              }
       }




       printf("Allocating space for asset...\n");
       alloc_asset(lines, clrs_l, nLines, 1);

       free(lines);
       free(clrs_l);

       printf("Starting render loop\n");
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
              if(dynamic_render_data->quit == true){break;}
              //printf("Got signal - Rendering...\n");
              // 2) get the dynamic data and load it  + launch all the kernels
              render_and_buffer_2(dynamic_render_data->view_real, dynamic_render_data->offset_real, dynamic_render_data->inst_head);

              ReleaseSemaphore( finish_use_sem, 1, NULL);
              


              // 2) Check if there is a command in the command buffer (from the api)
              ret = WaitForSingleObject(command_pend_sem, 0L); // decrement command

              if(ret != WAIT_TIMEOUT){
                     command_t new_cmd = *cmd;

                     /* do stuff */

                     switch(new_cmd.type){
                            case CMD_ADD_ASSET:

                                   if(!load_triangles_stl(new_cmd.filename, &trigs, (char **)(&clrs), &nTrigs)){break;};
                                   if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); break;}
                                   
                                   cmd->ret_val = alloc_asset(trigs, clrs, nTrigs, 0);
                                   free(new_cmd.filename);
                                   break;
                     }


                     ReleaseSemaphore(command_done_sem, 1, NULL);
              }

              // 5) check for static data updates

       }
       kill_all();
       printf("\nRenderer thread finished.\n");
       return 0;
}



void kill_all(){
       kill();
}
