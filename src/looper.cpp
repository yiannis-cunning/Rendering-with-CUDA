#include "looper.h"
#include "controlModes.h"
#include "sdl_window.h"
#include "common.h"

void kill();

int loop_back();

static dynamic_render_data_t *render_data = NULL;
static dynamic_render_data_t local_copy_render_data;
static HANDLE start_frame_sem;
static HANDLE render_done_sem;
static HANDLE dynamic_data_mutex;
//static dynamic_render_data_t * shared_dynamic_data = NULL;

static float pos[3] = {10, 10, 0};
static float view[3] = {-10, -10, -10};
static controller c(pos, view);
static windowT *wind = NULL;


void add_instance(float *pos_in, int asset_id, int type){
       instance_t *new_inst = (instance_t *)calloc(sizeof(instance_t), 1);
       new_inst->asset_id = asset_id;
       new_inst->is_visible = 1;
       new_inst->type = type;
       new_inst->next = NULL;
       cpyVec(pos_in, new_inst->offset);
       local_copy_render_data.nInstances += 1;

       instance_t *head = local_copy_render_data.inst_head;
       if(head == NULL){
              local_copy_render_data.inst_head = new_inst;
              return;
       } 
       while(head->next != NULL){
              head = head->next;
       }
       head->next = new_inst;


}




DWORD WINAPI looper_init(LPVOID param){

       // 1) Get args that link to cudacore sender
       thread_args_t *args = (thread_args_t *)param;

       start_frame_sem = args->sem1;
       render_done_sem = args->sem2;
       dynamic_data_mutex = args->mutex1;
       render_data = (dynamic_render_data_t *)args->p1;


       // 2) Start the SDL window and set up the start recivein I/O messages
       passert(c.v != NULL && c.offset != NULL,"Error initilizing position data." );

       passert(create_window(&wind, 900, 900) != 0, "Error initializing window.");
       SDL_ShowCursor(SDL_DISABLE);
       SDL_SetRelativeMouseMode(SDL_TRUE);

       local_copy_render_data.imageSurface = wind->imageSurface;
       local_copy_render_data.quit = false;
       
       float pos[3] = {0,0,0};
       printf("making instances");
       /*
       for(int y = 0; y < 1; y++){
              for(int z = 0; z < 1; z++){
                     for(int x = 0; x < 1; x++){
                            setVector(pos, x*5, 5*y + 5, 5*z);
                            add_instance(pos, 1, 0);
                     }
              }
       }*/
       setVector(pos, 0, 0, 5);
       add_instance(pos, 1, 0);
       
       instance_t *head = local_copy_render_data.inst_head;
       instance_t *new_inst = (instance_t *)calloc(sizeof(instance_t), 1);
       while(head->next != NULL){
              head = head->next;
       }
       head->next = new_inst;
       new_inst->next = NULL;
       new_inst->type = 1;
       new_inst->is_visible = 1;
       new_inst->asset_id = 2;
       setVector(new_inst->offset, 0, 0, 0);



       //setVector(pos, 0, -2000, 0);
       //add_instance(pos, 1);

       // 3) enter msgloop
       printf("Created windows in game loop thread - starting msg loop\n");
       
       loop_back();

       return 1;
}


void kill(){
       kill_window(wind);


       instance_t *head = local_copy_render_data.inst_head;
       instance_t *prev;
       while(head != NULL){
              prev = head;
              head = head->next;
              free(prev);
       }
}

// -500, -1000, 3385 

// msg, frame, and control loops
int loop_back(){

       DWORD ret;

       int paused = 0;
       float vec1[3] = {-10, -10, -10};
       float vec2[3] = {0, 0, 0};
       controller2 cntr2(vec1, vec2);


       // Start by sending initial render data to renderer

       cpyVec(cntr2.offset, local_copy_render_data.offset_real);
       cpyVec(cntr2.view, local_copy_render_data.view_real);

       memcpy(render_data, &local_copy_render_data, sizeof(dynamic_render_data_t));



       // Allow rendered to init and send first frame
       ReleaseSemaphore( start_frame_sem, 1, NULL);


       unsigned int t_start, t_end;
       while(wind->isRunning)
       {
              
              // 1) Input check
              while(SDL_PollEvent(&(wind->ev))){
                     switch ((wind->ev).type){
                            case SDL_QUIT:
                                   wind->isRunning = false;
                                   break;
                                   
                            case SDL_KEYDOWN:
                                   switch ((wind->ev).key.keysym.scancode){
                                          case SDL_SCANCODE_A:
                                                 c.press.a = 1;
                                                 break;
                                          case SDL_SCANCODE_S:
                                                 c.press.s = 1;
                                                 break;
                                          case SDL_SCANCODE_D:
                                                 c.press.d = 1;
                                                 break;
                                          case SDL_SCANCODE_W:
                                                 c.press.w = 1;
                                                 break;
                                          case SDL_SCANCODE_K:
                                                 c.press.i = 1;
                                                 break;
                                          case SDL_SCANCODE_J:
                                                 c.press.j = 1;
                                                 break;
                                          case SDL_SCANCODE_I:
                                                 c.press.k = 1;
                                                 break;
                                          case SDL_SCANCODE_L:
                                                 c.press.l = 1;
                                                 break;
                                          case SDL_SCANCODE_LSHIFT:
                                                 c.press.shift = 1;
                                                 break;
                                          case SDL_SCANCODE_SPACE:
                                                 c.press.space = 1;
                                                 break;
                                          case SDL_SCANCODE_R:
                                                 c.press.r = 1;
                                                 break;
                                          case SDL_SCANCODE_ESCAPE:
                                                 wind->isRunning = false;
                                                 break;
                                          case SDL_SCANCODE_P:
                                                 if(!paused){
                                                        SDL_ShowCursor(SDL_ENABLE);
                                                        SDL_SetRelativeMouseMode(SDL_FALSE);
                                                        paused = 1;
                                                 } else{
                                                        SDL_ShowCursor(SDL_DISABLE);
                                                        SDL_SetRelativeMouseMode(SDL_TRUE);
                                                        paused = 0;  
                                                 }
                                                 break;
                                   }
                                   break;
                            case SDL_KEYUP:
                                   switch ((wind->ev).key.keysym.scancode){
                                          case SDL_SCANCODE_A:
                                                 c.press.a = 0;
                                                 break;
                                          case SDL_SCANCODE_S:
                                                 c.press.s = 0;
                                                 break;
                                          case SDL_SCANCODE_D:
                                                 c.press.d = 0;
                                                 break;
                                          case SDL_SCANCODE_W:
                                                 c.press.w = 0;
                                                 break;
                                          case SDL_SCANCODE_K:
                                                 c.press.i = 0;
                                                 break;
                                          case SDL_SCANCODE_J:
                                                 c.press.j = 0;
                                                 break;
                                          case SDL_SCANCODE_I:
                                                 c.press.k = 0;
                                                 break;
                                          case SDL_SCANCODE_L:
                                                 c.press.l = 0;
                                                 break;
                                          case SDL_SCANCODE_LSHIFT:
                                                 c.press.shift = 0;
                                                 break;
                                          case SDL_SCANCODE_SPACE:
                                                 c.press.space = 0;
                                                 break;
                                          case SDL_SCANCODE_R:
                                                 c.press.r = 0;
                                                 break;
                                          case SDL_SCANCODE_M:
                                                 cntr2.rotate_mode();
                                                 break;
                                   }
                                   break;
                            case SDL_MOUSEMOTION:
                                   c.press.dxmov += (wind->ev).motion.xrel;
                                   c.press.dymov += (wind->ev).motion.yrel;
                                   break;

                     }
              }
              
              if(wind->isRunning == false){break;}
              
              // Update screen
              if((int)SDL_GetTicks() - (int)wind->last_update > 30){
                     memcpy(&cntr2.press, &c.press, sizeof(pressing));
                     bool chnge2 = cntr2.tick_update();
                     bool changed = c.tick_update();

                     //printf("Checking if changed...\n");
                     if(changed){
                            // 1) update state/position/dynamic data varibles
                            //printf("Changed\n");
                            cpyVec(cntr2.offset, local_copy_render_data.offset_real);
                            cpyVec(cntr2.view, local_copy_render_data.view_real);

                            fflush(stdout);
                            //printf("OFFSET: %f, %f, %f \t", cntr2.offset[0], cntr2.offset[1], cntr2.offset[2]);
                            //printf("VIEW:   %f, %f, %f", cntr2.view[0], cntr2.view[1], cntr2.view[2]);

                            // 2) check if there is a frame to buffer -> if not conitnue
                            ret = WaitForSingleObject(render_done_sem, 0L); // decrement command
                            if(ret != WAIT_TIMEOUT ){
                                   update_window(wind);
                                   memcpy(render_data, &local_copy_render_data, sizeof(dynamic_render_data_t));
                                   ReleaseSemaphore( 
                                   start_frame_sem,  // handle to semaphore
                                   1,            // increase count by one
                                   NULL);       // not interested in previous count
                            }

                            wind->last_update = SDL_GetTicks();
                            /*
                            t_start = SDL_GetTicks();
                            t_end = SDL_GetTicks();
                            printf("Time to update renderData: %d\n", t_end-t_start);
                            t_start = SDL_GetTicks();
                            render_and_buffer(d_render_data_gpu, h_render_data_gpu, render_data_cpu, 2, 2);
                            t_end = SDL_GetTicks();
                            printf("Time to render: %d\n", t_end-t_start);
                            t_start = SDL_GetTicks();
                            update_window(wind);
                            t_end = SDL_GetTicks();
                            printf("Time to update window: %d\n", t_end-t_start);
                            wind->last_update = SDL_GetTicks();*/ 
                     }
              }
       }

       ret = WaitForSingleObject(render_done_sem, INFINITE);
       local_copy_render_data.quit = true;
       memcpy(render_data, &local_copy_render_data, sizeof(dynamic_render_data_t));
       ReleaseSemaphore( start_frame_sem, 1, NULL);

       kill();
       printf("\nLoper thread finished\n");
       
       return 0; 

}



/*
CONTROLS
       : j = speed up
       : p = pause mouse mode
       : asdw/mouse - normal
       : m = rotate mode



*/