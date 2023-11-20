#include "looper.h"
#include "controlModes.h"
#include "sdl_window.h"
#include "common.h"

void kill();

int loop_back();

static dynamic_render_data_t *render_data = NULL;
static HANDLE start_frame_sem;
static HANDLE render_done_sem;
//static dynamic_render_data_t * shared_dynamic_data = NULL;

static float pos[3] = {10, 10, 0};
static float view[3] = {-10, -10, -10};
static controller c(pos, view);
static windowT *wind = NULL;


DWORD WINAPI looper_init(LPVOID param){

       // 1) Get args that link to cudacore sender
       thread_args_t *args = (thread_args_t *)param;

       start_frame_sem = args->sem1;
       render_done_sem = args->sem2;
       

       // 2) Start the SDL window and set up the start recivein I/O messages
       passert(c.v != NULL && c.offset != NULL,"Error initilizing position data." );
       

       passert(create_window(&wind, 900, 900) != 0, "Error initializing window.");
       SDL_ShowCursor(SDL_DISABLE);
       SDL_SetRelativeMouseMode(SDL_TRUE);

       


       //render_data = (dynamic_render_data_t *)calloc(sizeof(dynamic_render_data_t), 1);
       render_data = (dynamic_render_data_t *)args->p1;
       render_data->imageSurface = wind->imageSurface;
       cpyVec(pos, render_data->offset);
       cpyVec(view, render_data->view);

       printf("Sending the init command to other thread\n");
       ReleaseSemaphore( 
       start_frame_sem,  // handle to semaphore
       1,     // increase count by one
       NULL);       // not interested in previous count
       
       // wait for init to finish
       WaitForSingleObject(render_done_sem, INFINITE); // decrement command
       


       // 3) enter msgloop
       printf("Created windows in game loop thread - starting msg loop\n");
       
       loop_back();

       return 1;
}


/*
       passert(c.v != NULL && c.offset != NULL,"Error initilizing position data." );

       passert(create_window(&wind, 900, 900) != 0, "Error initializing window.");
       SDL_ShowCursor(SDL_DISABLE);
       SDL_SetRelativeMouseMode(SDL_TRUE);

       render_data = (dynamic_render_data_t *)calloc(sizeof(dynamic_render_data_t), 1);
       passert(render_data != NULL, "Error allocating render data.");
       render_data->head = NULL;
       render_data->nInstances = 0;
       cpyVec(pos, render_data->offset);
       cpyVec(view, render_data->view);


       return loop_back();

}

*/
void kill(){
       kill_window(wind);
}



// msg, frame, and control loops
int loop_back(){

       DWORD ret;
       int first = 1;

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
                     bool changed = c.tick_update();
                     if(changed){
                            // 1) set the dynamic render data
                            cpyVec(c.offset, render_data->offset);
                            cpyVec(c.v, render_data->view);

                            printf("Inputs have changed game state -> starting render frame\n");
                            // 2) post semaphore to start job
                            ReleaseSemaphore( 
                                   start_frame_sem,  // handle to semaphore
                                   1,            // increase count by one
                                   NULL);       // not interested in previous count
                            
                            WaitForSingleObject(render_done_sem, INFINITE); // decrement command
                            first = 0;
                            update_window(wind);
                            // 3) wait on semaphore for job to end
                            //ret = WaitForSingleObject(render_done_sem, INFINITE);
                            //update_window(wind);
                            //wind->last_update = SDL_GetTicks();
                            /*
                            t_start = SDL_GetTicks();
                            update_lens(render_data_cpu, c.v, c.offset);
                            update_GPU_lens(h_render_data_gpu, render_data_cpu);
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


       kill();
       
       return 0; 

}