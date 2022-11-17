#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "gpuController.h"
#include "linalg.h"
#include "Control/controlModes.h"
#include "File_Conversion/readSTL.h"

struct windowT{
       SDL_Window *window;
       SDL_Surface *windowSurface;
       SDL_Surface *imageSurface;

       bool isRunning;
       SDL_Event ev;
       Uint32 last_update;

       int height;
       int width;

       Uint32 lastMove;
};

int create_window(struct windowT **wind, int h, int w){
       SDL_Window *window = NULL;
       SDL_Surface *windowSurface = NULL;
       SDL_Surface *imageSurface = NULL;

       // Initialize window, get pinter to surfaceBitmap and holdBitmap and set initial image
       // Initialize window with type
       if(SDL_Init(SDL_INIT_VIDEO) < 0)
       {
              printf("Video initialization error %s", SDL_GetError());
              return 0;
       }
       else
       {
              // create window with paramaters and title and return the pointer to the window
             window = SDL_CreateWindow("Title", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_SHOWN);
              if(window == NULL)
              {
                     printf("Window creation error: %s", SDL_GetError());
                     return 0;
              }
              else
              {
                     // get a pointer to the surface and load in a image to image surface
                     windowSurface = SDL_GetWindowSurface(window);
                     imageSurface = SDL_LoadBMP("test.bmp");


                     if(imageSurface == NULL)
                     {
                            printf("Image loading error: %s", SDL_GetError());
                            return 0;
                     }
                     else
                     {
                            // Add the saved image to the screen
                            SDL_BlitSurface(imageSurface, NULL, windowSurface, NULL);

                            // Update the window
                            SDL_UpdateWindowSurface(window);
                            //SDL_Delay(2000);
                     }
              }
       }

       *wind = (struct windowT *)malloc(sizeof(struct windowT));
       if(*wind == NULL){printf("Error creating window structure."); return 0;}
       (*wind)->window = window;
       (*wind)->imageSurface = imageSurface;
       (*wind)->windowSurface = windowSurface;
       (*wind)->isRunning = true;
       (*wind)->last_update = 0;
       (*wind)->height = h;
       (*wind)->width = w;
       (*wind)->last_update = 0;
       return 1;
}

void update_window(struct windowT *wind){
       SDL_BlitSurface(wind->imageSurface, NULL, wind->windowSurface, NULL);
       SDL_UpdateWindowSurface(wind->window);
       wind->last_update = SDL_GetTicks();
}

void kill_window(struct windowT *wind){
       SDL_FreeSurface(wind->imageSurface);
       wind->imageSurface = NULL;
       wind->windowSurface = NULL;
       SDL_DestroyWindow(wind->window);
       wind->window = NULL;
       SDL_Quit();
       free(wind);
}

int main(int argc, char **argv){


       // INIT
       struct windowT *wind = NULL;
       int nTrigs = 0;
       float *trigs = NULL;
       Uint8 *clrs = NULL;
       struct cpu_data *render_data_cpu = NULL;
       struct gpu_data *h_render_data_gpu = NULL;
       struct gpu_data *d_render_data_gpu = NULL;

       // Initiate window
       if(create_window(&wind, 900, 900) == 0){printf("Error initializing window. \n");return 0;}
       SDL_ShowCursor(SDL_DISABLE);
       SDL_SetRelativeMouseMode(SDL_TRUE);
       
       // Create object data
       char *filename = (char *)"blaarkop.stl";
       if(!load_triangles_stl(filename, &trigs, (char **)(&clrs), &nTrigs)){return 0;};
       if(nTrigs == 0 || trigs == NULL || clrs == NULL){printf("Error creating object data. \n"); return 0;}


       // Create controller data
       float pos[3] = {10, 10, 0};
       float view[3] = {-10, -10, -10};
       controller c(pos, view);
       if(c.v == NULL || c.offset == NULL){printf("Error initilizing position data. \n"); return 0;}

       // Create render data and links to window
       if(create_render_data(&render_data_cpu, &h_render_data_gpu, &d_render_data_gpu, wind->imageSurface, nTrigs, trigs, clrs, c.v, c.offset) == 0){printf("Error init render data. \n");return 0;}
       free(trigs);
       free(clrs);
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
                            wind->last_update = SDL_GetTicks();
                     }
              }
       }



       kill_cpu_data(render_data_cpu);
       kill_gpu_data(h_render_data_gpu);
       cudaFree(d_render_data_gpu);
       kill_window(wind);
       
       return 0; 

}