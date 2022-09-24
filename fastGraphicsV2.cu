#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "gpuController.h"
#include "linalg.h"


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

void make_square(float *triangles, Uint8 *clrs, float *p, float *n, float *d, int clr){
       float size = vecMag(d);
       float v[3] = {0,0,0};
       float temp[3] ={0,0,0};
       
       addVec(p, d, v);
       cross(d, n, temp);
       normalize(temp);
       constMult(size, temp, temp);
       addVec(temp, v, v);
       cpyVec(v, triangles);
       cpyVec(v, triangles + 9);
       subVec(v, temp, v);
       subVec(v, temp, v);
       cpyVec(v, triangles + 3);
       subVec(v, d, v);
       subVec(v, d, v);
       cpyVec(v, triangles + 6);
       cpyVec(v, triangles + 12);
       addVec(v, temp, v);
       addVec(v, temp, v);
       cpyVec(v, triangles + 15);
       for(int i = 0; i < 2; i++){
              clrs[i*3] = 0xAA*(clr == 0 || clr == 3);
              clrs[i*3 + 1] = 0xAA*(clr == 1 || clr == 3);
              clrs[i*3 + 2] = 0xAA*(clr == 2 || clr == 3);
       }


}
// Takes up 18 floats on triangles and 6 BYTEs on clrs
void square_on_axis(float *triangles, Uint8 *clrs, float *v, int clr, float *p){
       float d[3] = {0,0,0};
       float n[3] = {0,0,0};
       cpyVec(v, n);
       if(v[0] == 1 || v[0] == -1){ // x axis
              setVector(d, 0, 0, 1);
       }
       else{
              setVector(d, 1, 0, 0);
       }
       make_square(triangles, clrs, p, n, d, clr);

}

// Takes up 108 floats on triangles and 36 BYTEs on clrs, made of 12 triangles
void create_cube(float *p, float *triangles, Uint8 *clrs, int clr){
       // From p make four squares in each direction
       float v[3] = {0,0,0};
       float t[3] = {0,0,0};
       for(int i = 0; i < 6; i ++){
              v[i%3] = (i < 3)*1 + (i >= 3)*(-1);
              addVec(p, v, t);
              square_on_axis(triangles + 18*i, clrs + 6*i, v, clr, t);
              setVector(v, 0, 0, 0);
       }
}

void create_object_data(int *nTrigs, float **trigs, float **v, Uint8 **clrs){
       
       int situation = 2;
       const float rotationZ[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
       if(situation == 0)
       {
              *nTrigs = 4;
              *trigs = (float *)malloc(*nTrigs*9*sizeof(float));
              *v = (float *)malloc(3*sizeof(float));
              setVector(*v, 3.0f, 3.0f, 3.0f);
              setVector(*trigs, 0, 0, 0);
              setVector(*trigs + 3, 0, 2, 0);
              setVector(*trigs + 6, 0, 1, 1);
              setVector(*trigs + 9, 0, 0, 0);
              setVector(*trigs + 12, 2, 0, 0);
              setVector(*trigs + 15, 1, 0, 1);
              for(int i = 0; i < 6; i++){constMult(-1.0f, *trigs + i*3, *trigs + 18 + i*3);}
              *clrs = (Uint8 *)malloc(*nTrigs*3);
              for(int i = 0; i < 4; i++){
                     (*clrs)[3*i] = 255 * (i == 0 || i == 3);
                     (*clrs)[3*i + 1] = 255 * (i == 1 || i == 3);
                     (*clrs)[3*i + 2] = 255 * (i == 2 || i == 3);
              }
       }
       else if(situation == 1){
              
              *nTrigs = 8;
              *trigs = (float *)malloc(*nTrigs*9*sizeof(float));
              *v = (float *)malloc(3*sizeof(float));
              setVector(*v, 3.0f, 3.0f, 3.0f);



              setVector(*trigs + 0, 1, 1, 1);
              setVector(*trigs + 3, 1, 1, -1);
              setVector(*trigs + 6, 1, -1, 1);
              setVector(*trigs + 9, 1, -1, -1);
              setVector(*trigs + 12, 1, 1, -1);
              setVector(*trigs + 15, 1, -1, 1);
              for(int i = 0; i < 6; i++){constMult(-1.0f, *trigs + i*3, *trigs + 18 + i*3);}

              
              for(int i = 0; i < 12; i++){
                     // Vector 12 + i --> float 3*(12 + i)
                     matix_mult(rotationZ, *trigs + 3*(i), *trigs + 3*(12 + i));
              }


              *clrs = (Uint8 *)malloc(*nTrigs*3);
              for(int i = 0; i < 8; i++){
                     // for each triangle set components RGB as a function of triangle number
                     (*clrs)[3*i + 0] = 255*(i == 2 || i == 3 || i == 5 || i == 4); //* (i == 1 || i == 2 || i == 8 || i == 7);
                     (*clrs)[3*i + 1] = 255*(i == 2 || i == 3 || i == 6 || i == 1);// * (i == 3 || i == 4 || i == 8 || i == 7);
                     (*clrs)[3*i + 2] = 255*(i == 2 || i == 3 || i == 7 || i == 8);// * (i == 5 || i == 6 || i == 8 || i == 7);
              }
       }
       else if(situation == 2){
              *nTrigs = 12*4;
              *trigs = (float *)malloc(*nTrigs*9*sizeof(float));
              *v = (float *)malloc(3*sizeof(float));
              setVector(*v, 10.0f, 10.0f, 10.0f);
              *clrs = (Uint8 *)malloc(*nTrigs*3);
              
              float temp[3] = {0,0,0};
              create_cube(temp, *trigs, *clrs, 0);

              setVector(temp, -3, -3, 0);
              create_cube(temp, *trigs + 108, *clrs + 36, 1);

              setVector(temp, -3, 0, 3);
              create_cube(temp, *trigs + 108*2, *clrs + 36*2, 2);

              setVector(temp, 0, -3, 3);
              create_cube(temp, *trigs + 108*3, *clrs + 36*3, 3);
              
       }

}

void print_object_data(int nTrigs, float *trigs, Uint8 *clrs){
       
       printf("\nPrinting object data for %d triangles...\n", nTrigs);
       for(int i = 0; i < nTrigs; i++){
              printf("Triangle %d:\n\t P1: (%f, %f, %f) \n\t P2: (%f, %f, %f) \n\t P3:(%f, %f, %f) \n\t Color: (R:%d, G:%d, B:%d)\n\n",
                     i, trigs[9*i], trigs[9*i + 1], trigs[9*i + 2], trigs[9*i + 3],
                     trigs[9*i + 4], trigs[9*i + 5], trigs[9*i + 6], trigs[9*i + 7],
                     trigs[9*i + 8], (int)clrs[i*3], (int)clrs[i*3 + 1], (int)clrs[i*3 + 2]);
       }

}

int main(int argc, char **argv){
       /*if(argc != 4){
              printf("Invalid input");
       }
       float a = atof(argv[1]);
       float b = atof(argv[2]);
       float c = atof(argv[3]);*/


       struct windowT *wind;
       int nTrigs = 0;
       float *trigs = NULL;
       float *v = NULL;
       Uint8 *clrs = NULL;
       struct cpu_data *render_data_cpu = NULL;
       struct gpu_data *h_render_data_gpu = NULL;
       struct gpu_data *d_render_data_gpu = NULL;

       if(create_window(&wind, 700, 700) == 0 || wind == NULL){printf("Error initializing window. \n");return 0;}

       create_object_data(&nTrigs, &trigs, &v, &clrs);
       float offset[3] = {0, 0, 0};
       print_object_data(nTrigs, trigs, clrs);
       if(nTrigs == 0 || trigs == NULL || v == NULL || clrs == NULL){printf("Error creating object data. \n"); return 0;}
       //nTrigs = 12;
       //setVector(v, a, b, c);
       
       create_render_data(&render_data_cpu, &h_render_data_gpu, &d_render_data_gpu, wind->imageSurface, nTrigs, trigs, clrs, v, offset);
       if(render_data_cpu == NULL || h_render_data_gpu == NULL || d_render_data_gpu == NULL){printf("Error initializing render data. \n");return 0;}


       Uint32 tt = 0;

       float pi = 3.14159256;
       float r = 10*sqrt(3);
       float phi = pi/4;
       float theta = pi/2;
       float inc = 2*pi*3/500;

       float xinc[3] = {0.1, 0, 0};
       float yinc[3] = {0, 0.1, 0};

       setVector(v, r*cos(theta)*sin(phi), r*sin(phi)*sin(theta), r*cos(phi));

       while(wind->isRunning)
       {
              while(SDL_PollEvent(&(wind->ev)))
              {
                     switch ((wind->ev).type){
                            case SDL_QUIT:
                                   wind->isRunning = false;
                                   break;
                            case SDL_KEYDOWN:
                                   switch ((wind->ev).key.keysym.scancode){
                                          case SDL_SCANCODE_A:
                                                 theta = theta + inc;
                                                 break;
                                          case SDL_SCANCODE_D:
                                                 theta = theta - inc;
                                                 break;
                                          case SDL_SCANCODE_S:
                                                 phi += inc;
                                                 break;
                                          case SDL_SCANCODE_W:
                                                 phi -= inc;
                                                 break;
                                          case SDL_SCANCODE_I:
                                                 addVec(offset, xinc, offset);
                                                 break;
                                          case SDL_SCANCODE_J:
                                                 addVec(offset, yinc, offset);
                                                 break;
                                          case SDL_SCANCODE_K:
                                                 subVec(offset, xinc, offset);
                                                 break;
                                          case SDL_SCANCODE_L:
                                                 subVec(offset, yinc, offset);
                                                 break;
                                   }
                                   break;

                     }
              }

              if((int)SDL_GetTicks() - (int)wind->last_update > 30){
                     
                     tt = SDL_GetTicks();
                     
                     //setVector(v, r*cos(w*tt), r*sin(w*tt), 9);
                     setVector(v, r*cos(theta)*sin(phi), r*sin(phi)*sin(theta), r*cos(phi));
                     // First update the lens on the CPU copy, then upload the changes to the GPU
                     update_lens(render_data_cpu, v, offset);
                     update_GPU_lens(h_render_data_gpu, render_data_cpu);
                     // Now we want to use the GPU to re-write the imageScreen
                     render_and_buffer(d_render_data_gpu, h_render_data_gpu, render_data_cpu);
                     //printf("rendered \n");

                     update_window(wind);
                     wind->last_update = SDL_GetTicks();
                     //printf("Time to render: %d \n", (int)wind->last_update - (int)tt);
              }
       }



       kill_cpu_data(render_data_cpu);
       kill_gpu_data(h_render_data_gpu);
       cudaFree(d_render_data_gpu);
       kill_window(wind);
       
       return 0; 

}