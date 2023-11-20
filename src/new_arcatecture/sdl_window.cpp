#include "common.h"
#include "sdl_window.h"





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
                            printf("Image loading error - test.bmp: %s", SDL_GetError());
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