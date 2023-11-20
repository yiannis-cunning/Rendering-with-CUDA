#include <C:\clibs\SDL2\include\SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>




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


int create_window(struct windowT **wind, int h, int w);

void update_window(struct windowT *wind);

void kill_window(struct windowT *wind);