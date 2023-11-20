#include <C:\clibs\SDL2\include\SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>



void passert(bool tf, const char *msg);


typedef struct instance_t{
       int asset_id;
       float pos[3];
       bool visible;
}instance_t;



typedef struct dynamic_render_data_t{
       SDL_Surface *imageSurface;

       int nInstances;
       instance_t *head;

       float view[3];
       float offset[3];
       // + other pre-calculated values


} dynamic_render_data_t;


typedef struct thread_args_t{
       HANDLE sem1;
       HANDLE sem2;
       void *p1;
       void *wind;
} thread_args_t;