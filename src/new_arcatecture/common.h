#pragma once
#include <C:\clibs\SDL2\include\SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>



void passert(bool tf, const char *msg);

typedef struct instance_t {
       float offset[3];
       bool is_visible;
       
       int asset_id;
       int type;

       instance_t *next;

} instance_t;



typedef struct dynamic_render_data_t{
       SDL_Surface *imageSurface;

       int nInstances;
       instance_t *inst_head;

       float view[3];
       float offset[3];
       // + other pre-calculated values

       float view_real[3];
       float offset_real[3];


} dynamic_render_data_t;


typedef struct thread_args_t{
       HANDLE sem1;
       HANDLE sem2;
       HANDLE mutex1;

       void *p1;
       void *wind;
} thread_args_t;


