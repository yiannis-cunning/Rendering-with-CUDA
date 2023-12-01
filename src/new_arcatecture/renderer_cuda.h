
#include <C:\clibs\SDL2\include\SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
#include <stddef.h>

#include "linalg.h"
#include "common.h"



int init(SDL_Surface *image);
       
int alloc_asset(float *trigs, uint8_t *colors, uint32_t nTrigs);

int kill();


void *get_device_trig_pointer();

void *get_static_render_data_p();

void *get_dynamic_render_data_p();

int update_dynamic_data(float *view, float *offset, instance_t *inst_data, int n_instances);



typedef struct instance2_t {
       int    asset_id;
       float  *buffer_loc;

       float offset[3];
} instance2_t;




typedef struct asset_t{    // array struct
       uint32_t      nTrigs;
       float         *trigs;
       uint8_t       *colors;
} asset_t;



typedef struct dynamic_render_data2_t{
       float offset[3];
       float view[3];

       instance2_t *instances_arr;
} dynamic_render_data2_t;


typedef struct static_render_data_t{
       // output frame buffer + info
       uint8_t       *pixels_arr;
       uint32_t      *depthScreen_arr;
       uint8_t       BPP;
       uint32_t      pitch;
       uint32_t      w;
       uint32_t      h;

       // Stored assets
       asset_t      *asset_pointer_arr;                           // Variable array -> needs to be resized on addtions

} static_render_data_t;