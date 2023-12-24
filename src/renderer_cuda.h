
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
       
int alloc_asset(float *trigs, uint8_t *colors, uint32_t nTrigs, int type);

int kill();


void *get_device_trig_pointer();

void *get_static_render_data_p();

void *get_dynamic_render_data_p();

//int update_dynamic_data(float *view, float *offset, instance_t *inst_data, int n_instances);



int render_and_buffer_2(float *view, float *offset, instance_t *instances_arr);


int rotate_asset(int asset_id, float *matrix);