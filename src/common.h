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

       bool quit;


} dynamic_render_data_t;


enum command_type_t {CMD_ADD_ASSET, CMD_ADD_INSTANCE};

typedef struct command_t{
       command_type_t type;

       char *filename;
       float pos[3];

       int ret_val;
} command_t;


typedef struct command_queue_t{
       HANDLE n_pending_commands;
       HANDLE n_empty_spots;
       command_t cmds[30];

       int fill_ptr;
       int take_ptr;

} command_queue_t;


typedef struct thread_args_t{
       HANDLE sem1;
       HANDLE sem2;
       HANDLE mutex1;

       HANDLE pending_cmd;
       HANDLE done_cmd;
       command_t *command;

       void *p1;
       void *wind;

       command_queue_t *command_queue;

} thread_args_t;




