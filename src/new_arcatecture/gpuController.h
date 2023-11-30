#pragma once

#include <C:\clibs\SDL2\include\SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "linalg.h"


struct cpu_data{
       // Object data
       int nTrigs;
       float *trigs;
       Uint8 *colors;

       // Pixel data
       Uint8 *pixels_arr;
       Uint8 BPP;
       int pitch;
       int h;
       int w;
       int pix_arr_size;
       int depth_arr_size;

       // Equation data
       float *v;
       float *hx;
       float *hy;
       float *c1;
       float *mag;
       float *offset;
       float *a;

       float view_real[3];
       float offset_real[3];
       float hx_real[3];
       float hy_real[3];
       float magv_real;
};


struct gpu_data{
       // List of all trigs in play
       // identical list of cordified triangles
       // buffered frame of image 
       // block filled with calcualtion constants
       int d_nTrigs;
       float *d_trigs;
       float *d_cords_arr;
       Uint8 *d_colors;

       Uint8 *d_pixels_arr;
       Uint8 d_BPP;
       int d_pitch;
       int d_w;
       int d_h;

       float *d_v;
       float *d_hx;
       float *d_hy;
       float *d_c1;
       float *d_mag;
       float *d_offset;

       float view_real[3];
       float offset_real[3];
       float hx_real[3];
       float hy_real[3];
       float magv_real;

       
       Uint32 *d_depthScreen;
};


int create_render_data(struct cpu_data **cpu_data, struct gpu_data **h_gpu_data, struct gpu_data **d_gpu_data, SDL_Surface *image, int nTrigs, const float *trigs, const Uint8 *clrs, float *v, float *o, float *v_real, float *o_real);

// pointer to cpu_data, pointer of image surface, pointer to trig data, amount of trigs(nTrigs = 3*nFloat3's = 9*nfloats)
int init_cpu_dat(struct cpu_data *dat, SDL_Surface *image, int nTrigs, const float *trigs, const Uint8 *clrs, float *v, float *o, float *v_real, float *o_real);

void update_lens(struct cpu_data *dat, float *v, float *o, float *view_real_in, float *offset_real_in);

void kill_cpu_data(struct cpu_data *dat);


// GPU data stuff
int init_gpu_dat(struct gpu_data *d_dat, struct gpu_data *h_dat, struct cpu_data *cpu_dat);

void kill_gpu_data(struct gpu_data *h_gDat);

void update_GPU_lens(struct gpu_data *h_dat, struct cpu_data *cdat, struct gpu_data *d_dat);

void render_and_buffer(struct gpu_data *d_gDat, struct gpu_data *h_gDat, struct cpu_data *cDat, int a, int b, void *gpu_data);

