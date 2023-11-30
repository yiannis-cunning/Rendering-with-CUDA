
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