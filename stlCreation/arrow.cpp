
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
#include <stdint.h>
#include "linalg.h"


#define PI 3.141592654

typedef struct stl_t{
       float *trigs;
       uint32_t nTrigs;

       uint32_t n_allocated;
}stl_t;

static stl_t arrow;

void passert(bool cond, const char *msg){
       if(!cond){
              printf("ERROR: %s\n", msg);
              exit(1);
       }
}



void check_resize(uint32_t n_toadd){
       if(arrow.nTrigs + n_toadd > arrow.n_allocated){
              
              float *new_alloc = (float *)calloc(sizeof(float)*9, (arrow.nTrigs + n_toadd)*2);
              if(new_alloc == NULL){
                     printf("ERROR allocating more space\n");
                     exit(1);
              }
              memcpy(new_alloc, arrow.trigs, arrow.nTrigs*sizeof(float)*9);

              if(arrow.trigs != NULL){
                     free(arrow.trigs);
              }
              arrow.trigs = new_alloc;
              arrow.n_allocated = (arrow.nTrigs + n_toadd)*2;
       }
}



void push_trig(float *veca, float *vecb, float *vecc){
       check_resize(1);
       cpyVec(veca, arrow.trigs + arrow.nTrigs*9);
       cpyVec(vecb, arrow.trigs + arrow.nTrigs*9 + 3);
       cpyVec(vecc, arrow.trigs + arrow.nTrigs*9 + 6);
       arrow.nTrigs += 1;

}




int add_cylinder(float *base_center_pos, float *direction, float radius, uint32_t n_faces, float height){
       if(n_faces < 3 || vecMag(direction) == 0){return - 1;}

       float base_start[3];
       float base_end[3];
       float top_start[3];
       float top_end[3];
       float zero[3] = {0,0,0};
       float top_mid[3] = {0,0,0};

       printf("Making all the trigs...\n");
       float inc = 2*PI/n_faces;
       float theta = 0;
       for(int i = 0; i < n_faces; i += 1){
              theta = inc*i;

              // Base circle
              setVector(base_start, radius*cos(theta), radius*sin(theta), 0);
              setVector(base_end, radius*cos(theta + inc), radius*sin(theta + inc), 0);
              setVector(top_start, radius*cos(theta), radius*sin(theta), height);
              setVector(top_end, radius*cos(theta + inc), radius*sin(theta + inc), height);

              setVector(top_end, 0, 0, height);

              // Base circle
              push_trig(zero, base_start, base_end);
              // top circle
              push_trig(zero, top_start, top_end);
              // wall
              push_trig(base_start, base_end, top_end);
              push_trig(top_start, top_end, base_start);

       }
       // Apply rotations and offset
       return 1;
       printf("Applying fix-ups...\n");
       float direct_norm[3];
       float xaxis[3] = {1, 0, 0};
       float yaxis[3] = {0, 1, 0};
       float v1[3];
       float v2[3];
       
       cpyVec(direction, direct_norm);
       normalize(direct_norm);
       
       cross(direct_norm, xaxis, v1);
       cross(direct_norm, yaxis, v2);
       if(vecMag(v2) > vecMag(v1)){
              cpyVec(v2, v1);
       }
       cross(v1, direct_norm, v2);
       normalize(v1);
       normalize(v2);

       float *trigs = arrow.trigs - n_faces*4*9;
       for(int i = 0; i < n_faces*4*9; i += 3){
              // A = [v1], [v2], [d]
              //trigs[i] = base_center_pos[0] + (v1[0]*trigs[i] + v2[0]*trigs[i + 1] + direct_norm[0]*trigs[i + 2]);
              //trigs[i + 1] = base_center_pos[1] + (v1[1]*trigs[i] + v2[1]*trigs[i + 1] + direct_norm[1]*trigs[i + 2]);
              //trigs[i + 2] = base_center_pos[2] + (v1[2]*trigs[i] + v2[2]*trigs[i + 1] + direct_norm[2]*trigs[i + 2]);
       }
       return 0;
}




int main(int argc, char *argv[]){
       if(argc != 2){
              printf("ERROR: Invalid args\n");
              return -1;
       }

       int fd = _open(argv[1], _O_CREAT | _O_WRONLY);
       if(fd == -1){
              printf("ERROR opening file %s\n", argv[1]);
       }
       printf("Making stl file at: %s \n", argv[1]);

       
       arrow.n_allocated = 100;
       arrow.nTrigs = 0;
       arrow.trigs = (float *)calloc(sizeof(float)*9, arrow.n_allocated);

       float pos[3] = {0,0,0};
       float direct[3] = {0.5, 0.5, 0.5};
       float radius = 10;
       uint32_t faces = 20;
       float height = 30;
       add_cylinder(pos, direct, radius, faces, height);

       printf("Number of trigs: %d\n", arrow.nTrigs);

       // 80 bytes of garbage followed by n_trigs
       char buffer = 0;
       // then uint32_t amt, then 12bytes, trig(9 floats) then 2 bytes
       int n = 0;
       for(int i = 0; i < 80; i++){
              n = _write(fd, &buffer, 1);
              passert(n == 1, "Writing stl header");
       }

       n = _write(fd, &(arrow.nTrigs), sizeof(uint32_t));
       passert(n == sizeof(uint32_t), "Writing ntrigs");

       for(int i = 0; i < arrow.nTrigs; i += 1){
              for(int j = 0; j < 12; j += 1){
                     n = _write(fd, &buffer, 1);
                     passert(n == 1, "Writing nothing");
              }
              for(int q = 0; q < 9; q += 1){
                     n = _write(fd, (arrow.trigs + i*9 + q), sizeof(float));
                     passert(n == sizeof(float), "Writing trig");
              }
              for(int j = 0; j < 2; j += 1){
                     n = _write(fd, &buffer, 1);
                     passert(n == 1, "Writing nothing");
              }


       }
       _close(fd);
}