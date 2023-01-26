#include "C:\Users\yiann\Desktop\sdlTest\renderingv2\linalg.h"
#include <math.h>
#include <stdlib.h>

struct pressing{
       bool w, a, s, d, i, j, k, l, shift, space, r;
       float dxmov, dymov;
       pressing(){
              w = 0; a = 0; s = 0; d = 0;
              i = 0; j = 0; k = 0; l = 0;
              shift = 0; space = 0; r = 0;
              dxmov = 0; dymov = 0;
       }
};


class flyMode{
       float pos[3], view[3], phi, theta, r, inc, incang, *v, *offset;
public:
       flyMode(float *p_0, float *v_0, float *v_in, float *offset_in);
       void send_position();
       bool tick_update(struct pressing *press);
};


class jumpMode{
       float *v, *offset, pos[3], view[3], phi, theta, r, inc, incang, velup, accel;
       unsigned int tLast;
public:
       jumpMode(float *p_0, float *v_0, float *v_in, float *offset_in);
       void send_position();
       bool tick_update(struct pressing *press);
};

class controller{
       struct flyMode *fly_data;
       jumpMode *jmp_data;
       int mode;
public:
       struct pressing press;
       float v[3];
       float offset[3];
       
       controller(float *p_0, float *v_0);
       void send_position();
       bool tick_update();
};
