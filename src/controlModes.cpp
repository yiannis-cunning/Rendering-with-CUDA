#include "controlModes.h"
#include <stdio.h>

controller2::controller2(float *view_init, float *offset_init){

       cpyVec(view_init, view);
       cpyVec(offset_init, offset);

       float v[3];
       subVec(view, offset, v);


       mode = CONTROL_FLY_MODE;
       phi = 1.593269;
       theta = 1.278754;
       r = vecMag(view_init);

       inc = 0.05;
       incang = 2*3.14159256*3/500;
}


bool controller2::tick_update(){
       float temp[3], temp2[3];
       float zaxis[3] = {0, 0, 1};
       float xaxis[3] = {1, 0, 0};
       float yaxis[3] = {0, 1, 0};
       bool changed;

       if(press.a == 1 || press.d == 1){
              if(mode == CONTROL_FLY_MODE){
                     cross(zaxis, view, temp);
                     normalize(temp);
                     constMult((inc/(press.j == 1 ? 0.01 : 1))*(press.d-press.a), temp, temp);
              }else{
                     constMult((inc/(press.j == 1 ? 0.01 : 1))*(press.a-press.d), xaxis, temp);
              }
              addVec(temp, offset, offset);
              changed = true;
       }
       if(press.s == 1 || press.w == 1){
              if(mode == CONTROL_FLY_MODE){
                     cpyVec(view, temp);
                     normalize(temp);
                     constMult((inc/(press.j == 1 ? 0.01 : 1))*(press.w-press.s), temp, temp);
              }else {
                     constMult(inc/(press.j == 1 ? 0.01 : 1)*(press.w-press.s), yaxis, temp);
              }
              addVec(temp, offset, offset);
              changed = true;
       }
       if(press.dxmov != 0 || press.dymov != 0){
              phi += incang*(press.dymov/100);
              phi = (phi < 3.14159256-incang) ? phi : phi - incang;
              phi = (phi > incang) ? phi : phi + incang;
              theta += incang*(press.dxmov/100);
              press.dxmov = 0;
              press.dymov = 0;
                            /*phi += incang*(press.i - press.k);
                            phi = (phi < pi-incang) ? phi : phi - incang;
                            phi = (phi > incang) ? phi : phi + incang;
                            theta += incang*(press.l - press.j);*/
              changed = true;
       }
       if(press.space == 1 || press.shift == 1){
              constMult((press.space - press.shift)*inc/(press.j == 1 ? 10 : 1), zaxis, temp);
              addVec(temp, offset, offset);
              changed = true;
       }
       
       setVector(view, r*cos(theta)*sin(phi), r*sin(phi)*sin(theta), r*cos(phi));

       return changed;
}



void controller2::rotate_mode(){
       if(mode == CONTROL_FLY_MODE){
              mode = CONTROL_XYZ_MODE;
       } else if(mode == CONTROL_XYZ_MODE){
              mode = CONTROL_FLY_MODE;
       }
}












// Set the data to initial position.
controller::controller(float *p_0, float *v_0)
{
       fly_data = nullptr;
       jmp_data = nullptr;
       mode = 1;

       if(mode == 1){jmp_data = new jumpMode(p_0, v_0, v, offset);}
       else{fly_data = new flyMode(p_0, v_0, v, offset);}

       send_position();
}

void controller::send_position()
{
       if(mode == 1){jmp_data->send_position();}
       else{fly_data->send_position();}
}

// Update the position based on the keys being pressed right now
bool controller::tick_update()
{
       if(mode == 1){return jmp_data->tick_update(&press);}
       else{return fly_data->tick_update(&press);}
}


jumpMode::jumpMode(float *p_0, float *v_0, float *v_in, float *offset_in){
       phi = 2.1862760354653;
       theta = -1*3*3.14159256/4;
       r = 10*sqrt(3);
       inc = 5; // 25
       incang = 2*3.14159256*3/500;
       velup = 0;
       tLast = 0;
       cpyVec(p_0, pos);
       cpyVec(v_0, view);
       v = v_in;
       offset = offset_in;
}

void jumpMode::send_position(){
       setVector(view, r*cos(theta)*sin(phi), r*sin(phi)*sin(theta), r*cos(phi));
       addVec(view, pos, offset);
       constMult(-1, view, v);
}

bool jumpMode::tick_update(struct pressing *press){
       float temp[3], temp2[3];
       float zaxis[3] = {0, 0, 1};
       float xaxis[3] = {1, 0, 0};
       float yaxis[3] = {0, 1, 0};
       bool changed;
       if(press->a == 1 || press->d == 1){
              cpyVec(view, temp);
              temp[2] = 0;
              cross(temp, zaxis, temp2);
              normalize(temp2);
              constMult((inc/(press->j == 1 ? 10 : 1))*(press->a-press->d), temp2, temp);
              constMult((inc/(press->j == 1 ? 10 : 1))*(press->a-press->d), xaxis, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       if(press->s == 1 || press->w == 1){
              cpyVec(view, temp);
              temp[2] = 0;
              normalize(temp);
              constMult(inc/(press->j == 1 ? 10 : 1)*(press->w-press->s), temp, temp);
              constMult(inc/(press->j == 1 ? 10 : 1)*(press->w-press->s), yaxis, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       if(press->dxmov != 0 || press->dymov != 0){
              phi += incang*(press->dymov/100);
              phi = (phi < 3.14159256-incang) ? phi : phi - incang;
              phi = (phi > incang) ? phi : phi + incang;
              theta += incang*(press->dxmov/100);
              press->dxmov = 0;
              press->dymov = 0;
                            /*phi += incang*(press->i - press->k);
                            phi = (phi < pi-incang) ? phi : phi - incang;
                            phi = (phi > incang) ? phi : phi + incang;
                            theta += incang*(press->l - press->j);*/
              changed = true;
       }
       if(press->space == 1 || press->shift == 1){
              constMult((press->space - press->shift)*inc/(press->j == 1 ? 10 : 1), zaxis, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       

       send_position();
       return changed;
}

flyMode::flyMode(float *p_0, float *v_0, float *v_in, float *offset_in){
       phi = 2.1862760354653;
       theta = -1*3*3.14159256/4;
       r = 10*sqrt(3);
       inc = 100;
       incang = 2*3.14159256*3/500;
       cpyVec(p_0, pos);
       cpyVec(v_0, view);
       v = v_in;
       offset = offset_in;
}

void flyMode::send_position(){
       setVector(view, r*cos(theta)*sin(phi), r*sin(phi)*sin(theta), r*cos(phi));
       addVec(view, pos, offset);
       constMult(-1, view, v);
}

bool flyMode::tick_update(struct pressing *press){
       float temp[3], temp2[3];
       float zaxis[3] = {0, 0, 1};
       bool changed;
       if(press->a == 1 || press->d == 1){
              cpyVec(view, temp);
              temp[2] = 0;
              cross(temp, zaxis, temp2);
              normalize(temp2);
              constMult(inc*(press->a-press->d), temp2, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       if(press->s == 1 || press->w == 1){
              cpyVec(view, temp);
              temp[2] = 0;
              normalize(temp);
              constMult(inc*(press->w-press->s), temp, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       if(press->dxmov != 0 || press->dymov != 0){
              phi += incang*(press->dymov/100);
              phi = (phi < 3.14159256-incang) ? phi : phi - incang;
              phi = (phi > incang) ? phi : phi + incang;
              theta += incang*(press->dxmov/100);
              press->dxmov = 0;
              press->dymov = 0;
                            /*phi += incang*(press->i - press->k);
                            phi = (phi < pi-incang) ? phi : phi - incang;
                            phi = (phi > incang) ? phi : phi + incang;
                            theta += incang*(press->l - press->j);*/
              changed = true;
       }
       if(press->space == 1 || press->shift == 1){
              constMult((press->space - press->shift)*inc, zaxis, temp);
              addVec(temp, pos, pos);
              changed = true;
       }
       send_position();
       return changed;
}