#include "linalg.h"
#include <math.h>

void cpyVec(float *v, float *ans)
{
       for(int i = 0; i <3; i++){
              ans[i] = v[i];
       }
}


void cross(float *v, float *w, float *ans)
{
       ans[0] = v[1] * w[2] - w[1] * v[2];
       ans[1] = v[2] * w[0] - v[0] * w[2];
       ans[2] = v[0] * w[1] - w[0] * v[1];
}

float dotProduct(float *v, float *w)
{
    float ans = 0;
    for(int i = 0; i < 3; i++){
        ans += v[i]*w[i];
    }
    return ans;
}

void addVec(float *v, float *w, float *ans)
{
       for(int i = 0; i < 3; i++){
              ans[i] = v[i] + w[i];
       }
}

void subVec(float *v, float *w, float *ans)
{
       for(int i = 0; i < 3; i++){
              ans[i] = v[i] - w[i];
       }
}

void constMult(float c, float *v, float *ans)
{
       for(int i = 0; i < 3; i++){
              ans[i] = v[i]*c;
       }
}

float vecMag(float *v)
{
       float ans = 0;
       for(int i = 0; i < 3; i++){
              ans += v[i]*v[i];
       }
       return sqrt(ans);
}

void normalize(float *v)
{
       float mag = 1/vecMag(v);
       constMult(mag, v, v);
}

void project(float *w, float *v, float *ans)
{
       float dot = dotProduct(w, v);
       float mag = vecMag(v);
       mag = mag * mag;
       constMult(dot/mag, v, ans);
}

float cosAngle(float *w, float *v)
{
       return (  acos(dotProduct(v,w)/ (vecMag(v) * vecMag(w)) )  );
}


void setVector(float *v, float x, float y, float z)
{
       v[0] = x;
       v[1] = y;
       v[2] = z;
}

bool vecEqual(float *v, float *w)
{
       if(v[0] == w[0] && v[1] == w[1] && v[2] == w[2]){return true;}
       return false;
}


void matix_mult(const float *A, float *v, float *ans)
{
       float temp[3] = {0,0,0};
       cpyVec(v, temp);
       for(int i = 0; i < 3; i++)
       {
              ans[i] = temp[0]*A[i*3] + temp[1]*A[1+i*3] + temp[2]*A[2+i*3];
       }
}