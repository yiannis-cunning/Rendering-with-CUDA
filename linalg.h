#pragma once

void cpyVec(float *v, float *ans);

void cross(float *v, float *w, float *ans);

float dotProduct(float *v, float *w);

void addVec(float *v, float *w, float *ans);

void subVec(float *v, float *w, float *ans);

void constMult(float c, float *v, float *ans);

float vecMag(float *v);

void normalize(float *v);

void project(float *w, float *v, float *ans);

float cosAngle(float *w, float *v);

void setVector(float *v, float x, float y, float z);

bool vecEqual(float *v, float *w);

void matix_mult(const float *A, float *v, float *ans);
