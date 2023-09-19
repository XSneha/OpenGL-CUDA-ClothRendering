#pragma once
#ifndef MASTER_HEADER_H
#define MASTER_HEADER_H
#include<windows.h>
#include<stdio.h>
//GL headers
#include<C:/glew-2.1.0/include/gl/glew.h>
#include<gl/gl.h>
//Cuda headers
#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
//local headers
//#include "MyWindow.h"
#include "vmath.h"

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"OpenGL32.lib")
#pragma comment(lib,"cudart.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
//local headers
#include "../sineWave.cu.h"

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"OpenGL32.lib")
#pragma comment(lib,"cudart.lib")

#define MYICON 101
#define TERRAIN_BITMAP 102
#define BACKGROUND_MUSIC 103
#define STAR_BITMAP 104
#define VEG_GRASS_1 105
#define VEG_GRASS_2 106
#define VEG_GRASS_3 107
#define VEG_GRASS_4 108

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//common globals
extern DWORD dwStyle;
extern HWND ghwnd;
extern FILE* gpFile;

//Projection matrix
extern mat4 perspectiveProjectionMatrix;
//View Matrix
extern mat4 viewMatrix;

//interop variable
extern bool bOnGPU;

//CUDA variables
extern cudaError_t cuda_result;

//Load Texture  
//bool LoadGLTexture(GLuint* texture, TCHAR resourceId[]);

void UnInitialize();

void InitialiseTry1();
void RenderTry1();
void UnInitialiseTry1();

void InitialiseTry2();
void RenderTry2OnCPU();
void RenderTry2OnGPU();
void UnInitialiseTry2();

void InitialiseTry3();
void RenderTry3OnCPU();
void RenderTry3OnGPU();
void UnInitialiseTry3();

void InitialiseTry4();
void RenderTry4OnCPU();
void RenderTry4OnGPU();
void UnInitialiseTry4();

void InitialiseTry5();
void RenderTry5OnCPU();
void RenderTry5OnGPU();
void UnInitialiseTry5();
void SwitchOnCPU();
void SwitchOnGPU();

#endif
