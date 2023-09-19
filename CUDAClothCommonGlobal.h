//"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

#include<windows.h>
#include<stdio.h>
//GL headers
#include<gl/glew.h>
#include<gl/gl.h>
//Cuda headers
#include<cuda_gl_interop.h>
#include<cuda_runtime.h>
//local headers
#include "vmath.h"
#include "sineWave.cu.h"

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"OpenGL32.lib")
#pragma comment(lib,"cudart.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600
#define MY_ARRAY_SIZE nv * 3

using namespace vmath;

enum {
	CUDA_CLOTH_ATTRIBUTE_POSITION = 0,
	CUDA_CLOTH_ATTRIBUTE_COLOR,
	CUDA_CLOTH_ATTRIBUTE_NORMAL,
	CUDA_CLOTH_ATTRIBUTE_TEXTCOORD
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void Initialize(void);
void Display(void);
void UnInitialize(void);

//CUDA variables
extern cudaError_t cuda_result;
extern struct cudaGraphicsResource* cuda_graphics_resource;
//mesh variables
//variables for sine wave

//extern const unsigned mesh_width;
//extern const unsigned mesh_hight;
extern int arraySize;
 
extern float animationTime;

extern DWORD dwStyle;
extern HWND ghwnd;
extern bool gbFullscree;
extern bool gbActiveWindow ;
extern WINDOWPLACEMENT wpPrev ;

extern FILE* gpFile;
extern bool bOnGPU;

extern HDC ghdc;
extern HGLRC ghrc;

//shader objects
extern GLint gVertexShaderObject;
extern GLint gFragmentShaderObject;
extern GLint gShaderProgramObject;

//shader binding objects
extern GLuint vao;
extern GLuint vboPos;
extern GLuint mvpMatrixUniform;
extern GLuint vboGPU;

//matrix mat4: vmath.h -> typedef : Float16(4 x 4)
extern mat4 perspectiveProjectionMatrix;

//
extern GLuint modelUniform;
extern GLuint viewUniform;
extern GLuint projectionUniform;
extern GLuint lightPosUniform;
extern GLuint viewPosUniform;
extern GLuint lightColorUniform;
extern GLuint objectColorUniform;

//Cloth calculation variables
//extern vmath::vec3* vertices;
extern float vertices[5000][3];

extern vmath::vec3* normals;
extern int nv, nn, nt;
void initBuffersForSheet();
void GetDataForCloth();
extern int nSprings, width, height;
