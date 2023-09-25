// Headers
#include <Windows.h>
#include <stdio.h>
#include <string>

#include <C:/glew-2.1.0/include/GL/glew.h>
#include <gl/GL.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>

#include "vmath.h"
#include "resource.h"

// Linker Options
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "winmm.lib")  // for PlaySound()
#pragma comment(lib, "cudart.lib") // for CUDA


// Defines
#define WIN_WIDTH  800
#define WIN_HEIGHT 600

#define PRIMITIVE_RESTART 0xffffff

using namespace vmath;

enum {
	CCR_ATTRIB_POSITION = 0,
	CCR_ATTRIB_COLOR,
	CCR_ATTRIB_NORMAL,
	CCR_ATTRIB_TEXCOORD
};


// Global Variables
extern const int gMeshWidth ;
extern const int gMeshHeight;
extern const int gMeshTotal;

#define MY_ARRAY_SIZE gMeshWidth*gMeshHeight*4

//global file
extern FILE* gpFile;

//Helper functions
/* helper functions for float3 */
__host__ __device__ float3 operator+(const float3& a, const float3& b);

__host__ __device__ float3 operator-(const float3& a, const float3& b);

__host__ __device__ float3 operator*(const float3& a, float b);

__host__ __device__ float3 operator*(float b, const float3& a);

__host__ __device__ float3 operator/(const float3& a, float b);

__host__ __device__ float3 operator/(float b, const float3& a);

__host__ __device__ float length(const float3& a);

__host__ __device__ float3 normalize(const float3& a);

__host__ __device__ float3 cross(const float3& a, const float3& b);

__host__ __device__ float3 make_float3(const float4& b);
