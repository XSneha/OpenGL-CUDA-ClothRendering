#include "CUDAClothCommonGlobal.h"

void UnInitialize(void) {
	dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
	ShowCursor(TRUE);
	SetWindowLong(ghwnd, GWL_STYLE, dwStyle);
	SetWindowPlacement(ghwnd, &wpPrev);
	SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOOWNERZORDER);

	if (vao) {
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}
	if (vboPos) {
		glDeleteBuffers(1, &vboPos);
		vboPos = 0;
	}
	if (cuda_graphics_resource) {
		cudaGraphicsUnregisterResource(cuda_graphics_resource);
		cuda_graphics_resource = NULL;
	}
	if (vboGPU) {
		glDeleteBuffers(1, &vboGPU);
		vboGPU = 0;
	}

	//safe release changes
	if (gShaderProgramObject) {
		glUseProgram(gShaderProgramObject);
		//shader cound to shaders attached to shader prog object
		GLsizei shaderCount;
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		GLuint* pShaders;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if (pShaders == NULL) {
			fprintf(gpFile, "Failed to allocate memory for pShaders");
			return;
		}
		//1st shader count is expected value we are passing and 2nd variable we are passing address in which
		//we are getting actual shader count currently attached to shader prog 
		glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);
		for (GLsizei i = 0; i < shaderCount; i++) {
			glDetachShader(gShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);
		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	/*glDetachShader(gShaderProgramObject , gVertexShaderObject);
	glDetachShader(gShaderProgramObject, gFragmentShaderObject);
	glDeleteShader(gVertexShaderObject);
	gVertexShaderObject = 0;
	glDeleteShader(gFragmentShaderObject);
	gFragmentShaderObject = 0;
	glDeleteShader(gShaderProgramObject);
	gShaderProgramObject = 0;
	glUseProgram(0);
	*/

	if (wglGetCurrentContext() == ghrc) {
		wglMakeCurrent(NULL, NULL);
	}
	if (ghrc) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}
	if (ghdc) {
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}
	if (gpFile) {
		fclose(gpFile);
		gpFile = NULL;
	}
}
