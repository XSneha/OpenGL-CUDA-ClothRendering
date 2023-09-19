#include"headers/Globals.h"
#define MY_ARRAY_SIZE t3mesh_width * t3mesh_hight * 4

// Static mesh

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXTCOORD
};

//shader objects
GLint t3gVertexShaderObject;
GLint t3gFragmentShaderObject;
GLint t3gShaderProgramObject;

//shader binding objects
GLuint t3vao;
GLuint t3vboPos;
GLuint t3mvpMatrixUniform;

//variables for sine wave
const unsigned t3mesh_width = 20;
const unsigned t3mesh_hight = 20;
float t3Pos[t3mesh_width][t3mesh_hight][4];
int t3arraySize = t3mesh_width * t3mesh_hight * 4;

float t3animationTime = 0.0f;

struct cudaGraphicsResource* t3cuda_graphics_resource = NULL;
GLuint t3vboGPU;

void InitialiseTry3() {

	//Vertex shader
	//create shader
	t3gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	const GLchar* vertexShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"in vec4 vPosition;"\
		"uniform mat4 u_mvpMatrix;"\
		"void main(void)"\
		"{"\
		"gl_Position = u_mvpMatrix * vPosition;"\
		"}";

	//provide source code to shader object
	glShaderSource(t3gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(t3gVertexShaderObject);

	//shader compilation error checking
	GLint infoLogLength = 0;
	GLint shaderCompilationStatus = 0;
	char* szBuffer = NULL;
	glGetShaderiv(t3gVertexShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(t3gVertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(t3gVertexShaderObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Vertex shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
				//UnInitialize();
				//exit(0);
			}
		}
	}

	//fragment shader
	//create shader
	t3gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* fragmentShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"out vec4 FragColor;"\
		"void main(void)"\
		"{"\
		"	FragColor = vec4(1.0,1.0,1.0,1.0);"\
		"}";

	//provide source code to shader object
	glShaderSource(t3gFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(t3gFragmentShaderObject);

	//shader compilation error checking 
	infoLogLength = 0;
	shaderCompilationStatus = 0;
	szBuffer = NULL;
	glGetShaderiv(t3gFragmentShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(t3gFragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(t3gFragmentShaderObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//Shader program
	//create shader program object
	t3gShaderProgramObject = glCreateProgram();

	//attach vertext shader to shader program object
	glAttachShader(t3gShaderProgramObject, t3gVertexShaderObject);
	//attach fragment shader to shader program object
	glAttachShader(t3gShaderProgramObject, t3gFragmentShaderObject);

	//bind attribute with the one which we have specified with in in vertex shader
	glBindAttribLocation(t3gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	//link shader
	glLinkProgram(t3gShaderProgramObject);

	//error checking for linking
	GLint shaderProgramLinkStatus = 0;
	szBuffer = NULL;
	glGetProgramiv(t3gShaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);
	if (shaderProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(t3gShaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetProgramInfoLog(t3gShaderProgramObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//get MVP uniform location 
	t3mvpMatrixUniform = glGetUniformLocation(t3gShaderProgramObject, "u_mvpMatrix");

	//initialize t3Pos array
	for (int i = 0; i < t3mesh_width; i++) {
		for (int j = 0; j < t3mesh_hight; j++) {
			for (int k = 0; k < 4; k++) {
				t3Pos[i][j][k] = 0.0f;
			}
		}
	}

	glGenVertexArrays(1, &t3vao); //
	glBindVertexArray(t3vao); //

	//bind buffer object for data communication
	glGenBuffers(1, &t3vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, t3vboPos);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	//glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	//glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &t3vboGPU);
	glBindBuffer(GL_ARRAY_BUFFER, t3vboGPU);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//OpenGL-CUDA interoperability buffer registration
	//below function map buffer object(t3vboGPU) to cuda graphic resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t3cuda_graphics_resource, t3vboGPU, cudaGraphicsMapFlagsWriteDiscard);
	// use write discard flag to wipe previous data
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "Buffer registration failed for cuda graphic resource\n");
		DestroyWindow(ghwnd);
	}
	else {
		fprintf(gpFile, "Buffer registered successfully\n");
		//henceforth t3cuda_graphics_resource <--> t3vboGPU
	}

	glBindVertexArray(0);
}
void RenderTry3OnGPU() {
	//start using opengl program object
	glUseProgram(t3gShaderProgramObject);
	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(t3mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);

	glBindVertexArray(t3vao); //bind t3vao
	//map cuda graphic resource
	cuda_result = cudaGraphicsMapResources(1, &t3cuda_graphics_resource, 0);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	float4* pPos = NULL;
	size_t numBytes;
	// get poingter for t3cuda_graphics_resource
	cuda_result = cudaGraphicsResourceGetMappedPointer((void**)&pPos, &numBytes, t3cuda_graphics_resource);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	//launch kernal
	LaunchCUDAKernal(pPos, t3mesh_width, t3mesh_hight, t3animationTime);
	//get pt3Pos unmmapped in cuda_graphic_resource
	cuda_result = cudaGraphicsUnmapResources(1, &t3cuda_graphics_resource, 0);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	// bind t3vboGPU t3cuda_graphics_resource <--> t3vboGPU
	glBindBuffer(GL_ARRAY_BUFFER, t3vboGPU);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glDrawArrays(GL_POINTS, 0, t3mesh_width * t3mesh_hight);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, (t3mesh_width * t3mesh_hight) - 1);

	glBindVertexArray(0); //unbind t3vao
	t3animationTime = t3animationTime + 0.01f;
	//stop using program
	glUseProgram(0);
}


void RenderTry3OnCPU() {
	void LaunchCPUKernalTry3(unsigned int t3mesh_width, unsigned int t3mesh_hight, float time);

	//start using opengl program object
	glUseProgram(t3gShaderProgramObject);
	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(t3mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);

	glBindVertexArray(t3vao); //bind t3vao

	LaunchCPUKernalTry3(t3mesh_width, t3mesh_hight, t3animationTime);

	glBindBuffer(GL_ARRAY_BUFFER, t3vboPos);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), t3Pos, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glDrawArrays(GL_POINTS, 0, t3mesh_width * t3mesh_hight);
	//glDrawArrays(GL_TRIANGLE_STRIP, 0, (t3mesh_width * t3mesh_hight)-1);
	int lines = (t3mesh_width * (t3mesh_hight - 1)) + t3mesh_width;
	glDrawArrays(GL_TRIANGLE_STRIP, 0 , t3arraySize);

	glBindVertexArray(0); //unbind t3vao
	t3animationTime = t3animationTime + 0.01f;
	//stop using program
	glUseProgram(0);
}

void UnInitialiseTry3() {
	if (t3vao) {
		glDeleteVertexArrays(1, &t3vao);
		t3vao = 0;
	}
	if (t3vboPos) {
		glDeleteBuffers(1, &t3vboPos);
		t3vboPos = 0;
	}
	if (t3cuda_graphics_resource) {
		cudaGraphicsUnregisterResource(t3cuda_graphics_resource);
		t3cuda_graphics_resource = NULL;
	}
	if (t3vboGPU) {
		glDeleteBuffers(1, &t3vboGPU);
		t3vboGPU = 0;
	}

	//safe release changes
	if (t3gShaderProgramObject) {
		glUseProgram(t3gShaderProgramObject);
		//shader cound to shaders attached to shader prog object
		GLsizei shaderCount;
		glGetProgramiv(t3gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		GLuint* pShaders;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if (pShaders == NULL) {
			fprintf(gpFile, "Failed to allocate memory for pShaders");
			return;
		}
		//1st shader count is expected value we are passing and 2nd variable we are passing address in which
		//we are getting actual shader count currently attached to shader prog 
		glGetAttachedShaders(t3gShaderProgramObject, shaderCount, &shaderCount, pShaders);
		for (GLsizei i = 0; i < shaderCount; i++) {
			glDetachShader(t3gShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);
		glDeleteProgram(t3gShaderProgramObject);
		t3gShaderProgramObject = 0;
		glUseProgram(0);
	}
}

void LaunchCPUKernalTry3(unsigned int t3mesh_width, unsigned int t3mesh_hight, float time) {
	for (int i = 0; i < t3mesh_width; i++) {
		for (int j = 0; j < t3mesh_hight; j++) {
			for (int k = 0; k < 4; k++) {
				float u = i / (float)t3mesh_width;
				float v = j / (float)t3mesh_hight;
				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;
				//float w = -8.0;
				float freq = 4.0f;
				float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

				if (k == 0) {
					t3Pos[i][j][k] = u;
				}
				else if (k == 1) {
					t3Pos[i][j][k] = v;
				}
				else if (k == 2) {
					t3Pos[i][j][k] = w;
				}
				else if (k == 3) {
					t3Pos[i][j][k] = 1.0f;
				}
				
				/*if (k == 0) {
					t3Pos[i][j][k] = u;
				}
				else if (k == 1) {
					t3Pos[i][j][k] = w;
				}
				else if (k == 2) {
					t3Pos[i][j][k] = v;
				}
				else if (k == 3) {
					t3Pos[i][j][k] = 1.0f;
				}*/
			}
		}
	}
}
