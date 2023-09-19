#include"headers/Globals.h"
#define MY_ARRAY_SIZE mesh_width * mesh_hight * 4

// Sine wave

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_COLOR,
	AMC_ATTRIBUTE_NORMAL,
	AMC_ATTRIBUTE_TEXTCOORD
};

//shader objects
GLint gVertexShaderObject;
GLint gFragmentShaderObject;
GLint gShaderProgramObject;

//shader binding objects
GLuint vao;
GLuint vboPos;
GLuint mvpMatrixUniform;

//variables for sine wave
const unsigned mesh_width = 1024;
const unsigned mesh_hight = 1024;
float pos[mesh_width][mesh_hight][4];
int arraySize = mesh_width * mesh_hight * 4;

float animationTime = 0.0f;

struct cudaGraphicsResource* cuda_graphics_resource = NULL;
GLuint vboGPU;

void InitialiseTry2() {

	//Vertex shader
	//create shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

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
	glShaderSource(gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gVertexShaderObject);

	//shader compilation error checking
	GLint infoLogLength = 0;
	GLint shaderCompilationStatus = 0;
	char* szBuffer = NULL;
	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(gVertexShaderObject, infoLogLength, &written, szBuffer);
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
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	const GLchar* fragmentShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"out vec4 FragColor;"\
		"void main(void)"\
		"{"\
		"	FragColor = vec4(1.0,0.7,0.0,1.0);"\
		"}";

	//provide source code to shader object
	glShaderSource(gFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gFragmentShaderObject);

	//shader compilation error checking 
	infoLogLength = 0;
	shaderCompilationStatus = 0;
	szBuffer = NULL;
	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(gFragmentShaderObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//Shader program
	//create shader program object
	gShaderProgramObject = glCreateProgram();

	//attach vertext shader to shader program object
	glAttachShader(gShaderProgramObject, gVertexShaderObject);
	//attach fragment shader to shader program object
	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	//bind attribute with the one which we have specified with in in vertex shader
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");

	//link shader
	glLinkProgram(gShaderProgramObject);

	//error checking for linking
	GLint shaderProgramLinkStatus = 0;
	szBuffer = NULL;
	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);
	if (shaderProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetProgramInfoLog(gShaderProgramObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//get MVP uniform location 
	mvpMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_mvpMatrix");

	//initialize pos array
	for (int i = 0; i < mesh_width; i++) {
		for (int j = 0; j < mesh_hight; j++) {
			for (int k = 0; k < 4; k++) {
				pos[i][j][k] = 0.0f;
			}
		}
	}

	glGenVertexArrays(1, &vao); //
	glBindVertexArray(vao); //

	//bind buffer object for data communication
	glGenBuffers(1, &vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	//glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	//glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboGPU);
	glBindBuffer(GL_ARRAY_BUFFER, vboGPU);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//OpenGL-CUDA interoperability buffer registration
	//below function map buffer object(vboGPU) to cuda graphic resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource, vboGPU, cudaGraphicsMapFlagsWriteDiscard);
	// use write discard flag to wipe previous data
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "Buffer registration failed for cuda graphic resource\n");
		DestroyWindow(ghwnd);
	}
	else {
		fprintf(gpFile, "Buffer registered successfully\n");
		//henceforth cuda_graphics_resource <--> vboGPU
	}

	glBindVertexArray(0);
}
void RenderTry2OnGPU() {
	//start using opengl program object
	glUseProgram(gShaderProgramObject);
	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);

	glBindVertexArray(vao); //bind vao
	//map cuda graphic resource
	cuda_result = cudaGraphicsMapResources(1, &cuda_graphics_resource, 0);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	float4* pPos = NULL;
	size_t numBytes;
	// get poingter for cuda_graphics_resource
	cuda_result = cudaGraphicsResourceGetMappedPointer((void**)&pPos, &numBytes, cuda_graphics_resource);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	//launch kernal
	LaunchCUDAKernal(pPos, mesh_width, mesh_hight, animationTime);
	//get ppos unmmapped in cuda_graphic_resource
	cuda_result = cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "failed while mapping resources\n");
		UnInitialize();
	}
	// bind vboGPU cuda_graphics_resource <--> vboGPU
	glBindBuffer(GL_ARRAY_BUFFER, vboGPU);

	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_hight);
	glBindVertexArray(0); //unbind vao
	animationTime = animationTime + 0.01f;
	//stop using program
	glUseProgram(0);
}


void RenderTry2OnCPU() {
	void LaunchCPUKernal(unsigned int mesh_width, unsigned int mesh_hight, float time);

	//start using opengl program object
	glUseProgram(gShaderProgramObject);
	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);

	glBindVertexArray(vao); //bind vao

	LaunchCPUKernal(mesh_width, mesh_hight, animationTime);

	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_hight);
	glBindVertexArray(0); //unbind vao
	animationTime = animationTime + 0.01f;
	//stop using program
	glUseProgram(0);
}

void UnInitialiseTry2() {
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
}

void LaunchCPUKernal(unsigned int mesh_width, unsigned int mesh_hight, float time) {
	for (int i = 0; i < mesh_width; i++) {
		for (int j = 0; j < mesh_hight; j++) {
			for (int k = 0; k < 4; k++) {
				float u = i / (float)mesh_width;
				float v = j / (float)mesh_hight;
				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;
				float freq = 4.0f;
				float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
				if (k == 0) {
					pos[i][j][k] = u;
				}
				else if (k == 1) {
					pos[i][j][k] = w;
				}
				else if (k == 2) {
					pos[i][j][k] = v;
				}
				else if (k == 3) {
					pos[i][j][k] = 1.0f;
				}
			}
		}
	}
}
