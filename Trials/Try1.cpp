#include "headers/Globals.h"

// Multicolored Quad

enum {
	SCS_ATTRIBUTE_POSITION = 0,
	SCS_ATTRIBUTE_COLOR,
	SCS_ATTRIBUTE_NORMAL,
	SCS_ATTRIBUTE_TEXTCOORD
};

//shader objects
GLint gt1VertexShaderObject;
GLint gt1FragmentShaderObject;
GLint t1gShaderProgramObject;

//shader binding objects
GLuint t1vao;
GLuint t1vboPos;
GLuint t1mvpMatrixUniform;
GLuint fdfoUniform;

//color variables
GLuint t1vboColor;
float fdfo_alpha = 1.0;

void InitialiseTry1() {
	//Vertex shader
	//create shader
	gt1VertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	//provide source object to shader
	const GLchar* vertexShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"in vec4 vPosition;"\
		"in vec4 vColor;"\
		"uniform mat4 u_mvpMatrix;"\
		"out vec4 outColor;"\
		"void main(void)"\
		"{"\
		"gl_Position = u_mvpMatrix * vPosition;"\
		"outColor = vec4(vColor.xyz,0.0);"\
		"}";


	glShaderSource(gt1VertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gt1VertexShaderObject);

	//shader compilation error checking here
	GLint infoLogLength = 0;
	GLint shaderCompilationStatus = 0;
	char* szBuffer = NULL;
	glGetShaderiv(gt1VertexShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(gt1VertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(gt1VertexShaderObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Vertex shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//fragment shader
	//create shader
	gt1FragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//provide source object to shader
	const GLchar* fragmentShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"out vec4 FragColor;"\
		"in vec4 outColor;"\
		"uniform float alpha;"
		"void main(void)"\
		"{"\
		"	FragColor = vec4(outColor.xyz,alpha);"\
		"}";

	glShaderSource(gt1FragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gt1FragmentShaderObject);
	//shader compilation error checking here
	infoLogLength = 0;
	shaderCompilationStatus = 0;
	szBuffer = NULL;
	glGetShaderiv(gt1FragmentShaderObject, GL_COMPILE_STATUS, &shaderCompilationStatus);
	if (shaderCompilationStatus == GL_FALSE) {
		glGetShaderiv(gt1FragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetShaderInfoLog(gt1FragmentShaderObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	//Shader program
	//create
	t1gShaderProgramObject = glCreateProgram();

	//attach vertext shader to shader program
	glAttachShader(t1gShaderProgramObject, gt1VertexShaderObject);
	//attach fragment shader to shader program 
	glAttachShader(t1gShaderProgramObject, gt1FragmentShaderObject);

	//bind attribute with the one which we have specified with in in vertex shader
	glBindAttribLocation(t1gShaderProgramObject, SCS_ATTRIBUTE_POSITION, "vPosition");
	// 
	glBindAttribLocation(t1gShaderProgramObject, SCS_ATTRIBUTE_COLOR, "vColor");

	//link shader
	glLinkProgram(t1gShaderProgramObject);
	//linking error cheking code
	GLint shaderProgramLinkStatus = 0;
	szBuffer = NULL;
	glGetProgramiv(t1gShaderProgramObject, GL_LINK_STATUS, &shaderProgramLinkStatus);
	if (shaderProgramLinkStatus == GL_FALSE) {
		glGetProgramiv(t1gShaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
		if (infoLogLength > 0) {
			szBuffer = (char*)malloc(infoLogLength);
			if (szBuffer != NULL) {
				GLint written;
				glGetProgramInfoLog(t1gShaderProgramObject, infoLogLength, &written, szBuffer);
				//print log to file
				fprintf(gpFile, "Fragment shader logs : %s \n", szBuffer);
				free(szBuffer);
				DestroyWindow(ghwnd);
			}
		}
	}

	fprintf(gpFile, "Vertext Fragment Shader lining done \n");

	//get MVP uniform location 
	t1mvpMatrixUniform = glGetUniformLocation(t1gShaderProgramObject, "u_mvpMatrix");
	fdfoUniform = glGetUniformLocation(t1gShaderProgramObject, "alpha");

	const GLfloat triangleVertices[] = {
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f
	};

	const GLfloat triangleColor[] = {
		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,
		1.0f,1.0f,0.0f
	};

	glGenVertexArrays(1, &t1vao); //?
	glBindVertexArray(t1vao); //?

	//push this vector array to shader
	glGenBuffers(1, &t1vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, t1vboPos);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW); //add vertex data

	glVertexAttribPointer(SCS_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL); //map data 
	glEnableVertexAttribArray(SCS_ATTRIBUTE_POSITION); //enable the mapped buffer

	glBindBuffer(GL_ARRAY_BUFFER, 0);//unbind

	//push this color array to shader
	glGenBuffers(1, &t1vboColor);
	glBindBuffer(GL_ARRAY_BUFFER, t1vboColor);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleColor), triangleColor, GL_STATIC_DRAW);//add color data

	glVertexAttribPointer(SCS_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SCS_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

	fprintf(gpFile, "Vao VBO initialization done \n");


	glBindVertexArray(0);
}

void RenderTry1() {

	//start using opengl program object
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)WIN_WIDTH / (GLfloat)WIN_HEIGHT, 0.1f, 100.0f);

	glUseProgram(t1gShaderProgramObject);

	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	mat4 translationMatrix = mat4::identity();

	translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);

	modelViewMatrix = translationMatrix;
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(t1mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);
	glUniform1f(fdfoUniform, fdfo_alpha);
	//push to shader ^

	glBindVertexArray(t1vao); //bind t1vao

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

	glBindVertexArray(0); //unbind t1vao

	//stop using program
	glUseProgram(0);
	glDisable(GL_BLEND);
}

void UnInitialiseTry1() {
	if (t1vao) {
		glDeleteVertexArrays(1, &t1vao);
		t1vao = 0;
	}
	if (t1vboPos) {
		glDeleteBuffers(1, &t1vboPos);
		t1vboPos = 0;
	}
	if (t1vboColor) {
		glDeleteBuffers(1, &t1vboColor);
		t1vboColor = 0;
	}
	//safe release changes
	if (t1gShaderProgramObject) {
		glUseProgram(t1gShaderProgramObject);
		//shader cound to shaders attached to shader prog object
		GLsizei shaderCount;
		glGetProgramiv(t1gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		GLuint* pShaders;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if (pShaders == NULL) {
			fprintf(gpFile, "Failed to allocate memory for pShaders");
			return;
		}
		//1st shader count is expected value we are passing and 2nd variable we are passing address in which
		//we are getting actual shader count currently attached to shader prog 
		glGetAttachedShaders(t1gShaderProgramObject, shaderCount, &shaderCount, pShaders);
		for (GLsizei i = 0; i < shaderCount; i++) {
			glDetachShader(t1gShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);
		glDeleteProgram(t1gShaderProgramObject);
		t1gShaderProgramObject = 0;
		glUseProgram(0);
	}
}