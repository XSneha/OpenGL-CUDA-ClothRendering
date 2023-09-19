#include "CUDAClothCommonGlobal.h"

struct cudaGraphicsResource* cuda_graphics_resource = NULL;
//variables for sine wave

cudaError_t cuda_result;

float animationTime = 0.0f;

//shader objects
GLint gVertexShaderObject;
GLint gFragmentShaderObject;
GLint gShaderProgramObject;

//shader binding objects
GLuint vao;
GLuint vboPos;
GLuint vboNormal;

GLuint vboGPU;

//shader uniforms
GLuint modelUniform;
GLuint viewUniform;
GLuint projectionUniform;

GLuint lightPosUniform;
GLuint viewPosUniform;
GLuint lightColorUniform;
GLuint objectColorUniform;

GLuint mvpMatrixUniform;

//matrix mat4: vmath.h -> typedef : Float16(4 x 4)
mat4 perspectiveProjectionMatrix;

void Initialize(void) {
	void Resize(int, int);

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex = 0;

	//device count
	int dev_count;

	cuda_result = cudaGetDeviceCount(&dev_count);
	if (cuda_result != cudaSuccess) {
		fprintf(gpFile, "Get device count failed\n");
		DestroyWindow(ghwnd);
	}
	else if (dev_count == 0) {
		fprintf(gpFile, "No device found\n");
		DestroyWindow(ghwnd);
	}
	else {
		fprintf(gpFile, "%d device found successfully\n", dev_count);
		cuda_result = cudaSetDevice(0);
		if (cuda_result != cudaSuccess) {
			fprintf(gpFile, "Failed to set device\n");
			DestroyWindow(ghwnd);
		}
		else {
			fprintf(gpFile, "Cuda device set successfully\n");
		}
	}
	ghdc = GetDC(ghwnd);

	ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0) {
		fprintf(gpFile, "ChoosePixelFormat() Failed\n");
		DestroyWindow(ghwnd);
	}
	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE) {
		fprintf(gpFile, "SetPixelFormat() Failed\n");
		DestroyWindow(ghwnd);
	}
	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL) {
		fprintf(gpFile, "wglCreateContext() Failed\n");
		DestroyWindow(ghwnd);
	}
	if (wglMakeCurrent(ghdc, ghrc) == FALSE) {
		fprintf(gpFile, "wglMakeCurrent() Failed\n");
		DestroyWindow(ghwnd);
	}
	//Glew initilalization code
	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//OpenGL realted logs
	fprintf(gpFile, "\n\n OpenGL vendor : %s \n", glGetString(GL_VENDOR));
	fprintf(gpFile, "OpenGL renderer : %s \n", glGetString(GL_RENDERER));
	fprintf(gpFile, "OpenGL renderer : %s \n", glGetString(GL_RENDERER));
	fprintf(gpFile, "OpenGL version : %s \n", glGetString(GL_VERSION));
	fprintf(gpFile, "GLSL version : %s \n\n ", glGetString(GL_SHADING_LANGUAGE_VERSION));

	//OpenGL enabled extensions
	GLint numExt;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);

	//loop
	for (int i = 0; i < numExt; i++) {
		fprintf(gpFile, "%s \n", glGetStringi(GL_EXTENSIONS, i));
	}

	//Vertex shader
	//create shader
	gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

#if 0
	const GLchar* vertexShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"in vec4 vPosition;"\
		"uniform mat4 u_mvpMatrix;"\
		"void main(void)"\
		"{"\
		"gl_Position = u_mvpMatrix * vPosition;"\
		"}";
#endif

#if 1
	const GLchar* vertexShaderSourceCode =
		"#version 440 core													\n"\
		"in vec3 vPosition;													\n"\
		"in vec3 normal;													\n"\
		"uniform mat4 model;												\n"\
		"uniform mat4 view;													\n"\
		"uniform mat4 projection;											\n"\
		"out vec3 FragPosition;													\n"\
		"out vec3 Normal;													\n"\
		"void main() {														\n"\
		"FragPosition = mat3(model) * vPosition.xyz;								\n"\
		"Normal = transpose(inverse(mat3(model))) * normal;					\n"\
		"gl_Position = projection * view * model * vec4(vPosition.xyz,1.0);	\n"\
		"}																	\n";
#endif

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

	#if 1
	const GLchar* fragmentShaderSourceCode =
		"#version 440 core														\n"
		"in vec3 Normal;														\n"
		"in vec3 FragPosition;														\n"
		"out vec4 FragColor;													\n"
		"uniform vec3 lightPos;													\n"
		"uniform vec3 viewPos;													\n"
		"uniform vec3 lightColor;												\n"
		"uniform vec3 objectColor;												\n"
		"void main() {															\n"
		"	// ambient															\n"
		"		float Ka = 0.4;													\n"
		"		vec3 ambient = vec3(Ka);										\n"
		"	// diffuse															\n"
		"		float Kd = 0.5;													\n"
		"		vec3 norm = normalize(Normal);									\n"
		"		vec3 lightDir = normalize(lightPos - FragPosition);					\n"
		"		float diff = max(dot(norm, lightDir), 0.0);						\n"
		"		vec3 diffuse = Kd * diff * lightColor;							\n"
		"	// specular															\n"
		"		float Ks = 0.1;													\n"
		"		float p = 64;													\n"
		"		vec3 viewDir = normalize(viewPos - FragPosition);					\n"
		"		vec3 halfDir = normalize(viewDir + lightDir);					\n"
		"		float spec = pow(max(dot(halfDir, norm), 0.0), p);				\n"
		"		vec3 specular = Ks * spec * lightColor;							\n"
		"		vec3 result = (ambient + diffuse) * objectColor + specular;		\n"
		"		FragColor = vec4(1.0,1.0,1.0, 1.0);									\n"
		"}																		\n";
	#endif


	//		"		FragColor = vec4(result, 1.0);									\n"

	//in vec3 FragPos;
	
	#if 0
	const GLchar* fragmentShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"out vec4 FragColor;"\
		"void main(void)"\
		"{"\
		"	FragColor = vec4(1.0,0.7,0.0,1.0);"\
		"}";
	#endif
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
	glBindAttribLocation(gShaderProgramObject, CUDA_CLOTH_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, CUDA_CLOTH_ATTRIBUTE_NORMAL, "normal");

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
	mvpMatrixUniform	= glGetUniformLocation(gShaderProgramObject, "u_mvpMatrix");
	modelUniform		= glGetUniformLocation(gShaderProgramObject, "model");
	viewUniform 		= glGetUniformLocation(gShaderProgramObject, "view");
	projectionUniform	= glGetUniformLocation(gShaderProgramObject, "projection");
	lightPosUniform		= glGetUniformLocation(gShaderProgramObject, "lightPos");
	viewPosUniform		= glGetUniformLocation(gShaderProgramObject, "viewPos");
	lightColorUniform	= glGetUniformLocation(gShaderProgramObject, "lightColor");
	objectColorUniform	= glGetUniformLocation(gShaderProgramObject, "objectColor");

	glGenVertexArrays(1, &vao); //
	glBindVertexArray(vao); //

	//initializing cloth hight and width
	initBuffersForSheet();
	GetDataForCloth();

	fprintf(gpFile ,"Vertex coordinates : %d",nv);
	for (int i = 0; i < nv; i++) {
		fprintf(gpFile,"\n  %f \t, %f \t, %f ", vertices[i][0], vertices[i][0], vertices[i][0]);
	}
	//bind buffer object for data communication
	glGenBuffers(1, &vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(CUDA_CLOTH_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CUDA_CLOTH_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//bind buffer object for data communication
	glGenBuffers(1, &vboNormal);
	glBindBuffer(GL_ARRAY_BUFFER, vboNormal);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(CUDA_CLOTH_ATTRIBUTE_NORMAL, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CUDA_CLOTH_ATTRIBUTE_NORMAL);
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

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	perspectiveProjectionMatrix = mat4::identity();
	//warmup resize
	Resize(WIN_WIDTH, WIN_HEIGHT);

}