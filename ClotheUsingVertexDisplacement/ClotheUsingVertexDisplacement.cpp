// ClotheUsingVertexDisplacement.cpp : Defines the entry point for the application.
//https://github.com/daw42/glslcookbook/blob/master/chapter09/shader/wave.vs
//https://subscription.packtpub.com/book/game-development/9781789342253/10/ch10lvl1sec91/animating-a-surface-with-vertex-displacement


#include<windows.h>
#include<stdio.h>
#include "vmath.h"

#include<C:/glew-2.1.0/include/gl/glew.h>
#include<gl/gl.h>

#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"OpenGL32.lib")

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

using namespace vmath;

enum {
	SCS_ATTRIBUTE_POSITION = 0,
	SCS_ATTRIBUTE_COLOR,
	SCS_ATTRIBUTE_NORMAL,
	SCS_ATTRIBUTE_TEXTCOORD
};

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

DWORD dwStyle;
HWND ghwnd;
bool gbFullscree = false;
bool gbActiveWindow = false;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

FILE* gpFile;

HDC ghdc = NULL;
HGLRC ghrc = NULL;

//shader objects
GLint gVertexShaderObject;
GLint gFragmentShaderObject;
GLint gShaderProgramObject;

//shader binding objects
GLuint vao_trngl;
GLuint vboPos_trngl;
GLuint vboColor_trngl;

GLuint vao_sqr;
GLuint vboPos_sqr;
GLuint vboColor_sqr;
GLuint vbo_cube_normal;

GLuint mvpMatrixUniform;
GLuint MaterialKdUniform;
GLuint MaterialKSUniform;
GLuint MAterialKaUniform;
GLuint MaterialShininessUniform;
GLuint ModelViewMatrixUniform;
GLuint NormalMatrixUniform;
GLuint TimeUniform;

GLfloat angle_tri, angle_sqr , time = 1.0f;

//matrix mat4: vmath.h -> typedef : Float16(4 x 4)
mat4 perspectiveProjectionMatrix;

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR cmdLine, int iCmdShow) {

	void Initialize(void);
	void Display(void);
	void Update(void);

	WNDCLASSEX wndclass;
	MSG msg;
	HWND hwnd;
	TCHAR szAppName[] = TEXT("OpenGl Template");
	bool bDone = false;

	if (fopen_s(&gpFile, "MyLog.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Failed to Open file Mylog.txt"), TEXT("ERROR"), MB_OK);
		return (0);
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.lpszClassName = szAppName;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpszMenuName = NULL;

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("3D rotation."),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_VISIBLE,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		0,
		0,
		hInstance,
		NULL);

	if (hwnd == NULL) {
		MessageBox(NULL, TEXT("Failed to Create Window."), TEXT("ERROR!"), MB_OK);
		exit(0);
	}
	ghwnd = hwnd;

	Initialize();
	ShowWindow(hwnd, iCmdShow);

	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	while (bDone == false) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) {
				bDone = true;
			}
			else {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else {
			if (gbActiveWindow == true) {
				Display();
				Update();
			}
		}
	}

	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam) {

	void Resize(int, int);
	void UnInitialize(void);
	void ToggleFullscreen(void);

	MONITORINFO mi = { sizeof(MONITORINFO) };

	switch (iMsg) {
	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;
	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;
	case WM_SIZE:
		Resize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_ERASEBKGND:
		return(0);
	case WM_KEYDOWN:
		switch (wParam) {
		case VK_ESCAPE:DestroyWindow(ghwnd);
			break;
		case 0x46:
		case 0x66:
			ToggleFullscreen();
			break;
		default:
			break;
		}
		break;
	case WM_DESTROY:
		UnInitialize();
		PostQuitMessage(0);
		break;
	default:break;
	}

	return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

void ToggleFullscreen(void) {
	MONITORINFO mi = { sizeof(MONITORINFO) };
	if (gbFullscree == false) {
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW) {
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi)) {
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		//ShowCursor(FALSE);
		gbFullscree = true;
	}
	else {
		//ShowCursor(TRUE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
		gbFullscree = false;
	}
}

void Initialize(void) {
	void Resize(int, int);

	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex = 0;

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

	//provide source object to shader
	const GLchar* vertexShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"													"\
		"in vec3 VertexPosition;							"\
		"in vec3 VertexNormal;								"\
		"in vec2 VertexTexCoord;							"\
		"													"\
		"out vec4 Position;									"\
		"out vec3 Normal;									"\
		"out vec2 TexCoord;									"\
		"													"\
		"uniform float Time;								"\
		"uniform float Freq = 2.5;							"\
		"uniform float Velocity = 2.5;						"\
		"uniform float Amp = 0.6;							"\
		"													"\
		"uniform mat4 ModelViewMatrix;						"\
		"uniform mat3 NormalMatrix;							"\
		"uniform mat4 MVP;									"\
		"													"\
		"void main()										"\
		"{													"\
		"    vec4 pos = vec4(VertexPosition,1.0);			"\
		"													"\
		"    float u = Freq * pos.x - Velocity * Time;		"\
		"    pos.y = Amp * sin( u );						"\
		"													"\
		"    vec3 n = vec3(0.0);							"\
		"    n.xy = normalize(vec2(cos( u ), 1.0));			"\
		"													"\
		"    Position = ModelViewMatrix * pos;				"\
		"    Normal = NormalMatrix * n;						"\
		"    TexCoord = VertexTexCoord;						"\
		"    gl_Position = MVP * pos;						"\
		"}													";

	//in attributes are variables which shaders are going to receive only once
	//uniform attributes are variables which are going to change i.e thode values can vary 

	glShaderSource(gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gVertexShaderObject);

	//shader compilation error checking here
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
			}
		}
	}

	//fragment shader
	//create shader
	gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	//provide source object to shader
	const GLchar* fragmentShaderSourceCode =
		"#version 440 core "\
		"\n"\
		"																"\
		"uniform struct LightInfo {										"\
		"    vec4 Position;												"\
		"    vec3 Intensity;											"\
		"} Light;														"\
		"																"\
		"uniform struct MaterialInfo {									"\
		"    vec3 Ka;													"\
		"    vec3 Kd;													"\
		"    vec3 Ks;													"\
		"    float Shininess;											"\
		"} Material;													"\
		"																"\
		"in vec4 Position;												"\
		"in vec3 Normal;												"\
		"in vec2 TexCoord;												"\
		"																"\
		"uniform float Time;											"\
		"																"\
		"layout ( location = 0 ) out vec4 FragColor;					"\
		"																"\
		"vec3 phongModel(vec3 kd) {										"\
		"    vec3 n = Normal;											"\
		"    vec3 s = normalize(Light.Position.xyz - Position.xyz);		"\
		"    vec3 v = normalize(-Position.xyz);							"\
		"    vec3 r = reflect( -s, n );									"\
		"    float sDotN = max( dot(s,n), 0.0 );						"\
		"    vec3 diffuse = Light.Intensity * kd * sDotN;				"\
		"    vec3 spec = vec3(0.0);										"\
		"    if( sDotN > 0.0 )											"\
		"        spec = Light.Intensity * Material.Ks *					"\
		"            pow( max( dot(r,v), 0.0 ), Material.Shininess );	"\
		"																"\
		"    return Material.Ka * Light.Intensity + diffuse + spec;		"\
		"}																"\
		"																"\
		"void main()													"\
		"{																"\
		"    FragColor = vec4( phongModel(Material.Kd) , 1.0 );			"\
		"}																";

	glShaderSource(gFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	//compile shader
	glCompileShader(gFragmentShaderObject);
	//shader compilation error checking here
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
	//create
	gShaderProgramObject = glCreateProgram();

	//attach vertext shader to shader program
	glAttachShader(gShaderProgramObject, gVertexShaderObject);
	//attach fragment shader to shader program 
	glAttachShader(gShaderProgramObject, gFragmentShaderObject);
							
	//bind attribute with the one which we have specified with in in vertex shader
	glBindAttribLocation(gShaderProgramObject, SCS_ATTRIBUTE_POSITION, "VertexPosition");
	glBindAttribLocation(gShaderProgramObject, SCS_ATTRIBUTE_NORMAL, "VertexNormal");
	glBindAttribLocation(gShaderProgramObject, SCS_ATTRIBUTE_TEXTCOORD, "VertexTexCoord");

	// we are determining position of attribute buffer
	//glBindAttribLocation(gShaderProgramObject, SCS_ATTRIBUTE_COLOR, "vColor");



	//link shader
	glLinkProgram(gShaderProgramObject);
	//linking error cheking code
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
	mvpMatrixUniform = glGetUniformLocation(gShaderProgramObject, "MVP");
	ModelViewMatrixUniform = glGetUniformLocation(gShaderProgramObject, "ModelViewMatrix");
	NormalMatrixUniform = glGetUniformLocation(gShaderProgramObject, "NormalMatrix.Kd");

	MaterialKdUniform = glGetUniformLocation(gShaderProgramObject, "Material.Kd");
	MaterialKSUniform = glGetUniformLocation(gShaderProgramObject, "Material.Ks");
	MAterialKaUniform = glGetUniformLocation(gShaderProgramObject, "Material.Ka");
	TimeUniform = glGetUniformLocation(gShaderProgramObject, "Time");
	MaterialShininessUniform = glGetUniformLocation(gShaderProgramObject, "Material.Shininess");


	const GLfloat triangleVertices[] = {
		//front
		0.0f, 1.0f, 0.0f,	//apex
		-1.0f, -1.0f, 1.0f, //left bottom
		1.0f, -1.0f, 1.0f,	//right bottom
		//left
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		//right
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, 1.0f,
		//back
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f
	};

	const GLfloat triangleColor[] = {
		1.0f,0.0f,0.0f,
		0.0f,1.0f,0.0f,
		0.0f,0.0f,1.0f,

		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 0.0f,

		1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,

		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 0.0f
	};

	glGenVertexArrays(1, &vao_trngl); //?
	glBindVertexArray(vao_trngl); //?

	//push this vector array to shader
	glGenBuffers(1, &vboPos_trngl);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos_trngl);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleVertices), triangleVertices, GL_STATIC_DRAW); //add vertex data

	glVertexAttribPointer(SCS_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL); //map data 
	glEnableVertexAttribArray(SCS_ATTRIBUTE_POSITION); //enable the mapped buffer

	glBindBuffer(GL_ARRAY_BUFFER, 0);//unbind

	//push this color array to shader
	glGenBuffers(1, &vboColor_trngl);
	glBindBuffer(GL_ARRAY_BUFFER, vboColor_trngl);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangleColor), triangleColor, GL_STATIC_DRAW);//add color data

	glVertexAttribPointer(SCS_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SCS_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

	glBindVertexArray(0);

	const GLfloat squareVertices[] = {
		//front
		-1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,

		//left 
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,

		//right
		1.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,

		//back
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,

		//top
		-1.0f, 1.0f, -1.0f,
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, -1.0f,

		//bottom
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f
	};

	const GLfloat squareColor[] = {
		//front
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		//left 
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		//right
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		//back
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,

		//top
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		//bottom
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f
	};

	glGenVertexArrays(1, &vao_sqr); //?
	glBindVertexArray(vao_sqr); //?

	//push this vector array to shader
	glGenBuffers(1, &vboPos_sqr);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos_sqr);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW); //add vertex data

	glVertexAttribPointer(SCS_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL); //map data 
	glEnableVertexAttribArray(SCS_ATTRIBUTE_POSITION); //enable the mapped buffer

	glBindBuffer(GL_ARRAY_BUFFER, 0);//unbind

	//push this color array to shader
	glGenBuffers(1, &vboColor_sqr);
	glBindBuffer(GL_ARRAY_BUFFER, vboColor_sqr);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareColor), squareColor, GL_STATIC_DRAW);//add color data

	glVertexAttribPointer(SCS_ATTRIBUTE_COLOR, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(SCS_ATTRIBUTE_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind



	const GLfloat cubeNormals[] = {
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,

		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,
		-1.0f, 0.0f, 0.0f,

		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 0.0f, 0.0f,

		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, -1.0f,

		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,

		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f
	};

	//push array of normals to shaders
	glGenBuffers(1, &vbo_cube_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_normal);//bind vbo with buffer array
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeNormals), cubeNormals, GL_STATIC_DRAW); //add vertex data

	glVertexAttribPointer(SCS_ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL); //map data 
	glEnableVertexAttribArray(SCS_ATTRIBUTE_NORMAL); //enable the mapped buffer

	glBindBuffer(GL_ARRAY_BUFFER, 0);//unbind


	glBindVertexArray(0);

	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	perspectiveProjectionMatrix = mat4::identity();
	//warmup resize
	Resize(WIN_WIDTH, WIN_HEIGHT);

}

void Resize(int width, int height) {
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	//gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
	perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void Display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//start using opengl program object
	glUseProgram(gShaderProgramObject);

	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	mat4 translationMatrix = mat4::identity();
	mat4 rotationMatrix = mat4::identity();

	translationMatrix = vmath::translate(-1.5f, 0.0f, -6.0f);
	rotationMatrix = vmath::rotate(angle_tri, 0.0f, 1.0f, 0.0f);
	modelViewMatrix = translationMatrix * rotationMatrix;
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	modelViewMatrix = mat4::identity();
	modelViewprojectionMatrix = mat4::identity();
	translationMatrix = mat4::identity();
	rotationMatrix = mat4::identity();
	mat4 scaleMatrix = mat4::identity();

	translationMatrix = vmath::translate(1.5f, 0.0f, -6.0f);
	rotationMatrix = vmath::rotate(angle_sqr, 1.0f, 0.0f, 0.0f);
	scaleMatrix = vmath::scale(0.75f, 0.75f, 0.75f);

	modelViewMatrix = translationMatrix * rotationMatrix * scaleMatrix;
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);
	
	//material properties : ambiant,diffuse,speculer
	glUniform3f(MaterialKdUniform, 0.9f, 0.5f, 0.3f);
	glUniform3f(MaterialKSUniform, 0.8f, 0.8f, 0.8f);
	glUniform3f(MAterialKaUniform, 0.2f, 0.2f, 0.2f);
	//material shinyness 
	glUniform1f(MaterialShininessUniform, 50.0f);
	time += 0.005;
	glUniform1f(TimeUniform,time);

	glUniformMatrix3fv(TimeUniform, 1, GL_FALSE, modelViewprojectionMatrix);

	//push to shader ^

	glBindVertexArray(vao_sqr); //bind vao

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
	glDrawArrays(GL_TRIANGLE_FAN, 20, 4);

	glBindVertexArray(0); //unbind vao

	//stop using program
	glUseProgram(0);

	//glFlush();
	SwapBuffers(ghdc);
}


void Update(void) {
	angle_tri = angle_tri + 0.1f;
	if (angle_tri >= 360.0f) {
		angle_tri = 0.0f;
	}

	angle_sqr = angle_sqr + 0.1f;
	if (angle_sqr >= 360.0f) {
		angle_sqr = 0.0f;
	}
}

void UnInitialize(void) {
	dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
	ShowCursor(TRUE);
	SetWindowLong(ghwnd, GWL_STYLE, dwStyle);
	SetWindowPlacement(ghwnd, &wpPrev);
	SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOOWNERZORDER);

	if (vao_trngl) {
		glDeleteVertexArrays(1, &vao_trngl);
		vao_trngl = 0;
	}

	if (vboPos_trngl) {
		glDeleteBuffers(1, &vboPos_trngl);
		vboPos_trngl = 0;
	}

	if (vboColor_trngl) {
		glDeleteBuffers(1, &vboColor_trngl);
		vboColor_trngl = 0;
	}

	if (vao_sqr) {
		glDeleteVertexArrays(1, &vao_sqr);
		vao_sqr = 0;
	}

	if (vboPos_sqr) {
		glDeleteBuffers(1, &vboPos_sqr);
		vboPos_sqr = 0;
	}

	if (vboColor_sqr) {
		glDeleteBuffers(1, &vboColor_sqr);
		vboColor_sqr = 0;
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
