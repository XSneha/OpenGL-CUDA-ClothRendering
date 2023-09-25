#include "Global.h"
#include<vector>

//Global file
FILE* gpFile = NULL;

//window variables
bool  gbActiveWindow = false;
bool  gbIsFullScreen = false;
HDC   ghDC = NULL;
HGLRC ghRC = NULL;
HWND  ghWnd = NULL;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

//window width and height
int wWidth;
int wHeight;

// Global Variables
const int gMeshWidth = 4*4;
const int gMeshHeight = 4*4;
const int gMeshTotal = gMeshWidth * gMeshHeight;

//mesh positions and velocities
float4 pos1[gMeshTotal] = { 0 };
float4 pos2[gMeshTotal] = { 0 };
float4 vel11[gMeshTotal] = { 0 };
float4 vel2[gMeshTotal] = { 0 };

GLuint gShaderProgramObject;
struct cudaGraphicsResource* graphicsResource[5] = { 0 };
//graphicsResource0 for po1 
//graphicsResource1 for po2
//graphicsResource2 for vel1 
//graphicsResource3 for vel2 
//graphicsResource4 for normals 
//graphicsResource5 for texccord 

GLuint vao;
GLuint vbo;
GLuint vbo_norm;
GLuint vbo_gpu[6];
GLuint vbo_index;
GLuint texCloths[2];
bool bOnGPU = true;
bool bWind = false;
bool bTex1 = false;
cudaError_t error;
GLuint mvpUniform;
mat4 perspectiveProjectionMatrix;
float cAngle = 0.0f;


// Global function declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// function declarations
	int initialize(void);
	void display(void);
	void ToggleFullScreen(void);

	// variables 
	bool bDone = false;
	int iRet = 0;
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("MyApp");

	// code
	// create file for logging
	if (fopen_s(&gpFile, "log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot Create log file!"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log.txt file created...\n");
	}

	// initialization of WNDCLASSEX
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(MYICON));

	// register class
	RegisterClassEx(&wndclass);

	// create window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szClassName,
		TEXT("Cuda Cloth Rendering"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100,
		100,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghWnd = hwnd;

	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent failed...\n");
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "initialize() successful...\n");
	}

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);
	ToggleFullScreen();

	// Game Loop 
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				display();
			}
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// function declarations
	void resize(int, int);
	void uninitialize(void);
	void reset(void);

	void ToggleFullScreen(void);

	// code
	switch (iMsg)
	{

	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_CHAR:
		switch (wParam)
		{
		case 'G':
		case 'g':
			if (!bOnGPU)
			{
				// copy data from vertex buffer of CPU to vertex buffer of GPU
				glBindBuffer(GL_COPY_READ_BUFFER, vbo);
				glBindBuffer(GL_COPY_WRITE_BUFFER, vbo_gpu[0]);

				glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, gMeshTotal * sizeof(float));

				glBindBuffer(GL_COPY_READ_BUFFER, 0);
				glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
				bOnGPU = true;
			}
			break;

		case 'W':
		case 'w':
			bWind = !bWind;
			break;

		case 'C':
		case 'c':
			if (bOnGPU)
			{
				// copy data from vertex buffer of GPU to vertex buffer of CPU
				glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[0]);
				vec4* p = (vec4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
				memcpy_s(pos1, gMeshTotal * sizeof(float4), p, gMeshTotal * sizeof(float4));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				bOnGPU = false;
			}
			break;

		case 'r':
		case 'R':
			reset();
			break;
		}

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x46:
			ToggleFullScreen();
			break;

		case VK_LEFT:
			if (cAngle > -2.0f * M_PI)
				cAngle -= 0.01f;
			break;

		case VK_RIGHT:
			if (cAngle < 2.0f * M_PI)
				cAngle += 0.01f;
			break;
		}
		break;

		// returned from here, to block DefWindowProc
		// We have our own painter
	case WM_ERASEBKGND:
		return(0);
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void ToggleFullScreen()
{
	MONITORINFO MI;

	if (gbIsFullScreen == false)
	{
		dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			MI = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghWnd, &wpPrev)
				&& GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &MI))
			{
				SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghWnd,
					HWND_TOP,
					MI.rcMonitor.left,
					MI.rcMonitor.top,
					MI.rcMonitor.right - MI.rcMonitor.left,
					MI.rcMonitor.bottom - MI.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		gbIsFullScreen = true;
	}
	else
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
		gbIsFullScreen = false;
	}
}

int initialize(void)
{
	// function declarations
	void resize(int, int);
	void uninitialize(void);
	BOOL loadTexture(GLuint*, TCHAR[]);
	void GetInitialPositions(vec4*);
	//void InitFont(void);

	// variable declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;
	GLenum result;

	GLuint vertexShaderObject;
	GLuint fragmentShaderObject;

	// code
	// initialize pdf structure
	ZeroMemory((void*)&pfd, sizeof(PIXELFORMATDESCRIPTOR));
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

	ghDC = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	// iPixelFormatIndex is 1 based, so 0 indicates error
	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}

	if (SetPixelFormat(ghDC, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}

	ghRC = wglCreateContext(ghDC);
	if (ghRC == NULL)
	{
		return(-3);
	}

	if (wglMakeCurrent(ghDC, ghRC) == FALSE)
	{
		return(-4);
	}

	//// C U D A /////////////////////////////////////////////////////////

	// cuda initialization
	int devCount;
	error = cudaGetDeviceCount(&devCount);
	if (error != cudaSuccess)
	{
		fprintf(gpFile, "cudaGetDeviceCount failed..\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}
	else if (devCount == 0)
	{
		fprintf(gpFile, "No CUDA device detected..\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}
	else
	{
		error = cudaSetDevice(0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaSetDevice failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}
	}

	//// Programable Pipeline ////////////////////////////////////////////

	result = glewInit();
	if (result != GLEW_OK) {
		fprintf(gpFile, "GLEW initialization failed..\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}

	// create vertex shader object
	vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	// vertex shader source code 
	const GLchar* vertexShaderSourceCode = (GLchar*)
		"#version 450 core" \
		"\n" \

		"in vec4 position;" \
		"in vec3 normal;" \
		"in vec2 texcoord;" \

		"uniform float front = 1.0f;" \
		"uniform mat4 u_m_matrix;" \
		"uniform mat4 u_v_matrix;" \
		"uniform mat4 u_p_matrix;" \
		"uniform vec4 u_light_position = vec4(0.0f, 0.0f, -6.0f, 1.0f);" \

		"out vec3 tnorm;" \
		"out vec3 light_direction;" \
		"out vec3 viewer_vector;" \
		"out vec2 out_Texcoord;" \

		"void main()" \
		"{" \

		"   vec4 eye_coordinates = u_v_matrix * u_m_matrix * position;" \
		"   tnorm = mat3(u_v_matrix * u_m_matrix) * normal * front;" \
		"   light_direction = vec3(u_light_position - eye_coordinates);" \
		"   float tn_dot_ldir = max(dot(tnorm, light_direction), 0.0);" \
		"   viewer_vector = vec3(-eye_coordinates.xyz);" \
		"	gl_Position = u_p_matrix * u_v_matrix * u_m_matrix * vec4(position.xyz , 1.0);" \
		"   out_Texcoord = texcoord;" \
		"}";

	// attach source code to vertex shader
	glShaderSource(vertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);

	// compile vertex shader source code
	glCompileShader(vertexShaderObject);

	// compilation errors 
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar* szInfoLog = NULL;

	glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(vertexShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Vertex Shader Compiler Info Log: %s", szInfoLog);
				free(szInfoLog);
				uninitialize();
				DestroyWindow(ghWnd);
			}
		}
	}

	// create fragment shader object
	fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	// fragment shader source code
	const GLchar* fragmentShaderSourceCode = (GLchar*)
		"#version 450 core" \
		"\n" \

		"in vec3 tnorm;" \
		"in vec3 light_direction;" \
		"in vec3 viewer_vector;" \
		"in vec2 out_Texcoord;" \

		"uniform vec3 u_la = vec3(0.4, 0.4, 0.4);" \
		"uniform vec3 u_ld = vec3(0.8, 0.8, 0.8);" \
		"uniform vec3 u_ls = vec3(1.0, 1.0, 1.0);" \
		"uniform vec3 u_ka = vec3(0.4, 0.4, 0.4);" \
		"uniform vec3 u_kd = vec3(0.8, 0.8, 0.8);" \
		"uniform vec3 u_ks = vec3(1.0, 1.0, 1.0);" \
		"uniform float u_shininess = 25.0;" \

		"uniform sampler2D u_sampler;" \

		"out vec4 FragColor;" \

		"void main (void)" \
		"{" \
		"   vec3 ntnorm = normalize(tnorm);" \
		"   vec3 nlight_direction = normalize(light_direction);" \
		"   vec3 nviewer_vector = normalize(viewer_vector);" \
		"   vec3 reflection_vector = reflect(-nlight_direction, ntnorm);" \
		"   float tn_dot_ldir = max(dot(ntnorm, nlight_direction), 0.0);" \

		"   vec3 ambient  = u_la * u_ka;" \
		"   vec3 diffuse  = u_ld * u_kd * tn_dot_ldir;" \
		"   vec3 specular = u_ls * u_ks * pow(max(dot(reflection_vector, nviewer_vector), 0.0), u_shininess);" \

		"   vec3 phong_ads_light = ambient + diffuse;" \

		"   FragColor = vec4(phong_ads_light, 1.0) * texture(u_sampler, out_Texcoord);" \
		"}";

	// attach source code to fragment shader
	glShaderSource(fragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);

	// compile fragment shader source code
	glCompileShader(fragmentShaderObject);

	// compile errors
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(fragmentShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Fragment Shader Compiler Info Log: %s", szInfoLog);
				free(szInfoLog);
				uninitialize();
				DestroyWindow(ghWnd);
			}
		}
	}

	// create shader program object 
	gShaderProgramObject = glCreateProgram();

	// attach vertex shader to shader program
	glAttachShader(gShaderProgramObject, vertexShaderObject);

	// attach fragment shader to shader program
	glAttachShader(gShaderProgramObject, fragmentShaderObject);

	// pre-linking binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject, CCR_ATTRIB_POSITION, "position");
	glBindAttribLocation(gShaderProgramObject, CCR_ATTRIB_NORMAL, "normal");
	glBindAttribLocation(gShaderProgramObject, CCR_ATTRIB_TEXCOORD, "texcoord");

	// link the shader program
	glLinkProgram(gShaderProgramObject);

	// linking errors
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, ("Shader Program Linking Info Log: %s"), szInfoLog);
				free(szInfoLog);
				uninitialize();
				DestroyWindow(ghWnd);
			}
		}
	}

	// post-linking retrieving uniform locations

	///// cloth mesh coordinates generation ///////////////////////////////////////
	int i, j;

	vec4* initial_positions = new vec4[gMeshTotal];
	vec3* initial_normals = new vec3[gMeshTotal];
	vec2* initial_texcoords = new vec2[gMeshTotal];

	int n = 0;

	GetInitialPositions(initial_positions);

	for (j = 0; j < gMeshHeight; j++)
	{
		float fj = (float)j / (float)gMeshHeight;
		for (i = 0; i < gMeshWidth; i++)
		{
			float fi = (float)i / (float)gMeshWidth;

			initial_normals[n] = vec3(0.0f);
			// texture coords
			initial_texcoords[n][0] = fi * 1.0f;
			initial_texcoords[n][1] = fj * 1.0f;
			n++;

		}
	}

	// create vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// vertex positions
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_positions, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_norm);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	glBufferData(GL_ARRAY_BUFFER, gMeshTotal * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

	// vertex positions
	glGenBuffers(6, vbo_gpu);

	// pos1 and pos2
	for (int i = 0; i < 2; i++)
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[i]);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_positions, GL_DYNAMIC_DRAW);

		// register our vbo with cuda graphics resource
		error = cudaGraphicsGLRegisterBuffer(&graphicsResource[i], vbo_gpu[i], cudaGraphicsMapFlagsWriteDiscard);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}
	}

	// vel1 and vel2
	for (int i = 2; i < 4; i++)
	{
		glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[i]);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), vel11, GL_DYNAMIC_DRAW);

		// register our vbo with cuda graphics resource
		error = cudaGraphicsGLRegisterBuffer(&graphicsResource[i], vbo_gpu[i], cudaGraphicsMapFlagsWriteDiscard);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}
	}

	// normals
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[4]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_normals, GL_DYNAMIC_DRAW);

	// register our vbo with cuda graphics resource
	error = cudaGraphicsGLRegisterBuffer(&graphicsResource[4], vbo_gpu[4], cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed..\n");
		uninitialize();
		DestroyWindow(ghWnd);
	}

	// texcoords
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[5]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_texcoords, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(CCR_ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CCR_ATTRIB_TEXCOORD);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// index buffer for cloth m esh
	int lines = (gMeshWidth * (gMeshHeight - 1)) + gMeshWidth;

	glGenBuffers(1, &vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, lines * 2 * sizeof(int), NULL, GL_STATIC_DRAW);

	int* e = (int*)glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, lines * 2 * sizeof(int), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

	// triangle mesh
	for (j = 0; j < gMeshHeight - 1; j++)
	{
		for (i = 0; i < gMeshWidth; i++)
		{
			*e++ = j * gMeshWidth + i;
			*e++ = (1 + j) * gMeshWidth + i;
		}
		*e++ = PRIMITIVE_RESTART;
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);

	delete[]initial_positions;
	delete[]initial_normals;
	delete[]initial_texcoords;

	// clear the depth buffer
	glClearDepth(1.0f);

	// primitive restart
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(PRIMITIVE_RESTART);

	// enable depth
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	// enable blend
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// face culling
	glEnable(GL_CULL_FACE);

	// clear the screen by OpenGL
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// textures
	glEnable(GL_TEXTURE_2D);
	loadTexture(&texCloths[0], MAKEINTRESOURCE(IDBITMAP_CLOTH2));
	loadTexture(&texCloths[1], MAKEINTRESOURCE(IDBITMAP_CLOTH2));

	// initialize font lib
	//InitFont();

	perspectiveProjectionMatrix = mat4::identity();

	// warm-up call to resize
	resize(WIN_WIDTH, WIN_HEIGHT);

	// play backgroud theme song
	//PlaySound(MAKEINTRESOURCE(IDWAV_THEME), // ID of WAVE resource
	//	GetModuleHandle(NULL), 				// handle of this module, which contains the resource
	//	SND_RESOURCE | SND_ASYNC);			// ID is of type resource | play async (i.e. non-blocking)

	return(0);
}

void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	wWidth = width;
	wHeight = height;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	perspectiveProjectionMatrix = perspective(45.0, (float)width / (float)height, 0.1f, 200.0f);
}

void display(void)
{
	void uninitialize(void);
	void DrawCloth(void);
	//static float alpha = 0.0f;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	DrawCloth();
	SwapBuffers(ghDC);
}

void uninitialize(void)
{
	if (vbo_gpu)
	{
		glDeleteBuffers(4, vbo_gpu);
		for (int i = 0; i < 4; i++)
			vbo_gpu[i] = 0;
	}

	if (vbo)
	{
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}

	if (vao)
	{
		glDeleteVertexArrays(1, &vao);
		vao = 0;
	}

	for (int i = 0; i < 5; i++)
	{
		if (graphicsResource[i])
		{
			cudaGraphicsUnregisterResource(graphicsResource[i]);
			graphicsResource[i] = NULL;
		}
	}

	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei shaderNumber;

		glUseProgram(gShaderProgramObject);
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);

		GLuint* pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);

			for (shaderNumber = 0; shaderNumber < shaderCount; shaderNumber++)
			{
				// detach shader
				glDetachShader(gShaderProgramObject, pShaders[shaderNumber]);

				// delete shader
				glDeleteShader(pShaders[shaderNumber]);
				pShaders[shaderNumber] = 0;
			}
			free(pShaders);
		}

		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);

	}

	// fullscreen check
	if (gbIsFullScreen == true)
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	// break the current context
	if (wglGetCurrentContext() == ghRC)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghRC)
	{
		wglDeleteContext(ghRC);
	}

	if (ghDC)
	{
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "Log file is closed...\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}

void launchCPUKernel( int width,  int height, float3 wind)
{
	vec3 make_vec3(float4);
	vec3 make_vec3(float*);


	//mass
	const float m = 3.0f;
	//time
	const float t = 50.0f * 4;
	
	const float k = 600.0f;
	const float c = 0.55f;
	const float rest_length = 1.00f;
	const float rest_length_diag = 1.41f;

	// latest position in global pos
	float4* ppos1 = pos1;
	float4* ppos2 = pos2;
	float4* pvel1 = vel11;
	float4* pvel2 = vel2;

	for (int count = 0; count < 800; count++)
	{
		for ( int x = 0; x < width; x++)
		{
			for ( int y = 0; y < height; y++)
			{
				int idx = (y * width) + x;
				float3 p = make_float3(ppos1[idx].x, ppos1[idx].y, ppos1[idx].z);
				float3 u = make_float3(pvel1[idx].x, pvel1[idx].y, pvel1[idx].z);

				float3 F = make_float3(0.0f, -10.0f, 0.0f) * m - c * u;
				int i = 0;

				F = F + wind;

				if (pvel1[idx].w >= 0.0f)
				{
					// calculate 8 connections
					// up
					if (y < height - 1)
					{
						i = idx + width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length - x) * normalize(d);
					}
					// down
					if (y > 0)
					{
						i = idx - width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length - x) * normalize(d);
					}
					// left
					if (x > 0)
					{
						i = idx - 1;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length - x) * normalize(d);
					}
					// right
					if (x < width - 1)
					{
						i = idx + 1;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length - x) * normalize(d);
					}

					// lower left
					if (x > 0 && y > 0)
					{
						i = idx - 1 - width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length_diag - x) * normalize(d);
					}
					// upper right
					if (x < (width - 1) && y < (height - 1))
					{
						i = idx + 1 + width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length_diag - x) * normalize(d);
					}
					// lower right
					if (x < (width - 1) && y > 0)
					{
						i = idx + 1 - width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length_diag - x) * normalize(d);
					}
					// upper left
					if (x > 0 && y < (height - 1))
					{
						i = idx - 1 + width;
						float3 q = make_float3(ppos1[i].x, ppos1[i].y, ppos1[i].z);
						float3 d = q - p;
						float x = length(d);
						F = F + -k * (rest_length_diag - x) * normalize(d);
					}

				}
				else
				{
					F = make_float3(0.0f, 0.0f, 0.0f);
				}

				float3 a = F / m;
				float3 s = u * t + 5.0f * a * t * t;
				float3 v = u + a * t;

				float3 pos = p + s;
				ppos2[idx] = make_float4(pos.x, pos.y, pos.z, 1.0f);
				pvel2[idx] = make_float4(v.x, v.y, v.z, vel2[idx].w);

			}
		}
		// swap pointers
		float4* tmp = ppos1;
		ppos1 = ppos2;
		ppos2 = tmp;

		tmp = pvel1;
		pvel1 = pvel2;
		pvel2 = tmp;
	}

	// normals
	float3* norm = new float3[gMeshTotal];
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			unsigned int idx = (y * width) + x;

			float3 p = make_float3(pos1[idx].x, pos1[idx].y, pos1[idx].z);
			float3 n = make_float3(0.0f, 0.0f, 0.0f);
			float3 a, b, c;

			if (y < height - 1)
			{
				c = make_float3(pos1[idx + width]) - p;
				if (x < width - 1)
				{
					a = make_float3(pos1[idx + 1]) - p;
					b = make_float3(pos1[idx + width + 1]) - p;
					n = n + cross(a, b);
					n = n + cross(b, c);
				}
				if (x > 0)
				{
					a = c;
					b = make_float3(pos1[idx + width - 1]) - p;
					c = make_float3(pos1[idx - 1]) - p;
					n = n + cross(a, b);
					n = n + cross(b, c);
				}
			}

			if (y > 0)
			{
				c = make_float3(pos1[idx - width]) - p;
				if (x > 0)
				{
					a = make_float3(pos1[idx - 1]) - p;
					b = make_float3(pos1[idx - width - 1]) - p;
					n = n + cross(a, b);
					n = n + cross(b, c);
				}
				if (x < width - 1)
				{
					a = c;
					b = make_float3(pos1[idx - width + 1]) - p;
					c = make_float3(pos1[idx + 1]) - p;
					n = n + cross(a, b);
					n = n + cross(b, c);
				}
			}

			norm[idx] = n;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	glBufferData(GL_ARRAY_BUFFER, gMeshTotal * 3 * sizeof(float), norm, GL_DYNAMIC_DRAW);

	delete[]norm;
}

// Convert image resource to image data
BOOL loadTexture(GLuint* texture, TCHAR imageResourceID[])
{
	// variables
	HBITMAP hBitmap = NULL;
	BITMAP bmp;
	BOOL bStatus = false;

	// data
	hBitmap = (HBITMAP)LoadImage(GetModuleHandle(NULL),
		imageResourceID,
		IMAGE_BITMAP,
		0, 0,
		LR_CREATEDIBSECTION
	);

	if (hBitmap)
	{
		bStatus = TRUE;
		GetObject(hBitmap, sizeof(BITMAP), &bmp);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		glGenTextures(1, texture);
		glBindTexture(GL_TEXTURE_2D, *texture);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmp.bmWidth, bmp.bmHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, bmp.bmBits);
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);

		DeleteObject(hBitmap);
	}

	return bStatus;
}


// cloth update
__global__ void cloth_kernel(float4* pos1, float4* pos2, float4* vel1, float4* vel2, unsigned int width, unsigned int height, float3 wind)
{

	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	unsigned int idx = (y * width) + x;

	if (idx >= width * height) return;

	const float m = 3.0f;
	const float t = 0.000008 * 4;
	const float k = 6000.0;
	const float c = 0.55;
	const float rest_length = 1.00;
	const float rest_length_diag = 1.41;

	float3 p = make_float3(pos1[idx].x, pos1[idx].y, pos1[idx].z);
	float3 u = make_float3(vel1[idx].x, vel1[idx].y, vel1[idx].z);
	float3 F = make_float3(0.0f, -10.0f, 0.0f) * m - c * u;
	int i = 0;

	F = F + wind;

	if (vel1[idx].w >= 0.0f)
	{
		// calculate 8 connections
		// up
		if (y < height - 1)
		{
			i = idx + width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length - x) * normalize(d);
		}
		// down
		if (y > 0)
		{
			i = idx - width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length - x) * normalize(d);
		}
		// left
		if (x > 0)
		{
			i = idx - 1;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length - x) * normalize(d);
		}
		// right
		if (x < width - 1)
		{
			i = idx + 1;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length - x) * normalize(d);
		}

		// lower left
		if (x > 0 && y > 0)
		{
			i = idx - 1 - width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length_diag - x) * normalize(d);
		}
		// upper right
		if (x < (width - 1) && y < (height - 1))
		{
			i = idx + 1 + width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length_diag - x) * normalize(d);
		}
		// lower right
		if (x < (width - 1) && y > 0)
		{
			i = idx + 1 - width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length_diag - x) * normalize(d);
		}
		// upper left
		if (x > 0 && y < (height - 1))
		{
			i = idx - 1 + width;
			float3 q = make_float3(pos1[i].x, pos1[i].y, pos1[i].z);
			float3 d = q - p;
			float x = length(d);
			F = F + -k * (rest_length_diag - x) * normalize(d);
		}

	}
	else
	{
		F = make_float3(0.0f, 0.0f, 0.0f);
	}

	float3 a = F / m;
	float3 s = u * t + 0.5f * a * t * t;
	float3 v = u + a * t;	

	float3 pos = p + s;

	pos2[idx] = make_float4(pos.x, pos.y, pos.z, 1.0f);
	vel2[idx] = make_float4(v.x, v.y, v.z, vel1[idx].w);

	return;
}

__global__ void cloth_normals(float4* pos, float3* norm, unsigned int width, unsigned int height)
{
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	unsigned int idx = (y * width) + x;

	if (idx >= width * height) return;

	float3 p = make_float3(pos[idx].x, pos[idx].y, pos[idx].z);
	float3 n = make_float3(0.0f, 0.0f, 0.0f);
	float3 a, b, c;

	if (y < height - 1)
	{
		c = make_float3(pos[idx + width]) - p;
		if (x < width - 1)
		{
			a = make_float3(pos[idx + 1]) - p;
			b = make_float3(pos[idx + width + 1]) - p;
			n = n + cross(a, b);
			n = n + cross(b, c);
		}
		if (x > 0)
		{
			a = c;
			b = make_float3(pos[idx + width - 1]) - p;
			c = make_float3(pos[idx - 1]) - p;
			n = n + cross(a, b);
			n = n + cross(b, c);
		}
	}

	if (y > 0)
	{
		c = make_float3(pos[idx - width]) - p;
		if (x > 0)
		{
			a = make_float3(pos[idx - 1]) - p;
			b = make_float3(pos[idx - width - 1]) - p;
			n = n + cross(a, b);
			n = n + cross(b, c);
		}
		if (x < width - 1)
		{
			a = c;
			b = make_float3(pos[idx - width + 1]) - p;
			c = make_float3(pos[idx + 1]) - p;
			n = n + cross(a, b);
			n = n + cross(b, c);
		}
	}

	norm[idx] = n;
}

void launchCUDAKernel(float4* pos1, float4* pos2, float4* vel1, float4* vel2, unsigned int meshWidth, unsigned int meshHeight, float3* norm, float3 wind, float xOffset)
{
	dim3 block(8, 8, 1);
	dim3 grid(meshWidth / block.x, meshHeight / block.y, 1);

	for (int i = 0; i < 400; i++)
	{
		cloth_kernel <<<grid, block >>> (pos1, pos2, vel1, vel2, meshWidth, meshHeight, wind);
		cloth_kernel <<<grid, block >>> (pos2, pos1, vel2, vel1, meshWidth, meshHeight, wind);

	}
	cloth_normals <<<grid, block >>> (pos1, norm, meshWidth, meshHeight);
}

void reset()
{
	void GetInitialPositions(vec4*);

	vec4* pos = new vec4[gMeshTotal];

	GetInitialPositions(pos);

	//// CPU related buffers /////////
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos, GL_DYNAMIC_DRAW);

	//// GPU related buffers /////////
	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[0]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[2]);
	glBufferData(GL_ARRAY_BUFFER, gMeshTotal * 4 * sizeof(float), vel11, GL_DYNAMIC_DRAW);
}  

void GetInitialPositions(vec4* pos)
{
	int i, j;
	int n = 0;
	for (j = 0; j < gMeshHeight; j++)
	{
		float fj = (float)j / (float)gMeshHeight;
		for (i = 0; i < gMeshWidth; i++)
		{
			float fi = (float)i / (float)gMeshWidth;
			vel11[n] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

			pos[n] = vec4((fi - 0.5f) * (float)gMeshWidth,20.0f,(fj - 0.5f) * (float)gMeshHeight,1.0);
			
			// stable points
			//&& (i % 12 == 0 || i == 47)
			if (j == (gMeshHeight - 1) )
				vel11[n].w = -1.0f;

			pos1[n] = make_float4(pos[n][0], pos[n][1], pos[n][2], pos[n][3]);
			n++;
		}
	}
}

void DrawCloth(void)
{
	// function declarations
	void launchCUDAKernel(float4*, float4*, float4*, float4*, unsigned int, unsigned int, float3*, float3, float);
	void launchCPUKernel( int,  int, float3);
	void cloth();
	//void RenderText(std::string, mat4, GLfloat, GLfloat, GLfloat, vec3);
//	mat4 renderOrtho = ortho(0.0f, 1000.0f, 0.0f, 1000.0f * ((float)wHeight / (float)wWidth), -1.0f, 1.0f);

	// variables
	LARGE_INTEGER start, end, elapsed;
	LARGE_INTEGER freq;

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	// use shader program
	glUseProgram(gShaderProgramObject);

	mat4 mMatrix = mat4::identity();
	mMatrix *= rotate(0.0f, 100.0f * sinf(cAngle), 0.0f);
	mMatrix *= vmath::translate(0.0f,0.5f,-8.0f);
	mMatrix *= vmath::scale(1.5f, 1.5f, 1.5f);

	mat4 vMatrix = mat4::identity();
	vMatrix *= lookat(
		vec3(0.0f, 0.0f, 80.0f),
		vec3(0.0f, 0.0f, 0.0f),
		vec3(0.0f, 1.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_m_matrix"), 1, GL_FALSE, mMatrix);
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_v_matrix"), 1, GL_FALSE, vMatrix);
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_p_matrix"), 1, GL_FALSE, perspectiveProjectionMatrix);

	cloth();

	mMatrix = mat4::identity();
	mMatrix *= rotate(0.0f, 100.0f * sinf(cAngle), 0.0f);
	mMatrix *= vmath::translate(-25.0f, 0.5f, -8.0f);
	mMatrix *= vmath::scale(1.5f, 1.5f, 1.5f);

	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_m_matrix"), 1, GL_FALSE, mMatrix);
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_v_matrix"), 1, GL_FALSE, vMatrix);
	glUniformMatrix4fv(glGetUniformLocation(gShaderProgramObject, "u_p_matrix"), 1, GL_FALSE, perspectiveProjectionMatrix);

	cloth();

	//////////////////////////////////////////////////////////////////////////////////////////

	// unuse program
	glUseProgram(0);

	QueryPerformanceCounter(&end);
	elapsed.QuadPart = end.QuadPart - start.QuadPart;

	elapsed.QuadPart *= 1000;
	elapsed.QuadPart /= freq.QuadPart;

}

void cloth() {

	float3 wind = make_float3(0.0f, 0.0f, 0.0f);
	if (bWind) wind = make_float3(0.0f, 0.5f, 5.0f);

	glBindVertexArray(vao);

	if (bOnGPU)
	{
		// 1. map with the resource
		error = cudaGraphicsMapResources(1, &graphicsResource[0], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResource 0 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsMapResources(1, &graphicsResource[1], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResource 1 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsMapResources(1, &graphicsResource[2], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResource 2 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsMapResources(1, &graphicsResource[3], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResource 3 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsMapResources(1, &graphicsResource[4], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsMapResource 4 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		// 2. get pointer to mapped resource
		float4* ppos1 = NULL;
		float4* ppos2 = NULL;
		float4* pvel1 = NULL;
		float4* pvel2 = NULL;
		float3* norm = NULL;

		size_t byteCount;
		error = cudaGraphicsResourceGetMappedPointer((void**)&ppos1, &byteCount, graphicsResource[0]);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer ppos1 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsResourceGetMappedPointer((void**)&ppos2, &byteCount, graphicsResource[1]);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer ppos2 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsResourceGetMappedPointer((void**)&pvel1, &byteCount, graphicsResource[2]);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer pvel1 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsResourceGetMappedPointer((void**)&pvel2, &byteCount, graphicsResource[3]);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer pvel2 failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsResourceGetMappedPointer((void**)&norm, &byteCount, graphicsResource[4]);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsResourceGetMappedPointer norm failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}


		// 3. launch the CUDA kernel
		static float xOffset = 0.0f;
		launchCUDAKernel(ppos1, ppos2, pvel1, pvel2, gMeshWidth, gMeshHeight, norm, wind, xOffset);
		xOffset += 1.11f;

		// 4. unmap the resource
		error = cudaGraphicsUnmapResources(1, &graphicsResource[0], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsUnmapResources(1, &graphicsResource[1], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsUnmapResources(1, &graphicsResource[2], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsUnmapResources(1, &graphicsResource[3], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

		error = cudaGraphicsUnmapResources(1, &graphicsResource[4], 0);
		if (error != cudaSuccess)
		{
			fprintf(gpFile, "cudaGraphicsUnmapResources failed..\n");
			uninitialize();
			DestroyWindow(ghWnd);
		}

	}
	else
	{
		launchCPUKernel(gMeshWidth, gMeshHeight, wind);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), pos1, GL_DYNAMIC_DRAW);
	}

	// bind to the respective buffer
	if (bOnGPU) glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[0]);
	else glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glVertexAttribPointer(CCR_ATTRIB_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CCR_ATTRIB_POSITION);

	if (bOnGPU) glBindBuffer(GL_ARRAY_BUFFER, vbo_gpu[4]);
	else glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);

	glVertexAttribPointer(CCR_ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CCR_ATTRIB_NORMAL);

	glActiveTexture(GL_TEXTURE0);
	if (bTex1) glBindTexture(GL_TEXTURE_2D, texCloths[0]);
	else glBindTexture(GL_TEXTURE_2D, texCloths[1]);

	int lines = (gMeshWidth * (gMeshHeight - 1)) + gMeshWidth;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);

	// draw now!

	// back side
	glUniform1f(glGetUniformLocation(gShaderProgramObject, "front"), 1.0f);
	glCullFace(GL_FRONT);
	glDrawElements(GL_TRIANGLE_STRIP, lines * 2, GL_UNSIGNED_INT, NULL);

	// front side
	glUniform1f(glGetUniformLocation(gShaderProgramObject, "front"), -1.0f);
	glCullFace(GL_BACK);
	glDrawElements(GL_TRIANGLE_STRIP, lines * 2, GL_UNSIGNED_INT, NULL);
	glBindVertexArray(0);
}

/* helper functions for float3 */
__host__ __device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator*(const float3& a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator*(float b, const float3& a)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator/(const float3& a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ float3 operator/(float b, const float3& a)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ float length(const float3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ float3 normalize(const float3& a)
{
	return a / length(a);
}

__host__ __device__ float3 cross(const float3& a, const float3& b)
{
	return make_float3(
		(a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x)
	);
}

__host__ __device__ float3 make_float3(const float4& b)
{
	return make_float3(b.x, b.y, b.z);
}
