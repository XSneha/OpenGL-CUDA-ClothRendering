//"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

#include "headers/Globals.h"


using namespace vmath;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

DWORD dwStyle;
HWND ghwnd;
bool gbFullscree = false;
bool gbActiveWindow = false;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

FILE* gpFile;
FILE* extFile;

HDC ghdc = NULL;
HGLRC ghrc = NULL;

//matrix mat4: vmath.h -> typedef : Float16(4 x 4)
mat4 perspectiveProjectionMatrix;

bool bOnGPU = false;
//CUDA variables
cudaError_t cuda_result;

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR cmdLine, int iCmdShow) {

	void Initialize(void);
	void Display(void);

	WNDCLASSEX wndclass;
	MSG msg;
	HWND hwnd;
	TCHAR szAppName[] = TEXT("OpenGl Template");
	bool bDone = false;

	if (fopen_s(&gpFile, "MyLog.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Failed to Open file Mylog.txt"), TEXT("ERROR"), MB_OK);
		return (0);
	}

	if (fopen_s(&extFile, "Extensions.txt", "w") != 0) {
		MessageBox(NULL, TEXT("Failed to Open file Extensions.txt"), TEXT("ERROR"), MB_OK);
		return (0);
	}

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.lpszClassName = szAppName;
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIcon = LoadIcon(hInstance, NULL);
	wndclass.hIconSm = LoadIcon(hInstance, NULL);
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpszMenuName = NULL;

	RegisterClassEx(&wndclass);

	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("Sine Wave."),
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
		case 'C':
		case 'c':
			//SwitchOnCPU();
			bOnGPU = false;
			break;
		case 'G':
		case 'g':
			bOnGPU = true;
			//SwitchOnGPU();
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
		ShowCursor(FALSE);
		gbFullscree = true;
	}
	else {
		ShowCursor(TRUE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
		gbFullscree = false;
	}
}

void Initialize(void) {
	void Resize(int, int);
	void InitializeCUDAKernal();

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
	InitializeCUDAKernal();

	//Glew initilalization code
	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK) {
		wglDeleteContext(ghrc);
		ghrc = NULL;
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	//OpenGL realted logs
	fprintf(extFile, "\n\n OpenGL vendor : %s \n", glGetString(GL_VENDOR));
	fprintf(extFile, "OpenGL renderer : %s \n", glGetString(GL_RENDERER));
	fprintf(extFile, "OpenGL renderer : %s \n", glGetString(GL_RENDERER));
	fprintf(extFile, "OpenGL version : %s \n", glGetString(GL_VERSION));
	fprintf(extFile, "GLSL version : %s \n\n ", glGetString(GL_SHADING_LANGUAGE_VERSION));

	//OpenGL enabled extensions
	GLint numExt;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);

	//loop
	for (int i = 0; i < numExt; i++) {
		fprintf(extFile, "%s \n", glGetStringi(GL_EXTENSIONS, i));
	}

	InitialiseTry1();
	InitialiseTry2();
	InitialiseTry3();
	InitialiseTry4();
	InitialiseTry5();

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

void InitializeCUDAKernal() {

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

}

void Resize(int width, int height) {
	if (height == 0)
		height = 1;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void Display(void) {
	void UnInitialize(void);
	void LaunchCPUKernal(unsigned int mesh_width, unsigned int mesh_hight, float animationTime);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (bOnGPU == true) {
		//RenderTry2OnGPU();
		//RenderTry1();
		//RenderTry3OnGPU();
		//RenderTry4OnGPU();
		RenderTry5OnGPU();
	}
	else {
		//RenderTry2OnCPU();
		//RenderTry1();
		//RenderTry3OnCPU();
		//RenderTry4OnCPU();
		RenderTry5OnCPU();
	}

	//glFlush();
	SwapBuffers(ghdc);
}


void UnInitialize(void) {
	dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
	ShowCursor(TRUE);
	SetWindowLong(ghwnd, GWL_STYLE, dwStyle);
	SetWindowPlacement(ghwnd, &wpPrev);
	SetWindowPos(ghwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOOWNERZORDER);

	UnInitialiseTry1();
	UnInitialiseTry2();
	UnInitialiseTry3();
	UnInitialiseTry4();

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
