#include "headers/Globals.h"

#define MY_ARRAY_SIZE gMeshWidth*gMeshHeight*4
#define PRIMITIVE_RESTART 0xffffff

enum {
	T5_ATTRIB_POSITION = 0,
	T5_ATTRIB_COLOR,
	T5_ATTRIB_NORMAL,
	T5_ATTRIB_TEXCOORD
};

GLuint t5VertexShaderObject;
GLuint t5FragmentShaderObject;
GLuint t5ShaderProgramObject;

GLuint t5Vao;
GLuint t5vbo;
GLuint t5vbo_norm;

//vbo GPU array is for
//0 : pos1
//1 : pos2
//2 : vel1
//3 : vel2
//4 : normal
// 5 : textcoord
GLuint t5vbo_gpu[5];
GLuint t5vbo_index;
cudaGraphicsResource* t5cuda_graphics_resource[5];

// Mesh Variables
const int gMeshWidth = 6 * 8;
const int gMeshHeight = 6 * 8;
const int gMeshTotal = gMeshWidth * gMeshHeight;
float4 gvel[gMeshTotal] = { 0 };
float4 gpos[gMeshTotal] = { 0 };
//for swapping
float4 gvel1[gMeshTotal] = { 0 };
float4 gpos1[gMeshTotal] = { 0 };

bool gbWind = false;

void SwitchOnCPU()
{
	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[0]);
	vec4* p = (vec4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
	memcpy_s(gpos, gMeshTotal * sizeof(float4), p, gMeshTotal * sizeof(float4));
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void SwitchOnGPU() 
{
	glBindBuffer(GL_COPY_READ_BUFFER, t5vbo);
	glBindBuffer(GL_COPY_WRITE_BUFFER, t5vbo_gpu[0]);
	glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, gMeshTotal * sizeof(float));
	glBindBuffer(GL_COPY_READ_BUFFER, 0);
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
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


void InitialiseTry5() {
	void GetPresetData(vec4 * pos);

	// create vertex shader object
	t5VertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	// vertex shader source code 
	const GLchar* t5vertexShaderSourceCode = (GLchar*)
		"#version 450 core" \
		"\n" \

		"in vec4 position;" \
		"in vec3 normal;" \
		"in vec2 texcoord;" \

		"uniform float front = 1.0f;" \
		"uniform mat4 u_m_matrix;" \
		"uniform mat4 u_v_matrix;" \
		"uniform mat4 u_p_matrix;" \
		"uniform vec4 u_light_position = vec4(0.0f, 0.0f, 5.0f, 1.0f);" \

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
	glShaderSource(t5VertexShaderObject, 1, (const GLchar**)&t5vertexShaderSourceCode, NULL);

	// compile vertex shader source code
	glCompileShader(t5VertexShaderObject);

	// compilation errors 
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar* szInfoLog = NULL;

	glGetShaderiv(t5VertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(t5VertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(t5VertexShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Compilation of vertex shader failed with info: %s", szInfoLog);
				free(szInfoLog);
				UnInitialize();
			}
		}
	}
	fprintf(gpFile, "Vertex shader Compilation done!");


	// create fragment shader object
	t5FragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	// fragment shader source code
	const GLchar* t5fragmentShaderSourceCode = (GLchar*)
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


		"   FragColor = vec4(1.0,1.0,1.0,1.0);" \
		"}";

	// attach source code to fragment shader
	glShaderSource(t5FragmentShaderObject, 1, (const GLchar**)&t5fragmentShaderSourceCode, NULL);

	// compile fragment shader source code
	glCompileShader(t5FragmentShaderObject);

	// compile errors
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(t5FragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(t5FragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(t5FragmentShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Fragment Shader Compilation failed with error: %s", szInfoLog);
				free(szInfoLog);
				UnInitialize();
			}
		}
	}
	fprintf(gpFile, "Fragment shader Compilation done!");

	t5ShaderProgramObject = glCreateProgram();

	glAttachShader(t5ShaderProgramObject, t5VertexShaderObject);

	glAttachShader(t5ShaderProgramObject, t5FragmentShaderObject);

	// pre-linking binding to vertex attribute
	glBindAttribLocation(t5ShaderProgramObject, T5_ATTRIB_POSITION, "position");
	glBindAttribLocation(t5ShaderProgramObject, T5_ATTRIB_NORMAL, "normal");
	glBindAttribLocation(t5ShaderProgramObject, T5_ATTRIB_TEXCOORD, "texcoord");

	// link the shader program
	glLinkProgram(t5ShaderProgramObject);

	// linking errors
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(t5ShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(t5ShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar*)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(t5ShaderProgramObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, ("Shader Program Linking failed with logs: %s"), szInfoLog);
				free(szInfoLog);
				UnInitialize();
			}
		}
	}

	fprintf(gpFile, "Linking Shader programes done!");


	int i, j;
	vec4* initial_positions = new vec4[gMeshTotal];
	vec3* initial_normals = new vec3[gMeshTotal];
	vec2* initial_texcoords = new vec2[gMeshTotal];

	int n = 0;

	GetPresetData(initial_positions);

	for (j = 0; j < gMeshHeight; j++)
	{
		float fj = (float)j / (float)gMeshHeight;
		for (i = 0; i < gMeshWidth; i++)
		{
			float fi = (float)i / (float)gMeshWidth;

			initial_normals[n] = vec3(0.0f);

			// texture coords
			initial_texcoords[n][0] = fi * 5.0f;
			initial_texcoords[n][1] = fj * 5.0f;

			n++;

		}
	}

	// create t5Vao
	glGenVertexArrays(1, &t5Vao);
	glBindVertexArray(t5Vao);

	// vertex positions
	glGenBuffers(1, &t5vbo);
	glBindBuffer(GL_ARRAY_BUFFER, t5vbo);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_positions, GL_DYNAMIC_DRAW);


	glGenBuffers(1, &t5vbo_norm);
	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_norm);
	glBufferData(GL_ARRAY_BUFFER, gMeshTotal * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	

	// vertex positions
	glGenBuffers(6, t5vbo_gpu);

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[0]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_positions, GL_DYNAMIC_DRAW);

	// register our t5vbo with cuda graphics resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t5cuda_graphics_resource[0], t5vbo_gpu[0], cudaGraphicsMapFlagsWriteDiscard);
	if (cuda_result != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed.. for pos1\n");
		UnInitialize();
	}

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[1]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_positions, GL_DYNAMIC_DRAW);

	// register our t5vbo with cuda graphics resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t5cuda_graphics_resource[1], t5vbo_gpu[1], cudaGraphicsMapFlagsWriteDiscard);
	if (cuda_result != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed.. for pos2\n");
		UnInitialize();
	}

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[2]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), gvel, GL_DYNAMIC_DRAW);

	// register our t5vbo with cuda graphics resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t5cuda_graphics_resource[2], t5vbo_gpu[2], cudaGraphicsMapFlagsWriteDiscard);
	if (cuda_result != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed..for vel1\n");
		UnInitialize();
	}

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[3]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), gvel, GL_DYNAMIC_DRAW);

	// register our t5vbo with cuda graphics resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t5cuda_graphics_resource[3], t5vbo_gpu[3], cudaGraphicsMapFlagsWriteDiscard);
	if (cuda_result != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed.. for vel2\n");
		UnInitialize();
	}

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[4]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_normals, GL_DYNAMIC_DRAW);
	

	// register our t5vbo with cuda graphics resource
	cuda_result = cudaGraphicsGLRegisterBuffer(&t5cuda_graphics_resource[4], t5vbo_gpu[4], cudaGraphicsMapFlagsWriteDiscard);
	if (cuda_result != cudaSuccess)
	{
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed..\n");
		UnInitialize();
	}

	// texcoords
	glBindBuffer(GL_ARRAY_BUFFER, t5vbo_gpu[5]);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), initial_texcoords, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(T5_ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(T5_ATTRIB_TEXCOORD);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// index buffer for cloth mesh
	int lines = (gMeshWidth * (gMeshHeight - 1)) + gMeshWidth;

	glGenBuffers(1, &t5vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t5vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, lines * 2 * sizeof(int), NULL, GL_STATIC_DRAW);

	int* e = (int*)glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER, 0, lines * 2 * sizeof(int), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

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
}


void GetPresetData(vec4* pos)
{
	int i, j;
	int n = 0;
	for (j = 0; j < gMeshHeight; j++)
	{
		float fj = (float)j / (float)gMeshHeight;
		for (i = 0; i < gMeshWidth; i++)
		{
			float fi = (float)i / (float)gMeshWidth;
			gvel[n] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			pos[n] = vec4((fi - 0.5f) * (float)gMeshWidth, 20.0f, (fj - 0.5f) * (float)gMeshHeight, 1.0);
			// stable points
			if (j == (gMeshHeight - 1) && (i % 6 == 0 || i == 47))
				gvel[n].w = -1.0f;
			gpos[n] = make_float4(pos[n][0], pos[n][1], pos[n][2], pos[n][3]);

			fprintf(gpFile, "\n\t%f\t %f\t %f\t %f", pos[n][0], pos[n][1], pos[n][2], pos[n][3]);

			n++;
		}
	}
	fprintf(gpFile, "\nGot basic data for Initialization");
}

float cAngle = 1.0;
void RenderTry5OnCPU() {
	void launchCPUKernel(unsigned int width, unsigned int height, float3 wind);
	//mat5 renderOrtho = ortho(0.0f, 1000.0f, 0.0f, 1000.0f * ((float)WIN_HEIGHT / (float)WIN_WIDTH), -1.0f, 1.0f);

	// variables
	LARGE_INTEGER start, end, elapsed;
	LARGE_INTEGER freq;

	glEnable(GL_POLYGON);

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	// use shader program
	glUseProgram(t5ShaderProgramObject);

	vmath::mat4 mMatrix = mat4::identity();
	cAngle += 0.002f;
	mMatrix *= vmath::rotate(100.0f * sinf(cAngle), 0.0f, 1.0f, 0.0f);

	mat4 vMatrix = mat4::identity();
	vMatrix *= lookat(
		vec3(0.0f, 0.0f, 80.0f),
		vec3(0.0f, 0.0f, 0.0f),
		vec3(0.0f, 1.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(t5ShaderProgramObject, "u_m_matrix"), 1, GL_FALSE, mMatrix);
	glUniformMatrix4fv(glGetUniformLocation(t5ShaderProgramObject, "u_v_matrix"), 1, GL_FALSE, vMatrix);
	glUniformMatrix4fv(glGetUniformLocation(t5ShaderProgramObject, "u_p_matrix"), 1, GL_FALSE, perspectiveProjectionMatrix);

	float3 wind = make_float3(0.0f, 0.0f, 0.0f);
	if (gbWind)
		wind = make_float3(0.0f, 0.0f, 8.0f);

	glBindVertexArray(t5Vao);

	launchCPUKernel(gMeshWidth, gMeshHeight, wind);

	glBindBuffer(GL_ARRAY_BUFFER, t5vbo);
	glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), gpos, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(T5_ATTRIB_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(T5_ATTRIB_POSITION);

	
	//Add while adding lights
		glBindBuffer(GL_ARRAY_BUFFER, t5vbo_norm);

		glVertexAttribPointer(T5_ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(T5_ATTRIB_NORMAL);

		glActiveTexture(GL_TEXTURE0);
		//if (bTex1) glBindTexture(GL_TEXTURE_2D, texCloths[0]);
		//else glBindTexture(GL_TEXTURE_2D, texCloths[1]);
	
	int lines = (gMeshWidth * (gMeshHeight - 1)) + gMeshWidth;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t5vbo_index);

	// draw now!

	// back side
	glUniform1f(glGetUniformLocation(t5ShaderProgramObject, "front"), 1.0f);
	glCullFace(GL_FRONT);
	glDrawElements(GL_TRIANGLE_STRIP, lines * 2, GL_UNSIGNED_INT, NULL);
//	glDrawArrays(GL_LINES, 0, gMeshWidth * gMeshHeight);

	// front side
	glUniform1f(glGetUniformLocation(t5ShaderProgramObject, "front"), -1.0f);
	glCullFace(GL_BACK);
	glDrawElements(GL_TRIANGLE_STRIP, lines * 2, GL_UNSIGNED_INT, NULL);
	glBindVertexArray(0);
	//	glDrawArrays(GL_POINTS, 0, gMeshWidth * gMeshHeight);

		//////////////////////////////////////////////////////////////////////////////////////////

		// unuse program
	glUseProgram(0);

	QueryPerformanceCounter(&end);
	elapsed.QuadPart = end.QuadPart - start.QuadPart;

	elapsed.QuadPart *= 1000;
	elapsed.QuadPart /= freq.QuadPart;
}

void launchCPUKernel(unsigned int width, unsigned int height, float3 wind) {

	const float m = 1.0f;
	const float t = 0.000005 * 4;
	const float k = 6000.0;
	const float c = 0.55;
	const float rest_length = 1.00;
	const float rest_length_diag = 1.41;

	// latest position in global pos

	float4* ppos1 = gpos;
	float4* ppos2 = gpos1;
	float4* pvel1 = gvel;
	float4* pvel2 = gvel1;

	for (int count = 0; count < 800; count++)
	{
		for (unsigned int x = 0; x < width; x++)
		{
			for (unsigned int y = 0; y < height; y++)
			{
				unsigned int idx = (y * width) + x;
				float3 p = make_float3(ppos1[idx].x, ppos1[idx].y, ppos1[idx].z);
				float3 u = make_float3(pvel1[idx].x, pvel1[idx].y, pvel1[idx].z);
				float3 F = make_float3(0.0f, -10.0f, 0.0f) * m - (u * c);

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
				float3 s = u * t + 0.5f * a * t * t;
				float3 v = u + a * t;
				float3 pos = p + s;
				ppos2[idx] = make_float4(pos.x, pos.y, pos.z, 1.0f);
				pvel2[idx] = make_float4(v.x, v.y, v.z, gvel1[idx].w);

			}
		}
		// swap pointers
		float4* tmp = ppos1;
		ppos1 = ppos2;
		ppos2 = tmp;

		tmp = pvel1;
		pvel1 = pvel2;
		pvel2 = tmp;
		//free(tmp);
		//tmp = NULL;
	}
	//if (pvel1 != NULL) {
	//	free(pvel1);
	//}
	//if (pvel2 != NULL) {
	//	free(pvel2);
	//}
	//if (ppos2 != NULL) {
	//	free(ppos2);
	//}
	//if (ppos1 != NULL) {
	//	free(ppos1);
	//}
//	pvel1 = NULL;
//	pvel2 = NULL;
//	ppos1 = NULL;
//	ppos2 = NULL;
}

void RenderTry5OnGPU() {


}

void UnInitialiseTry5() {
	if (t5Vao) {
		glDeleteVertexArrays(1, &t5Vao);
		t5Vao = 0;
	}
	if (t5vbo) {
		glDeleteBuffers(1, &t5vbo);
		t5vbo = 0;
	}
	for (int i = 0; i < 5; i++) {
		if (t5cuda_graphics_resource[i]) {
			cudaGraphicsUnregisterResource(t5cuda_graphics_resource[i]);
			t5cuda_graphics_resource[i] = NULL;
		}
	}
	for (int i = 0; i < 5; i++) {
		if (t5vbo_gpu[i]) {
			glDeleteBuffers(1, &t5vbo_gpu[i]);
			t5vbo_gpu[i] = 0;
		}
	}


	//safe release changes
	if (t5ShaderProgramObject) {
		glUseProgram(t5ShaderProgramObject);
		//shader cound to shaders attached to shader prog object
		GLsizei shaderCount;
		glGetProgramiv(t5ShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		GLuint* pShaders;
		pShaders = (GLuint*)malloc(sizeof(GLuint) * shaderCount);
		if (pShaders == NULL) {
			fprintf(gpFile, "Failed to allocate memory for pShaders");
			return;
		}
		//1st shader count is expected value we are passing and 2nd variable we are passing address in which
		//we are getting actual shader count currently attached to shader prog 
		glGetAttachedShaders(t5ShaderProgramObject, shaderCount, &shaderCount, pShaders);
		for (GLsizei i = 0; i < shaderCount; i++) {
			glDetachShader(t5ShaderProgramObject, pShaders[i]);
			glDeleteShader(pShaders[i]);
			pShaders[i] = 0;
		}
		free(pShaders);
		glDeleteProgram(t5ShaderProgramObject);
		t5ShaderProgramObject = 0;
		glUseProgram(0);
	}
}



