#include "CUDAClothCommonGlobal.h"

//const unsigned mesh_width = 1024;
//const unsigned mesh_hight = 1024;
//float pos[mesh_width][mesh_hight][4];
//int arraySize = mesh_width * mesh_hight * 4;

/*void initializeMeshPosition() {
	//initialize pos array
	for (int i = 0; i < mesh_width; i++) {
		for (int j = 0; j < mesh_hight; j++) {
			for (int k = 0; k < 4; k++) {
				pos[i][j][k] = 0.0f;
			}
		}
	}
}*/

void Display(void) {
	void UnInitialize(void);
	//void LaunchCPUKernal(unsigned int mesh_width, unsigned int mesh_hight, float animationTime);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//start using opengl program object
	glUseProgram(gShaderProgramObject);

	//OpenGL drawing
	//set modelview and projection matrix to dentity 
	mat4 modelViewMatrix = mat4::identity();
	mat4 modelViewprojectionMatrix = mat4::identity();
	modelViewprojectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;
	glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewprojectionMatrix);
	glUniformMatrix4fv(modelUniform, 1, GL_FALSE, modelViewMatrix);
	glUniformMatrix4fv(viewUniform, 1, GL_FALSE, modelViewMatrix);
	glUniformMatrix4fv(projectionUniform, 1, GL_FALSE, perspectiveProjectionMatrix);

	glBindVertexArray(vao); //bind vao

	if (bOnGPU == true) {
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
		LaunchCUDAKernal(pPos, width, height, animationTime);
		//get ppos unmmapped in cuda_graphic_resource
		cuda_result = cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0);
		if (cuda_result != cudaSuccess) {
			fprintf(gpFile, "failed while mapping resources\n");
			UnInitialize();
		}
		// bind vboGPU cuda_graphics_resource <--> vboGPU
		glBindBuffer(GL_ARRAY_BUFFER, vboGPU);
	}
	else {
		//LaunchCPUKernal(mesh_width, mesh_hight, animationTime);
		//getDatafromCPU
		//GetDataForCloth();
		glBindBuffer(GL_ARRAY_BUFFER, vboPos);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(float), vertices, GL_DYNAMIC_DRAW);
		
		glBindBuffer(GL_ARRAY_BUFFER, vboPos);
		glBufferData(GL_ARRAY_BUFFER, MY_ARRAY_SIZE * sizeof(vmath::vec3), normals, GL_DYNAMIC_DRAW);

	}
	glVertexAttribPointer(CUDA_CLOTH_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(CUDA_CLOTH_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_TRIANGLES, 0, width * height);
	glBindVertexArray(0); //unbind vao
	animationTime = animationTime + 0.01f;
	//stop using program
	glUseProgram(0);

	//glFlush();
	SwapBuffers(ghdc);
}
/*
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
					pos[i][j][k] = w ;
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
*/