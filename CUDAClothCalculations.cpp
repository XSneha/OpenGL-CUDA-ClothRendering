#include "CUDAClothCommonGlobal.h"
#include <iostream>

float vertices[5000][3];

vmath::vec3* normals = nullptr;
vmath::ivec3* triangles = nullptr;
vmath::vec3* velocities = nullptr;
vmath::vec3* acc = nullptr;
vmath::vec3 constantVelocity = vmath::vec3(0.0f);
vmath::vec3 constantOmega = vmath::vec3(0.0f);
vmath::mat4 transform = vmath::mat4(1.0f);
vmath::vec3* oldPositions = nullptr;
//std::vector<int>* bins = nullptr;
int nSprings = 10, width = 10, height = 10;
int spring_i[5000], spring_j[5000];
float spring_restLength[5000], spring_ks[5000], spring_kd[5000];
float ksStr = 0.12f, kdStr = 0.0012f;
float ksBend = 0.01f, kdBend = 0.0001f;
float ksShear = 0.06f, kdShear = 0.0006f;

int nv = 500, nn = 500, nt = 500;
float spacing = 0.2f;
bool usePBD = true;
bool selfCollisions = true;

using namespace std;

void initBuffersForSheet() {
    for (int i = 0; i < nv; i++) {
        vertices[i][0] = 0; 
        vertices[i][1] = 0; 
        vertices[i][2] = 0;
    }            
    
    //vertices = new vmath::vec3[nv];
    normals = new vmath::vec3[nn];
    triangles = new vmath::ivec3[nt];

    velocities = new vmath::vec3[nv];
    acc = new vmath::vec3[nv];

    std::fill_n(velocities, nv, vmath::vec3(0, 0, 0));
    std::fill_n(acc, nv, vmath::vec3(0, 0, 0));
}

void GetDataForCloth() {
    if (width > 0 && height > 0 && spacing > 0.0f) {
        fprintf(gpFile,"Assertion true : width > 0 && height > 0 && spacing > 0.0f ");
    }
    int m = width;
    int n = height;
    nv = (m + 1) * (n + 1);
    nn = nv;
    nt = 2 * m * n;

    initBuffersForSheet();
   
    if (selfCollisions) {
  //      bins = new vector<uint16_t>[m * n * max(m, n)];
    }
    if (usePBD) {
        oldPositions = new vmath::vec3[nv];
    }

    // Create the vertices, normals and velocities
    for (size_t i = 0; i < m + 1; i++) {
        for (size_t j = 0; j < n + 1; j++) {
            vertices[i * (n + 1) + j][0] = (1.0f * i * spacing) + (-spacing / 2 * (m + 1)) ;
            vertices[i * (n + 1) + j][1] = 0 ;
            vertices[i * (n + 1) + j][2] = (-1.0f * j * spacing) + spacing / 2 * (n + 1);

            /*vertices[i * (n + 1) + j] =
                vmath::vec3(1.0f * i * spacing, 0, -1.0f * j * spacing) +
                vmath::vec3(-spacing / 2 * (m + 1), 0, spacing / 2 * (n + 1));*/
            normals[i * (n + 1) + j] = vmath::vec3(0, 0, 1);
        }
    }
    // Create the triangles
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            triangles[2 * (i * n + j)] = vmath::ivec3(
                i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j);
            triangles[2 * (i * n + j) + 1] =
                vmath::ivec3(i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1,
                    (i + 1) * (n + 1) + j);
        }
    }

    // Create the springs
    uint32_t nStructuralSprings = 2 * m * n + m + n;
    uint32_t nShearSprings = 2 * m * n;
    uint32_t nBendSprings = (m + 1) * (n - 1) + (m - 1) * (n + 1);
    nSprings = nStructuralSprings + nShearSprings + nBendSprings;
    //springs = new Spring[nStructuralSprings + nShearSprings + nBendSprings];
    
#if 1
    spring_i[nSprings];
    spring_j[nSprings];
    spring_restLength[nSprings];
    spring_ks[nSprings];
    spring_kd[nSprings];
#endif

    int springIndex = 0;
    // Structural springs
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n + 1; j++) {
            //springs[springIndex++] = Spring(i * (n + 1) + j, (i + 1) * (n + 1) + j,
             //   spacing, usePBD ? -1 : ksStr, kdStr);
            spring_i[springIndex] = i * (n + 1) + j;
            spring_j[springIndex] = (i + 1) * (n + 1) + j;
            spring_restLength[springIndex] = spacing;
            spring_ks[springIndex] = usePBD ? -1 : ksStr;
            spring_kd[springIndex] = kdStr;
            springIndex++;
        }
    }
    for (size_t i = 0; i < m + 1; i++) {
        for (size_t j = 0; j < n; j++) {
            //springs[springIndex++] = Spring(i * (n + 1) + j, i * (n + 1) + j + 1,
             //spacing, usePBD ? -1 : ksStr, kdStr);
            spring_i[springIndex] = i * (n + 1) + j;
            spring_j[springIndex] = i * (n + 1) + j + 1;
            spring_restLength[springIndex] = spacing;
            spring_ks[springIndex] = usePBD ? -1 : ksStr;
            spring_kd[springIndex] = kdStr;
            springIndex++;
        }
    }
    // Shear springs
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
          //  springs[springIndex++] =
           //     Spring(i * (n + 1) + j, (i + 1) * (n + 1) + j + 1, spacing * sqrt(2),
           //         ksShear, kdShear);
            spring_i[springIndex] = i * (n + 1) + j;
            spring_j[springIndex] = (i + 1) * (n + 1) + j + 1;
            spring_restLength[springIndex] = spacing * sqrt(2);
            spring_ks[springIndex] = ksShear;
            spring_kd[springIndex] = kdShear;
            springIndex++;
           // springs[springIndex++] =
            //    Spring(i * (n + 1) + j + 1, (i + 1) * (n + 1) + j, spacing * sqrt(2),
             //       ksShear, kdShear);
            spring_i[springIndex] = i * (n + 1) + j + 1;
            spring_j[springIndex] = (i + 1) * (n + 1) + j;
            spring_restLength[springIndex] = spacing * sqrt(2);
            spring_ks[springIndex] = ksShear;
            spring_kd[springIndex] = kdShear;
            springIndex++;
        }
    }
    // Bend springs
    for (size_t i = 0; i < m + 1; i++) {
        for (size_t j = 0; j < n - 1; j++) {
            //springs[springIndex++] = Spring(i * (n + 1) + j, i * (n + 1) + j + 2,
             //   2 * spacing, ksBend, kdBend);
            spring_i[springIndex] = i * (n + 1) + j;
            spring_j[springIndex] = i * (n + 1) + j + 2;
            spring_restLength[springIndex] = 2 * spacing;
            spring_ks[springIndex] = ksBend;
            spring_kd[springIndex] = kdBend;
            springIndex++;
        }
    }
    for (size_t i = 0; i < m - 1; i++) {
        for (size_t j = 0; j < n + 1; j++) {
            //springs[springIndex++] = Spring(i * (n + 1) + j, (i + 2) * (n + 1) + j,
            //    2 * spacing, ksBend, kdBend);
            spring_i[springIndex] = i * (n + 1) + j;
            spring_j[springIndex] = (i + 2) * (n + 1) + j;
            spring_restLength[springIndex] = 2 * spacing;
            spring_ks[springIndex] = ksBend;
            spring_kd[springIndex] = kdBend;
            springIndex++;
        }
    }
}
