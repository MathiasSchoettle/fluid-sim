#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <bigduckgl/bigduckgl.h>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TIME(func) \
{ \
	auto start = std::chrono::high_resolution_clock::now(); \
	func; \
	auto end = std::chrono::high_resolution_clock::now(); \
	std::chrono::duration<double> duration = end - start; \
	std::cout << "Execution time: " << duration.count() * 1000 << " ms" << std::endl; \
}

#define MAX_TIMES 100

struct particle {
	float4 position;
	float4 velocity;
	float4 color;
};

class simulation {
	GLuint vao;
	GLuint vbo;
	cudaGraphicsResource *cuda_resource;
	std::vector<particle> particles;
	float grid_size;

public:
	float particle_diameter = 1.0f;
	float particle_render_factor = 1.5f;
	float delta_time = 0.025f;
	glm::vec3 gravity = glm::vec3(0, -0.25f, 0);
	int particle_count;
	// pressure params
	float k = 0.54, k_near = 40.0, roh_0 = 20.0;
	// viscosity params
	float sigma = 15.0, beta = 7.5;
	// spring params - not yet used
	float alpha = 2.5, gamma = 0;
	float L_frac = 0.1, k_spring = 0.3;

	bool pause = false;
	std::map<std::string, std::vector<float>> times;

	simulation(float grid_size);
	~simulation();
	void initialize();
	void set_data();
	void draw();
	void step();
};