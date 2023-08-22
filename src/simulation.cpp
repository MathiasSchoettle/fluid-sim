#include "simulation.h"
#include "simulation-cuda.h"

#include <chrono>
#include <random>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

using namespace std;

simulation::simulation(float grid_size) : grid_size(grid_size) {
	initialize();
}

simulation::~simulation() {
	fin();
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);	
}

void simulation::initialize() {
	set_data();
	init(particles.size(), grid_size);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, (particles.size()) * sizeof(particle), particles.data(), GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 48, 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 48, (const void *) 16);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 48, (const void *) 32);
}

void simulation::draw() {
	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, particle_count);
}

void simulation::step() {
	if (pause) return;
	
	cudaGraphicsGLRegisterBuffer(&cuda_resource, vbo, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cuda_resource, 0);

	size_t size;
	particle *device_pointer;
	cudaGraphicsResourceGetMappedPointer((void**)&device_pointer, &size, cuda_resource);

	((cuda_fast_step(device_pointer, particles, particle_diameter, grid_size, delta_time, particle_diameter * 1.5f, sigma, beta, k, roh_0, k_near, float3 {gravity.x, gravity.y, gravity.z}, times)));

	cudaGraphicsUnregisterResource(cuda_resource);
	cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

float get_rand() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void add_circle(std::vector<particle> &particles, int radius, float4 pos, float4 color, float particle_diameter, float grid_size) {
	float mult = 0.75 * particle_diameter;

	for (int x = -radius; x < radius; ++x)
		for (int y = -radius; y < radius; ++y)
			for (int z = -radius; z < radius; ++z) {

				particle p;
				p.position = float4 {x * mult + pos.x, y * mult + pos.y, z * mult + pos.z, 1.0f};

				// float4 diff = p.position - pos;

				// if (sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z))
				// 	continue;

				p.velocity = float4 {0, 0, 0, 0};
				p.color = color;
				particles.push_back(p);
			}
}

void simulation::set_data() {
	particles.clear();

	add_circle(particles, 20, float4 {30.f, 55.f, 165.f, 0.f}, float4 {0.6f, 0.6f, 0.6f, 1.0f}, particle_diameter, grid_size);
	add_circle(particles, 20, float4 {160.f, 30.f, 20.f, 0.f}, float4 {0.6f, 0.6f, 0.6f, 1.0f}, particle_diameter, grid_size);
	add_circle(particles, 20, float4 {170.f, 35.f, 160.f, 0.f}, float4 {0.2f, 0.9f, 0.9f, 1.0f}, particle_diameter, grid_size);
	add_circle(particles, 20, float4 {40.f, 70.f, 30.f, 0.f}, float4 {0.2f, 0.4f, 0.3f, 1.0f}, particle_diameter, grid_size);

	add_circle(particles, 25, float4{110.f, 170.f, 100.f, 0.f}, float4{0.2f, 0.2f, 0.2f, 1.0f}, particle_diameter, grid_size);

	// shuffle so updates are more random
	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(particles), std::end(particles), rng);
	
	particle_count = particles.size();
	glNamedBufferSubData(vbo, 0, (particles.size()) * sizeof(particle), particles.data());
}