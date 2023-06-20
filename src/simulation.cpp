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
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(particle), 0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (const void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (const void *)(6 * sizeof(float)));
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

	((cuda_fast_step(device_pointer, particles, particle_diameter, grid_size, delta_time, particle_diameter * 1.5f, sigma, beta, k, roh_0, k_near, gravity, times)));

	cudaGraphicsUnregisterResource(cuda_resource);
	cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

float get_rand() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void add_circle(std::vector<particle> &particles, int radius, glm::vec3 pos, glm::vec3 color, float particle_diameter, float grid_size) {
	float mult = 0.75 * particle_diameter;

	for (int x = -radius; x < radius; ++x)
		for (int y = -radius; y < radius; ++y)
			for (int z = -radius; z < radius; ++z) {

				particle p;
				p.position = glm::vec3(
					x * mult,
					y * mult,
					z * mult
				);

				p.position += pos;
			
				if (glm::length(p.position - pos) > radius * mult) {
					continue;
				}

				p.velocity = glm::vec3(0);
				p.color = color;
				particles.push_back(p);
			}
}

void simulation::set_data() {
	particles.clear();

	add_circle(particles, 20, glm::vec3(30, 55, 165), glm::vec3(0.6, 0.6, 0.6), particle_diameter, grid_size);
	add_circle(particles, 20, glm::vec3(160, 30, 20), glm::vec3(0.6, 0.6, 0.6), particle_diameter, grid_size);
	add_circle(particles, 20, glm::vec3(170, 35, 160), glm::vec3(0.6, 0.6, 0.6), particle_diameter, grid_size);
	add_circle(particles, 20, glm::vec3(40, 70, 30), glm::vec3(0.6, 0.6, 0.6), particle_diameter, grid_size);

	add_circle(particles, 25, glm::vec3(110, 170, 100), glm::vec3(0.2, 0.4, 0.9), particle_diameter, grid_size);

	// shuffle so updates are more random
	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(particles), std::end(particles), rng);
	
	particle_count = particles.size();
	glNamedBufferSubData(vbo, 0, (particles.size()) * sizeof(particle), particles.data());
}