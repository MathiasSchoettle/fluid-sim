#include "simulation.h"
#include "simulation-cuda.h"

#include <chrono>
#include <random>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

using namespace std;

simulation::simulation() {
	initialize();
}

simulation::~simulation() {
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);	
}

void simulation::initialize() {
	set_data();

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
	glNamedBufferSubData(vbo, 0, (particles.size()) * sizeof(particle), particles.data());
	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, particle_count);
}

void simulation::step() {

	if (pause) return;

	cuda_fast_step(particles, 1.5f, 100, delta_time, particle_diameter * 1.5f, sigma, beta, k, roh_0, k_near, gravity);

	return;

	std::vector<glm::vec3> old_positions(particles.size());
	float h = particle_diameter * 2;

	// gravity
	for (auto &particle : particles) {
		particle.velocity += delta_time * gravity;
	}

	// viscosity
	#pragma omp parallel
	for (int i = 0; i < particles.size() - 1; ++i)
		for (int j = i + 1; j < particles.size(); ++j) {
			auto &current = particles[i];
			auto &other   = particles[j];
			auto pos_diff = other.position - current.position;
			float q = glm::length(pos_diff) / h;

			if (q < 1) {
				auto u = glm::dot((current.velocity - other.velocity), pos_diff);
				
				if (u > 0) {
					auto I = delta_time * (1 - q) * (sigma * u + beta * u * u) * glm::normalize(pos_diff);
					I *= 0.5f;
					current.velocity -= I;
					other.velocity += I; 
				}
			}
		}

	// velocity prediction
	for (int i = 0; i < particles.size(); ++i) {
		old_positions[i] = particles[i].position;
		particles[i].position += delta_time * particles[i].velocity;
	}

	// TODO

	// //adjust springs
	// for (int j = 0; j < positions.size(); ++j) {
	// 	for (int i = 0; i < j; ++i) {

	// 		auto distance = glm::length(positions[j] - positions[i]);
	// 		float q = distance / h;

	// 		if (q < 1) {

	// 			auto spring = tuple(i, j);
	// 			if (springs.find(spring) == springs.end()) {
	// 				springs[spring] = h;
	// 			}

	// 			auto d = gama * springs[spring];

	// 			if (glm::length(positions[j] - positions[i]) > L_frac + d) {
	// 				springs[spring] += delta_t * alpha * (distance - L_frac - d);
	// 			} else if (glm::length(positions[j] - positions[i]) < L_frac + d) {
	// 				springs[spring] -= delta_t * alpha * (L_frac - d - distance);
	// 			}
	// 		}
	// 	}
	// }

	// spring deletion
	// for (auto it = springs.cbegin(); it != springs.cend();) {
	// 	if (it->second > h) {
	// 		it = springs.erase(it);
	// 	}
	// }

	// // spring adjustments
	// for (auto [key, val] : springs) {
	// 	auto r = positions[get<1>(key)] - positions[get<0>(key)];
	// 	auto distance = glm::length(r);
	// 	auto D = delta_t * delta_t * k_spring * (1 - val / h) * (val - distance) * glm::normalize(r);
	// 	positions[get<0>(key)] -= D * 0.5f;
	// 	positions[get<1>(key)] += D * 0.5f;
	// }

	// double density
	for (int i = 0; i < particles.size(); ++i) {
		auto &current = particles[i];
		float roh = 0;
		float roh_near = 0;

		for (int j = 0; j < particles.size(); ++j) {
			if (j == i) continue;

			auto other = particles[j];
			float q = glm::length(other.position - current.position) / h;

			if (q < 1) {
				roh += (1 - q) * (1 - q);
				roh_near += (1 - q) * (1 - q) * (1 - q);
			}	
		}

		float P = k * (roh - roh_0);
		float P_near = k_near * roh_near;
		glm::vec3 d_x = glm::vec3(0);

		for (int j = 0; j < particles.size(); ++j) {
			if (j == i) continue;

			auto &other = particles[j];
			auto pos_diff = other.position - current.position;
			float q = glm::length(other.position - current.position) / h;

			if (q < 1) {
				glm::vec3 D = delta_time * delta_time * (P * (1 - q) + P_near * (1 - q) * (1 - q)) * glm::normalize(pos_diff);
				D *= 0.5f;
				other.position += D;
				d_x = d_x - D;
			}
		}

		current.position += d_x;
	}

	// compute next velocity and set position
	for (int i = 0; i < particles.size(); ++i)
		particles[i].velocity = (particles[i].position - old_positions[i]) / delta_time;
}

float get_rand() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void simulation::set_data() {
	particles.clear();

	float mult = 0.75;
	float pos_offset = 100;

	for (int x = 0; x < per_side; ++x)
		for (int y = 0; y < per_side; ++y)
			for (int z = 0; z < per_side * 3; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 42.5,
					y * particle_diameter * mult + 10,
					z * particle_diameter * mult + 42.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.913, 0.878, 0.784) * 0.75f;
				particles.push_back(p);
			}
	
	for (int x = 0; x < per_side; ++x)
		for (int y = 0; y < per_side; ++y)
			for (int z = 0; z < per_side; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 42.5,
					y * particle_diameter * mult + 60,
					z * particle_diameter * mult + 42.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.213, 0.278, 0.784) * 0.75f; // blue
				particles.push_back(p);
			}
	

	for (int x = 0; x < per_side * 2; ++x)
		for (int y = 0; y < per_side; ++y)
			for (int z = 0; z < per_side; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 32.5,
					y * particle_diameter * mult + 60,
					z * particle_diameter * mult + 12.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.213, 0.878, 0.784) * 0.75f; // turquouise
				particles.push_back(p);
			}

	for (int x = 0; x < per_side; ++x)
		for (int y = 0; y < per_side * 2; ++y)
			for (int z = 0; z < per_side; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 22.5,
					y * particle_diameter * mult + 70,
					z * particle_diameter * mult + 62.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.913, 0.878, 0.784) * 0.75f;
				particles.push_back(p);
			}

	for (int x = 0; x < per_side; ++x)
		for (int y = 0; y < per_side * 2; ++y)
			for (int z = 0; z < per_side; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 22.5,
					y * particle_diameter * mult + 20,
					z * particle_diameter * mult + 22.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.413, 0.878, 0.984) * 0.75f;
				particles.push_back(p);
			}

	
	for (int x = 0; x < per_side; ++x)
		for (int y = -30; y < per_side * 2.5; ++y)
			for (int z = 0; z < per_side; ++z) {
				particle p;
				p.position = glm::vec3(
					x * particle_diameter * mult + 62.5,
					y * particle_diameter * mult + 60,
					z * particle_diameter * mult + 52.5
				);

				p.velocity = glm::vec3(0);
				p.color = glm::vec3(0.05, 0.478, 0.784) * 0.75f;
				particles.push_back(p);
			}

	// shuffle so updates are more random
	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(particles), std::end(particles), rng);
	
	particle_count = particles.size();
}