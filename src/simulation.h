#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <bigduckgl/bigduckgl.h>

struct particle {
	glm::vec3 position;
	glm::vec3 velocity;
};

class simulation {
	GLuint vao;
	GLuint vbo;
	std::vector<particle> particles;


public:
	float particle_diameter = 2.0f;
	float delta_time = 0.1f;
	glm::vec3 gravity = glm::vec3(0);
	int particle_count = 1000;
	// pressure params
	float k = 1.0, k_near = 20.0, roh_0 = 8.0;
	// viscosity params
	float sigma = 0.0, beta = 0.0;
	// spring params - not yet used
	float alpha = 0.3, gamma = 0.1;
	float L_frac = 0.1, k_spring = 0.3;

	simulation();
	void initialize();
	void set_data();
	void draw();
	void step();
};