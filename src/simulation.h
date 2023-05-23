#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <bigduckgl/bigduckgl.h>

struct particle {
	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 color;
};

class simulation {
	GLuint vao;
	GLuint vbo;
	std::vector<particle> particles;

public:
	float particle_diameter = 1.0f;
	float delta_time = 0.025f;
	glm::vec3 gravity = glm::vec3(0, -1.5f, 0);
	int per_side = 20;
	int particle_count;
	// pressure params
	float k = 1, k_near = 40.0, roh_0 = 13.0;
	// viscosity params
	float sigma = 2.5, beta = 3.5;
	// spring params - not yet used
	float alpha = 2.5, gamma = 0;
	float L_frac = 0.1, k_spring = 0.3;

	bool pause = false;

	simulation();
	~simulation();
	void initialize();
	void set_data();
	void draw();
	void step();
};