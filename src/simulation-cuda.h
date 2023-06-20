#pragma once

#include "simulation.h"

#include <vector>
#include <map>

struct spring {
	float rest_length;
	float distance;
	int start, end;
};

void cuda_step(std::vector<particle> &particles, float delta_time, float h, 
				float sigma, float beta, float k, float roh_0, float k_near);

void cuda_sph_step(std::vector<particle> &particles);

void cuda_fast_step(particle *device_pointer, std::vector<particle> &particles, float cell_size, int grid_size, float delta_time, float h, 
				float sigma, float beta, float k, float roh_0, float k_near, glm::vec3 gravity, std::map<std::string, std::vector<float>> &times);

void fin();

void init(int count, int grid_size);
