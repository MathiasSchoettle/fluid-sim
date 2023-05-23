#include "simulation-cuda.h"

#include <iostream>

using namespace std;

__global__ void integrate(particle *particles, int particle_count, glm::vec3 *forces, float mass, float dt) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	int dimensions = 100;
	float epsilon = 0.001f;
	float damping = -0.37f;

	if (index >= particle_count) return;

	particles[index].velocity += dt * forces[index] / mass;
	particles[index].position += dt * particles[index].velocity;

	if (particles[index].position.x - epsilon < 0.0f)
	{
		particles[index].velocity.x *= damping;
		particles[index].position.x = epsilon;
	}
	else if(particles[index].position.x + epsilon > dimensions - 1.f) 
	{
		particles[index].velocity.x *= damping;
		particles[index].position.x = dimensions - 1 - epsilon;
	}
			
	if (particles[index].position.y - epsilon < 0.0f)
	{
		particles[index].velocity.y *= damping;
		particles[index].position.y = epsilon;
	}
	else if(particles[index].position.y + epsilon > dimensions - 1.f) 
	{
		particles[index].velocity.y *= damping;
		particles[index].position.y = dimensions - 1 - epsilon;
	}
			
	if (particles[index].position.z - epsilon < 0.0f)
	{
		particles[index].velocity.z *= damping;
		particles[index].position.z = epsilon;
	}
	else if(particles[index].position.z + epsilon > dimensions - 1.f) 
	{
		particles[index].velocity.z *= damping;
		particles[index].position.z = dimensions - 1 - epsilon;
	}
}

__device__ float std_kernel(float dist_squared, float radius)
{
	float x = 1.0f - dist_squared / (radius * radius);
	return 315.f / ( 64.f * 3.1415 * (radius * radius * radius) ) * x * x * x;
}

__global__ void density_pressure(particle *particles, int particle_count, float *densities, float *pressures, float mass, float gas_constant, float rest_density, float radius) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index < particle_count) {
		glm::vec3 origin = particles[index].position;

		float sum = 0.f;

		for (int i = 0; i < particle_count; ++i) {
			if (i == index) continue;
			glm::vec3 diff = origin - particles[i].position;
			float dist_squared = glm::dot(diff, diff);
			sum += std_kernel(dist_squared, radius);
		}

		densities[index] = sum * mass + 0.00001f;
		pressures[index] = gas_constant * ( densities[index] - rest_density); 
	}
}

__device__ float spiky_kernel_first_derivative(float distance, float radius)
{
	float x = 1.0f - distance / radius;
	return -45.0f / ( 3.1415 * (radius * radius * radius * radius) ) * x * x;
}

__device__ float spiky_kernel_second_derivative(float distance, float radius) {
	float x = 1.0f - distance / radius;
	return 90.f / ( 3.1415 * (radius * radius * radius * radius * radius) ) * x;
}

__device__ glm::vec3 spiky_kernel(float distance, float radius, glm::vec3 dir_from_center)
{
	return spiky_kernel_first_derivative(distance, radius) * dir_from_center;
}


__global__ void compute_forces(particle *particles, int particle_count, glm::vec3 *forces, float *densities, float *pressures, float radius, float mass, float viscosity_coefficient)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	forces[index] = glm::vec3(0, 0, 0);
	
	float particleDensity2 = densities[index] * densities[index];
	for (int j = 0; j < particle_count; j++) {
		float distance = glm::length(particles[index].position - particles[j].position);
		
		if (distance > 0.0f) {
			glm::vec3 direction = (particles[index].position - particles[j].position ) / distance;
			forces[index] -= mass * mass * (pressures[index] / particleDensity2 + pressures[j] / (densities[j] * densities[j])) * spiky_kernel(distance, radius, direction);
			forces[index] += viscosity_coefficient * mass * mass * (particles[j].velocity - particles[index].velocity ) / densities[j] * spiky_kernel_second_derivative(distance, radius);
		}
	}

	// Gravity
//	forces[index] += glm::vec3(0, -9.89 * 2000.f, 0);
}

void cuda_sph_step(std::vector<particle> &particles) {

	float mass = 4.f;
	float dt = 0.0008f;
	float radius = 1.0f;
	float viscosity_coefficient = 2.5f;
	float gas_constant = 2000.0f;
	float rest_density = 9.0f;
	
	particle *particles_gpu;
	cudaMalloc((void**)&particles_gpu, particles.size() * sizeof(particle));
	cudaMemcpy(particles_gpu, particles.data(), particles.size() * sizeof(particle), cudaMemcpyHostToDevice);

	glm::vec3 *forces_gpu;
	cudaMalloc((void**)&forces_gpu, particles.size() * sizeof(glm::vec3));

	float *densities_gpu;
	cudaMalloc((void**)&densities_gpu, particles.size() * sizeof(float));

	float *pressures_gpu;
	cudaMalloc((void**)&pressures_gpu, particles.size() * sizeof(float));

	int block_size = 256;
	int num_blocks = (particles.size() + block_size - 1) / block_size;
	
	density_pressure<<<num_blocks, block_size>>>(particles_gpu, particles.size(), densities_gpu, pressures_gpu, mass, gas_constant, rest_density, radius);
	compute_forces<<<num_blocks, block_size>>>(particles_gpu, particles.size(), forces_gpu, densities_gpu, pressures_gpu, radius, mass, viscosity_coefficient);
	integrate<<<num_blocks, block_size>>>(particles_gpu, particles.size(), forces_gpu, mass, dt);

	cudaDeviceSynchronize();

	cudaMemcpy(particles.data(), particles_gpu, particles.size() * sizeof(particle), cudaMemcpyDeviceToHost);

	cudaFree(particles_gpu);
}
