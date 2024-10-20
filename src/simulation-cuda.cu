#include "simulation-cuda.h"
#include "cuda-helper.h"

#include <iostream>
#include <chrono>
#include <cub/cub.cuh> 

using namespace std;

#define MAX_PARTICLES 10

__device__ __forceinline__ int compute_grid(const particle &particle, float cell_size, int grid_size) {
	int x = particle.position.x / cell_size;
	int y = particle.position.y / cell_size;
	int z = particle.position.z / cell_size;
	return linear_grid_index(x, y, z, grid_size);
}

__global__ void bin_particles(particle *particles, int particle_count, int *cell_ids, int *particle_ids, float cell_size, int grid_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	cell_ids[index] = compute_grid(particles[index], cell_size, grid_size);
	particle_ids[index] = index;
}

__global__ void find_cell_ranges(int *cell_ids, int *particle_ids, int *cell_starts, int *cell_ends, int particle_count) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count - 1) return;

	unsigned int left  = cell_ids[index];
	unsigned int right = cell_ids[index + 1];

	if (index == 0) {
		cell_starts[left] = 0;
	}

	if (index == particle_count - 2) {
		cell_ends[right] = particle_count;
	}
	
	if(left != right) {
		cell_ends[left] = index + 1;
		cell_starts[right] = index + 1;
	}
}

__global__ void viscosity(particle *particles, const int *cell_starts, const int *cell_ends, const int *particle_ids_sorted, int particle_count, float cell_size, int grid_size, float delta_time, float sigma, float beta, float h) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < particle_count; index += blockDim.x * gridDim.x) {
		const float3 position = float4to3(particles[index].position);
		const float3 velocity = float4to3(particles[index].velocity);
		const int linear_index = compute_grid(position, cell_size, grid_size);
		const int3 current_index = grid_index(linear_index, grid_size);

		for (int x = -1; x < 2; ++x)
			for (int y = -1; y < 2; ++y)
				for (int z = -1; z < 2; ++z) {

					const int3 current = make_int3(current_index.x + x, current_index.y + y, current_index.z + z);

					if (current.x < 0 || current.x > grid_size - 1) continue;
					if (current.y < 0 || current.y > grid_size - 1) continue;
					if (current.z < 0 || current.z > grid_size - 1) continue;

					const int current_linear = linear_grid_index(current.x, current.y, current.z, grid_size);

					const int cell_start = cell_starts[current_linear];
					const int cell_end = cell_ends[current_linear];

					// viscosity over neighbours
					for (int i = cell_start; i < cell_end && i < cell_start + MAX_PARTICLES; ++i) {

						const int particle_index = particle_ids_sorted[i];
						if (particle_index > index) continue;

						const float3 other_pos = float4to3(particles[particle_index].position);
						auto pos_diff = other_pos - position;
						const float q = length(pos_diff) / h;
						
						if (q < 1) {
							const float3 other_velocity = float4to3(particles[particle_index].velocity);
							const auto u = dot((velocity - other_velocity), pos_diff);
							
							if (u > 0) {
								normalize(pos_diff);
								auto I = delta_time * (1.0f - q) * (sigma * u + beta * u * u) * pos_diff;
								I *= 0.5f;
								particles[index].velocity = particles[index].velocity - float3to4(I);
								particles[particle_index].velocity = particles[particle_index].velocity + float3to4(I);
							}
						}
					}
				}
	}
}

__global__ void slow_down(particle *particles, int particle_count) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	// slow down any particles which achieved unrealistic speeds
	if (length(float4to3(particles[index].velocity)) > 30.0f) {
		normalize(particles[index].velocity);
		particles[index].velocity = particles[index].velocity * 30.0f;
	}
}

__device__ void dampen(particle &p, float grid_size) {
	float damping = -0.3f;
	float epsilon = 0.0001f;

	if (p.position.x - epsilon < 0.0f) {
		p.velocity.x *= damping;
		p.position.x = epsilon;
	}
	else if (p.position.x + epsilon > grid_size - 1.0f) {
		p.velocity.x *= damping;
		p.position.x = grid_size - 1 - epsilon;
	}

	if (p.position.y - epsilon < 0.0f) {
		p.velocity.y *= damping;
		p.position.y = epsilon;
	}
	else if (p.position.y + epsilon > grid_size - 1.0f) {
		p.velocity.y *= damping;
		p.position.y = grid_size - 1 - epsilon;
	}
	
	if (p.position.z - epsilon < 0.0f) {
		p.velocity.z *= damping;
		p.position.z = epsilon;
	}
	else if (p.position.z + epsilon > grid_size - 1.0f) {
		p.velocity.z *= damping;
		p.position.z = grid_size - 1 - epsilon;
	}
}

__global__ void pos_update(particle *particles, int particle_count, float4 *old_posis, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	old_posis[index] = particles[index].position;
	particles[index].position += delta_time * particles[index].velocity;
}

__global__ void gpu_step_first(const particle *particles, const int *cell_starts, const int *cell_ends, const int *particle_ids_sorted, float *rohs, float *roh_nears, float cell_size, int grid_size, int particle_count, float h) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < particle_count; index += blockDim.x * gridDim.x) {
		const float3 position = float4to3(particles[index].position);
		const float3 velocity = float4to3(particles[index].velocity);
		const int linear_index = compute_grid(position, cell_size, grid_size);
		const int3 current_index = grid_index(linear_index, grid_size);

		// double density
		float roh = 0;
		float roh_near = 0;

		for (int x = -1; x < 2; ++x)
			for (int y = -1; y < 2; ++y)
				for (int z = -1; z < 2; ++z) {

					int3 current = make_int3(current_index.x + x, current_index.y + y, current_index.z + z);

					if (current.x < 0 || current.x > grid_size - 1) continue;
					if (current.y < 0 || current.y > grid_size - 1) continue;
					if (current.z < 0 || current.z > grid_size - 1) continue;

					int current_linear = linear_grid_index(current.x, current.y, current.z, grid_size);

					for (int i = cell_starts[current_linear]; i < cell_ends[current_linear] && i < cell_starts[current_linear] + MAX_PARTICLES; ++i) {

						int particle_index = particle_ids_sorted[i];
						if (particle_index == index) continue;
						
						float q = length(float4to3(particles[particle_index].position - particles[index].position)) / h;

						if (q < 1) {
							roh += (1 - q) * (1 - q);
							roh_near += (1 - q) * (1 - q) * (1 - q);
						}
					}
				}
		
		rohs[index] = roh;
		roh_nears[index] = roh_near;
	}
}

__global__ void gpu_step_second(particle *particles, const int *cell_starts, const int *cell_ends, const int *particle_ids_sorted, const float *rohs, const float *roh_nears, float cell_size, int grid_size, int particle_count, float delta_time, float h, float k, float roh_0, float k_near) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < particle_count; index += blockDim.x * gridDim.x) {
		const float3 position = float4to3(particles[index].position);
		const float3 velocity = float4to3(particles[index].velocity);
		const int linear_index = compute_grid(position, cell_size, grid_size);
		const int3 current_index = grid_index(linear_index, grid_size);

		float P = k * (rohs[index] - roh_0);
		float P_near = k_near * roh_nears[index];
		float3 d_x = float3 {0.0f, 0.0f, 0.0f};

		for (int x = -1; x < 2; ++x)
			for (int y = -1; y < 2; ++y)
				for (int z = -1; z < 2; ++z) {

					int3 current = make_int3(current_index.x + x, current_index.y + y, current_index.z + z);

					if (current.x < 0 || current.x > grid_size - 1) continue;
					if (current.y < 0 || current.y > grid_size - 1) continue;
					if (current.z < 0 || current.z > grid_size - 1) continue;

					int current_linear = linear_grid_index(current.x, current.y, current.z, grid_size);

					for (int i = cell_starts[current_linear]; i < cell_ends[current_linear] && i < cell_starts[current_linear] + MAX_PARTICLES; ++i) {

						int particle_index = particle_ids_sorted[i];
						if (particle_index == index) continue;

						float3 pos_diff = float4to3(particles[particle_index].position - particles[index].position);
						float q = length(float4to3(particles[particle_index].position - particles[index].position)) / h;

						if (q < 1) {
							normalize(pos_diff);
							float3 D = delta_time * delta_time * (P * (1 - q) + P_near * (1 - q) * (1 - q)) * pos_diff;
							D *= 0.5f;
							particles[particle_index].position = particles[particle_index].position + float3to4(D);
							d_x = d_x - D;
						}
					}
				}

		// compute next velocity and set positions
		particles[index].position = particles[index].position + float3to4(d_x);
	}
}

__global__ void update_velocity(particle *particles, int particle_count, float4 *old_posis, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	particles[index].velocity = (particles[index].position - old_posis[index]) * (1 / delta_time);
}

__global__ void count_me(int *cell_starts, int *cell_ends, int particle_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int count = cell_ends[index] - cell_starts[index];

	if (count > 5)
		printf("%d\n", count);
}

__global__ void apply_gravity(particle *particles, int particle_count, float3 gravity, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	particles[index].velocity = particles[index].velocity + delta_time * float3to4(gravity);
}

__global__ void dampen_all(particle *particles, int particle_count, float grid_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	dampen(particles[index], grid_size);
}

__global__ void count_neighbours(particle *particles, int *cell_starts, int *cell_ends, int particle_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int count = cell_ends[index] - cell_starts[index];

	if (count > 50) {
		printf("%d\n", count);
	}
}

static int *cell_ids;
static int *particle_ids;
static int *cell_ids_sorted;
static int *particle_ids_sorted;
static int *cell_starts;
static int *cell_ends;
static float4 *old_posis;
static void *temp_storage;
static size_t temp_storage_bytes;
static float *rohs;
static float *roh_nears;

void init(int count, int grid_size) {

	int grid_count = grid_size * grid_size * grid_size;

	CUDA_CHECK(cudaMalloc((void**)&cell_ids, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&particle_ids, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&cell_ids_sorted, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&particle_ids_sorted, count * sizeof(int)))
	
	CUDA_CHECK(cudaMalloc((void**)&cell_starts, grid_count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&cell_ends, grid_count * sizeof(int)))

	CUDA_CHECK(cudaMalloc((void**)&old_posis, count * sizeof(float4)))

	CUDA_CHECK(cudaMalloc((void**)&rohs, count * sizeof(float)))
	CUDA_CHECK(cudaMalloc((void**)&roh_nears, count * sizeof(float)))
}

void fin() {
	CUDA_CHECK(cudaFree(cell_ids))
	CUDA_CHECK(cudaFree(particle_ids))
	CUDA_CHECK(cudaFree(cell_ids_sorted))
	CUDA_CHECK(cudaFree(particle_ids_sorted))
	CUDA_CHECK(cudaFree(cell_starts))
	CUDA_CHECK(cudaFree(cell_ends))
	CUDA_CHECK(cudaFree(old_posis))
	CUDA_CHECK(cudaFree(rohs))
	CUDA_CHECK(cudaFree(roh_nears))
}

void cuda_fast_step(particle *device_pointer, vector<particle> &particles, float cell_size, int grid_size, float delta_time, float h, 
				float sigma, float beta, float k, float roh_0, float k_near, float3 gravity, std::map<std::string, std::vector<float>> &times) {

	int block_size = 256;
	int num_blocks = (particles.size() + block_size - 1) / block_size;
	int grid_count = grid_size * grid_size * grid_size;
	
	CUDA_CHECK(cudaMemset(cell_starts, -1, grid_count * sizeof(int)))
	CUDA_CHECK(cudaMemset(cell_ends, -1, grid_count * sizeof(int)))

	CUDA_TIME((bin_particles<<<num_blocks, block_size>>>(device_pointer, particles.size(), cell_ids, particle_ids, cell_size, grid_size)), "binning");
	CUDA_TIME(cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, cell_ids, cell_ids_sorted, particle_ids, particle_ids_sorted, particles.size()), "sort 1");
	CUDA_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));
	CUDA_TIME(cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, cell_ids, cell_ids_sorted, particle_ids, particle_ids_sorted, particles.size()), "sort 2");
	CUDA_CHECK(cudaFree(temp_storage))

	CUDA_TIME((find_cell_ranges<<<num_blocks, block_size>>>(cell_ids_sorted, particle_ids_sorted, cell_starts, cell_ends, particles.size())), "cell ranges");
	CUDA_TIME((apply_gravity<<<num_blocks, block_size>>>(device_pointer, particles.size(), gravity, delta_time)), "gravity");

	int blocks, threads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, viscosity);
	CUDA_TIME((viscosity<<<blocks, threads>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, particles.size(), cell_size, grid_size, delta_time, sigma, beta, h)), "viscosity");
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_TIME((slow_down<<<num_blocks, block_size>>>(device_pointer, particles.size())), "slow down");
	CUDA_TIME((pos_update<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "pos update");
	
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, gpu_step_first);
	CUDA_TIME((gpu_step_first<<<blocks, threads>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, rohs, roh_nears, cell_size, grid_size, particles.size(), h)), "step 1");
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, gpu_step_second);
	CUDA_TIME((gpu_step_second<<<blocks, threads>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, rohs, roh_nears, cell_size, grid_size, particles.size(), delta_time, h, k, roh_0, k_near)), "step 2");
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_TIME((update_velocity<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "update velocity");
	CUDA_TIME((slow_down<<<num_blocks, block_size>>>(device_pointer, particles.size())), "slow");
	CUDA_TIME((dampen_all<<<num_blocks, block_size>>>(device_pointer, particles.size(), grid_size)), "dampen");
}
