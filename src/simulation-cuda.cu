#include "simulation-cuda.h"

#include <iostream>
#include <chrono>
#include <cub/cub.cuh> 

using namespace std;

#define MAX_PARTICLES 10

#define CUDA_CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		std::cerr << "CUDA error occurred: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl;\
		exit(1);\
	}\
}

#define CUDA_TIME(call, name) \
{ \
	cudaEvent_t start, end; \
	cudaEventCreate(&start); \
	cudaEventCreate(&end); \
	cudaEventRecord(start); \
	call; \
	cudaEventRecord(end); \
	cudaEventSynchronize(end); \
	float time = 0; \
	cudaEventElapsedTime(&time, start, end); \
	if (times[name].size() > MAX_TIMES) times[name].pop_back(); \
	times[name].insert(times[name].begin(), time); \
	cudaEventDestroy(start); \
	cudaEventDestroy(end); \
}

__device__ __forceinline__ int linear_grid_index(int x, int y, int z, const int grid_size) {
	return x + y * grid_size + z * grid_size * grid_size;
}

__device__ int3 grid_index(int linear_index, const int grid_size) {
	int z = linear_index / (grid_size * grid_size);
	linear_index -= z * (grid_size * grid_size);
	int y = linear_index / grid_size;
	linear_index -= y * grid_size;
	return make_int3(linear_index, y, z);
}

__device__ int compute_grid(particle p, float cell_size, int grid_size) {
	int x = p.position.x / cell_size;
	int y = p.position.y / cell_size;
	int z = p.position.z / cell_size;
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

__global__ void viscosity(particle *particles, int *cell_starts, int *cell_ends, int *particle_ids_sorted, int particle_count, float cell_size, int grid_size, float delta_time, float sigma, float beta, float h) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int linear_index = compute_grid(particles[index], cell_size, grid_size);
	int3 current_index = grid_index(linear_index, grid_size);

	for (int x = -1; x < 2; ++x)
		for (int y = -1; y < 2; ++y)
			for (int z = -1; z < 2; ++z) {

				int3 current = make_int3(current_index.x + x, current_index.y + y, current_index.z + z);

				if (current.x < 0 || current.x > grid_size - 1) continue;
				if (current.y < 0 || current.y > grid_size - 1) continue;
				if (current.z < 0 || current.z > grid_size - 1) continue;

				int current_linear = linear_grid_index(current.x, current.y, current.z, grid_size);

				// viscosity over neighbours
				for (int i = cell_starts[current_linear]; i < cell_ends[current_linear] && i < cell_starts[current_linear] + MAX_PARTICLES; ++i) {

					int particle_index = particle_ids_sorted[i];
					if (particle_index > index) continue;

					auto pos_diff = particles[particle_index].position - particles[index].position;
					float q = glm::length(pos_diff) / h;

					if (q < 1) {
						auto u = glm::dot((particles[index].velocity - particles[particle_index].velocity), pos_diff);
						
						if (u > 0) {
							auto I = delta_time * (1.0f - q) * (sigma * u + beta * u * u) * glm::normalize(pos_diff);
							I *= 0.5f;
							particles[index].velocity -= I;
							particles[particle_index].velocity += I;
						}
					}
				}
			}
}

__global__ void slow_down(particle *particles, int particle_count) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	// slow down any particles which achieved unrealistic speeds
	if (glm::length(particles[index].velocity) > 50.0f) {
		particles[index].velocity = glm::normalize(particles[index].velocity) * 50.0f;
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

__global__ void pos_update(particle *particles, int particle_count, glm::vec3 *old_posis, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	old_posis[index] = particles[index].position;
	particles[index].position += delta_time * particles[index].velocity;
}	

__global__ void gpu_step_fast(particle *particles, int *cell_starts, int *cell_ends, int *particle_ids_sorted, float cell_size, int grid_size, int particle_count, float delta_time, float h, float k, float roh_0, float k_near, glm::vec3 *old_posis) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int linear_index = compute_grid(particles[index], cell_size, grid_size);
	int3 current_index = grid_index(linear_index, grid_size);

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
					
					float q = glm::length(particles[particle_index].position - particles[index].position) / h;

					if (q < 1) {
						roh += (1 - q) * (1 - q);
						roh_near += (1 - q) * (1 - q) * (1 - q);
					}
				}
			}

	float P = k * (roh - roh_0);
	float P_near = k_near * roh_near;
	glm::vec3 d_x = glm::vec3(0);

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

					auto pos_diff = particles[particle_index].position - particles[index].position;
					float q = glm::length(particles[particle_index].position - particles[index].position) / h;

					if (q < 1) {
						glm::vec3 D = delta_time * delta_time * (P * (1 - q) + P_near * (1 - q) * (1 - q)) * glm::normalize(pos_diff);
						D *= 0.5f;
						particles[particle_index].position += D;
						d_x = d_x - D;
					}
				}
			}

	// compute next velocity and set positions
	particles[index].position += d_x;
}

__global__ void update_velocity(particle *particles, int particle_count, glm::vec3 *old_posis, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	particles[index].velocity = (particles[index].position - old_posis[index]) / delta_time;
}

__global__ void count_me(int *cell_starts, int *cell_ends, int particle_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int count = cell_ends[index] - cell_starts[index];

	if (count > 5)
		printf("%d\n", count);
}

__global__ void apply_gravity(particle *particles, int particle_count, glm::vec3 gravity, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	particles[index].velocity += delta_time * gravity;
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


// limit the amount of springs per particle maybe only 2 or 3 per adjacent grid element?
// create these and keep track of them, then update using them
__global__ void create_springs(particle *particles, int particle_count, spring *springs, float cell_size, int grid_size, float alpha, float delta_t, float gamma, float h, float L_frac) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	int linear_index = compute_grid(particles[index], cell_size, grid_size);
	int3 current_index = grid_index(linear_index, grid_size);

	for (int x = -1; x < 2; ++x)
		for (int y = -1; y < 2; ++y)
			for (int z = -1; z < 2; ++z) {
				
			}

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

	// // spring deletion
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
}

__global__ void collisions(particle *particles, glm::vec3 *old_posis, int particle_count, float interaction_radius, float delta_time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	particle &current = particles[index];

	float radius = 20;
	glm::vec3 circle(100, 100, 100);
	float my = 0.225f;

	glm::vec3 n = circle - current.position;
	float dist = glm::length(n) - radius;

	if (dist > interaction_radius) return;

	n = glm::normalize(n);
	glm::vec3 old = old_posis[index];
	glm::vec3 v = current.position - old;

	glm::vec3 v_normal = glm::dot(v, n) * n;
	glm::vec3 v_tangent = v - v_normal;

	glm::vec3 I = v_normal - my * v_tangent;

	current.position -= delta_time * I;
}

static int *cell_ids;
static int *particle_ids;
static int *cell_ids_sorted;
static int *particle_ids_sorted;
static int *cell_starts;
static int *cell_ends;
static glm::vec3 *old_posis;
static void *temp_storage;
static size_t temp_storage_bytes;

void init(int count, int grid_size) {

	int grid_count = grid_size * grid_size * grid_size;

	CUDA_CHECK(cudaMalloc((void**)&cell_ids, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&particle_ids, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&cell_ids_sorted, count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&particle_ids_sorted, count * sizeof(int)))
	
	CUDA_CHECK(cudaMalloc((void**)&cell_starts, grid_count * sizeof(int)))
	CUDA_CHECK(cudaMalloc((void**)&cell_ends, grid_count * sizeof(int)))

	CUDA_CHECK(cudaMalloc((void**)&old_posis, count * sizeof(glm::vec3)))
}

void fin() {
	CUDA_CHECK(cudaFree(cell_ids))
	CUDA_CHECK(cudaFree(particle_ids))
	CUDA_CHECK(cudaFree(cell_ids_sorted))
	CUDA_CHECK(cudaFree(particle_ids_sorted))
	CUDA_CHECK(cudaFree(cell_starts))
	CUDA_CHECK(cudaFree(cell_ends))
	CUDA_CHECK(cudaFree(old_posis))
}

void cuda_fast_step(particle *device_pointer, vector<particle> &particles, float cell_size, int grid_size, float delta_time, float h, 
				float sigma, float beta, float k, float roh_0, float k_near, glm::vec3 gravity, std::map<std::string, std::vector<float>> &times) {

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
	CUDA_TIME((viscosity<<<num_blocks, block_size>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, particles.size(), cell_size, grid_size, delta_time, sigma, beta, h)), "viscosity");
	CUDA_TIME((slow_down<<<num_blocks, block_size>>>(device_pointer, particles.size())), "slow down");
	CUDA_TIME((pos_update<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "pos update");
	CUDA_TIME((gpu_step_fast<<<num_blocks, block_size>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, cell_size, grid_size, particles.size(), delta_time, h, k, roh_0, k_near, old_posis)), "step");
	CUDA_TIME((collisions<<<num_blocks, block_size>>>(device_pointer, old_posis, particles.size(), h, delta_time)), "collisions");
	CUDA_TIME((update_velocity<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "update velocity");
	CUDA_TIME((slow_down<<<num_blocks, block_size>>>(device_pointer, particles.size())), "slow");
	CUDA_TIME((dampen_all<<<num_blocks, block_size>>>(device_pointer, particles.size(), grid_size)), "dampen");
}
