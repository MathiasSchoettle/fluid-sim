#include "simulation-cuda.h"

#include <iostream>
#include <chrono>
#include <cub/cub.cuh> 

__device__ __inline__ float2 operator*(float l, const float2 &r) { return { l*r.x, l*r.y }; }
__device__ __inline__ float2 operator*(const float2 &l, float r) { return { l.x*r, l.y*r }; }
__device__ __inline__ float2 operator*(const float2 &l, const float2 &r) { return { l.x*r.x, l.y*r.y }; }

__device__ __inline__ float2 operator+(const float2 &l, const float2 &r) { return { l.x+r.x, l.y+r.y }; }

__device__ __inline__ float3 operator*(float l, const float3 &r) { return { l*r.x, l*r.y, l*r.z }; }
__device__ __inline__ float3 operator*(const float3 &l, float r) { return { l.x*r, l.y*r, l.z*r }; }
__device__ __inline__ float3 operator*(const float3 &l, const float3 &r) { return { l.x*r.x, l.y*r.y, l.z*r.z }; }

__device__ __inline__ float3& operator*=(float3 &l, const float &r) { l.x*=r, l.y*=r, l.z*=r; return l; }
__device__ __inline__ float3& operator+=(float3 &l, const float3 &r) { l.x+=r.x, l.y+=r.y, l.z+=r.z; return l; }

__device__ __inline__ float3 operator/(const float3 &l, float r) { return { l.x/r, l.y/r, l.z/r }; }
__device__ __inline__ float3 operator/=(float3 &l, float r) { return { l.x/=r, l.y/=r, l.z/=r }; }

__device__ __inline__ float3 operator+(const float3 &l, const float3 &r) { return { l.x+r.x, l.y+r.y, l.z+r.z }; }
__device__ __inline__ float3 operator-(const float3 &l, const float3 &r) { return { l.x-r.x, l.y-r.y, l.z-r.z }; }
__device__ __inline__ float3 operator-(const float3 &l) { return { -l.x, -l.y, -l.z }; }

__device__ __inline__ float4 operator+(const float4 &l, const float4 &r) { return { l.x+r.x, l.y+r.y, l.z+r.z, l.w+r.w }; }
__device__ __inline__ float4 operator-(const float4 &l, const float4 &r) { return { l.x-r.x, l.y-r.y, l.z-r.z, l.w-r.w }; }
__device__ __inline__ float4& operator+=(float4 &l, const float4 &r) { l.x+=r.x, l.y+=r.y, l.z+=r.z, l.w+=r.w; return l; }

__device__ __inline__ float4 operator*(float l, const float4 &r) { return { l*r.x, l*r.y, l*r.z, l*r.w }; }
__device__ __inline__ float4 operator*(const float4 &l, float r) { return { l.x*r, l.y*r, l.z*r, l.w*r }; }
__device__ __inline__ float4 operator*(const float4 &l, const float4 &r) { return { l.x*r.x, l.y*r.y, l.z*r.z, l.w*r.w }; }

__device__ __forceinline__ float3 float4to3(const float4 &value) { return make_float3(value.x, value.y, value.z); }
__device__ __forceinline__ float4 float3to4(const float3 &value) { return make_float4(value.x, value.y, value.z, 0); }

__forceinline__ __device__ float length(const float3 &v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__forceinline__ __device__ void normalize(float3 &v) {
    float inv_len = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x = v.x * inv_len;
    v.y = v.y * inv_len;
    v.z = v.z * inv_len;
}

__forceinline__ __device__ __host__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ __host__ void cross(float3 &dest, const float3 &a, const float3 &b) {
    dest.x = a.y*b.z - a.z*b.y;
    dest.y = a.z*b.x - a.x*b.z;
    dest.z =  a.x*b.y - a.y*b.x;
}

__forceinline__  __device__ __host__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}



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

__device__ __forceinline__ int3 grid_index(int linear_index, const int grid_size) {
	int z = linear_index / (grid_size * grid_size);
	linear_index -= z * (grid_size * grid_size);
	int y = linear_index / grid_size;
	linear_index -= y * grid_size;
	return make_int3(linear_index, y, z);
}

__device__ __forceinline__ int compute_grid(const float3 &pos, float cell_size, int grid_size) {
	int x = pos.x / cell_size;
	int y = pos.y / cell_size;
	int z = pos.z / cell_size;
	return linear_grid_index(x, y, z, grid_size);
}

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

__global__ void viscosity_different(float4 *posis, float4 *velosis, const int *cell_starts, const int *cell_ends, const int *particle_ids_sorted, int particle_count, float cell_size, int grid_size, float delta_time, float sigma, float beta, float h) {
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < particle_count; index += blockDim.x * gridDim.x) {
		const float3 position = float4to3(posis[index]);
		const float3 velocity = float4to3(velosis[index]);
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

						const float3 other_pos = float4to3(posis[particle_index]);
						auto pos_diff = other_pos - position;
						const float q = length(pos_diff) / h;
						
						if (q < 1) {
							const float3 other_velocity = float4to3(velosis[particle_index]);
							const auto u = dot((velocity - other_velocity), pos_diff);
							
							if (u > 0) {
								normalize(pos_diff);
								auto I = delta_time * (1.0f - q) * (sigma * u + beta * u * u) * pos_diff;
								I *= 0.5f;
								velosis[index] = velosis[index] - float3to4(I);
								velosis[particle_index] = velosis[particle_index] + float3to4(I);
							}
						}
					}
				}
	}
}

__global__ void viscosity(float4 *posis, float4 *velosis, const int *cell_starts, const int *cell_ends, const int *particle_ids_sorted, int particle_count, float cell_size, int grid_size, float delta_time, float sigma, float beta, float h) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= particle_count) return;

	const float3 position = float4to3(posis[index]);
	const float3 velocity = float4to3(velosis[index]);
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

					const float3 other_pos = float4to3(posis[particle_index]);
					auto pos_diff = other_pos - position;
					const float q = length(pos_diff) / h;
					
					if (q < 1) {
						const float3 other_velocity = float4to3(velosis[particle_index]);
						const auto u = dot((velocity - other_velocity), pos_diff);
						
						if (u > 0) {
							normalize(pos_diff);
							auto I = delta_time * (1.0f - q) * (sigma * u + beta * u * u) * pos_diff;
							I *= 0.5f;
							velosis[index] = velosis[index] - float3to4(I);
							velosis[particle_index] = velosis[particle_index] + float3to4(I);
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
	float my = 0.025f;

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

	std::vector<particle> p(particles.size());
	cudaMemcpy(p.data(), device_pointer, particles.size() * sizeof(particles), cudaMemcpyDeviceToHost);

	vector<float4> posis;
	vector<float4> velosis;

	for (auto part : p) {
		velosis.push_back(make_float4(part.velocity.x, part.velocity.y,part.velocity.z, 0));
		posis.push_back(make_float4(part.position.x, part.position.y,part.position.z, 0));
	}

	float4 *cuda_posis;
	float4 *cuda_velosis;
	cudaMalloc(&cuda_posis, particles.size() * sizeof(float4));
	cudaMalloc(&cuda_velosis, particles.size() * sizeof(float4));

	cudaMemcpy(cuda_posis, posis.data(), particles.size() * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_posis, posis.data(), particles.size() * sizeof(float4), cudaMemcpyHostToDevice);

	CUDA_TIME((bin_particles<<<num_blocks, block_size>>>(device_pointer, particles.size(), cell_ids, particle_ids, cell_size, grid_size)), "binning");
	CUDA_TIME(cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, cell_ids, cell_ids_sorted, particle_ids, particle_ids_sorted, particles.size()), "sort 1");
	CUDA_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));
	CUDA_TIME(cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, cell_ids, cell_ids_sorted, particle_ids, particle_ids_sorted, particles.size()), "sort 2");
	CUDA_CHECK(cudaFree(temp_storage))

	CUDA_TIME((find_cell_ranges<<<num_blocks, block_size>>>(cell_ids_sorted, particle_ids_sorted, cell_starts, cell_ends, particles.size())), "cell ranges");
	CUDA_TIME((apply_gravity<<<num_blocks, block_size>>>(device_pointer, particles.size(), gravity, delta_time)), "gravity");

	// slow
	// CUDA_TIME((viscosity<<<num_blocks, block_size>>>(cuda_posis, cuda_velosis, cell_starts, cell_ends, particle_ids_sorted, particles.size(), cell_size, grid_size, delta_time, sigma, beta, h)), "viscosity");

	int blocks, threads;
	cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, viscosity_different);
	CUDA_TIME((viscosity_different<<<blocks, threads>>>(cuda_posis, cuda_velosis, cell_starts, cell_ends, particle_ids_sorted, particles.size(), cell_size, grid_size, delta_time, sigma, beta, h)), "viscosity");

	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(posis.data(), cuda_posis, particles.size() * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaMemcpy(velosis.data(), cuda_velosis, particles.size() * sizeof(float4), cudaMemcpyDeviceToHost);

	for (int i = 0; i < particles.size(); ++i) {
		auto pos = posis[i];
		auto vel = velosis[i];
		p[i].position = glm::vec3(pos.x, pos.y, pos.z);
		p[i].velocity = glm::vec3(vel.x, vel.y, vel.z);
	}

	cudaMemcpy(device_pointer, p.data(), particles.size() * sizeof(particles), cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaFree(cuda_posis));
	CUDA_CHECK(cudaFree(cuda_velosis));

	CUDA_TIME((pos_update<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "pos update");

	// even slower	
	CUDA_TIME((gpu_step_fast<<<num_blocks, block_size>>>(device_pointer, cell_starts, cell_ends, particle_ids_sorted, cell_size, grid_size, particles.size(), delta_time, h, k, roh_0, k_near, old_posis)), "step");

	CUDA_TIME((collisions<<<num_blocks, block_size>>>(device_pointer, old_posis, particles.size(), h, delta_time)), "collisions");
	CUDA_TIME((update_velocity<<<num_blocks, block_size>>>(device_pointer, particles.size(), old_posis, delta_time)), "update velocity");
	CUDA_TIME((slow_down<<<num_blocks, block_size>>>(device_pointer, particles.size())), "slow");
	CUDA_TIME((dampen_all<<<num_blocks, block_size>>>(device_pointer, particles.size(), grid_size)), "dampen");
}
