#define CUDA_CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		std::cerr << "CUDA error occurred: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl;\
		exit(1);\
	}\
}

// #define CUDA_TIME(call, name) call;

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
__device__ __inline__ float4& operator-=(float4 &l, const float4 &r) { l.x-=r.x, l.y-=r.y, l.z-=r.z, l.w-=r.w; return l; }

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

__forceinline__ __device__ void normalize(float4 &v) {
	float inv_len = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w);
	v.x = v.x * inv_len;
	v.y = v.y * inv_len;
	v.z = v.z * inv_len;
	v.w = v.w * inv_len;
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
