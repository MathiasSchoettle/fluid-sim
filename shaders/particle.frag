#version 450

layout (location = 0) out vec4 g_col;

uniform float sprite_size;
uniform float f;
uniform float n;
uniform ivec2 screen_size;
uniform sampler2D g_depth;
uniform mat4 invProj;
uniform mat4 invView;

in vec3 color;

#define M_PI 3.1415926535897932384626433832795

float linear_depth(float depth)
{
	depth = 2.0 * depth - 1.0;
	return 2.0 * n * f / (f + n - depth * (f - n));
}

float depth_sample(float linear_depth)
{
	float non_linear_depth = (f + n - 2.0 * n * f / linear_depth) / (f - n);
	return (non_linear_depth + 1.0) / 2.0;
}

void main() {
	vec2 distance = gl_PointCoord - vec2(0.5, 0.5);
	float mag_sq = dot(distance, distance);
	
	if (mag_sq > 0.25)
		discard;

	vec2 distance_world_space = distance * sprite_size * 2;
	float depth_offset = sqrt(sprite_size * sprite_size - dot(distance_world_space, distance_world_space));


	float linear_depth_particle = linear_depth(gl_FragCoord.z);
	// gl_FragDepth = depth_sample(linear_depth_particle - depth_offset);

	vec4 other = texture(g_depth, gl_FragCoord.xy / screen_size);

	float linear_depth_wall = linear_depth(other.z);

	float alpha = cos(length(distance) * 2);

	vec3 world_pos = vec3(distance_world_space, linear_depth_wall - linear_depth_particle) / sprite_size;

	float vec_length = length(world_pos);
	
	float pi_half = M_PI / 2.0;

	float temp = vec_length * pi_half + pi_half;

	alpha = cos(temp) + 1;

	// back
	if (linear_depth_particle < linear_depth_wall && linear_depth_particle + depth_offset > linear_depth_wall) {
		if (other.z != 0) {
			g_col = vec4(1, 1, 1, alpha);
			return;
		}
	}

	// front
	if (linear_depth_particle > linear_depth_wall && linear_depth_particle - depth_offset < linear_depth_wall) {
		if (other.z != 0) {
			g_col = vec4(1, 1, 1, alpha);
			return;
		}
	}
}
