#version 440

layout (location = 0) out vec4 g_norm;
layout (location = 1) out vec4 g_col;

uniform float sprite_size;
uniform float f;
uniform float n;
uniform ivec2 screen_size;
uniform sampler2D g_depth;
uniform float eps;

in vec3 color;

float linear_depth(float depth)
{
	depth = 2.0 * depth - 1.0;
	return 2.0 * n * f / (f + n - depth * (f - n));
}

void main() {
	vec2 distance = gl_PointCoord - vec2(0.5);
	float mag_sq = dot(distance, distance);
	
	if (mag_sq > 0.25)
		discard;

	vec2 world_space_distance = distance * 2.0 * sprite_size;
	float depth_offset = sqrt(sprite_size * sprite_size - dot(world_space_distance, world_space_distance));

	
	float depth = linear_depth(gl_FragCoord.z) - depth_offset;
	float surface_depth = linear_depth(texture(g_depth, gl_FragCoord.xy / screen_size).x);
}
