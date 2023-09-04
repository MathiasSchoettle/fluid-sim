#version 440

layout (location = 1) out vec4 g_col;
layout (location = 2) out vec4 g_depth;

uniform float sprite_size;
uniform float f;
uniform float n;
uniform float eps;

in vec3 color;

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
	vec2 distance = gl_PointCoord - vec2(0.5);
	
	if (length(distance) > 0.5)
		discard;

	vec2 distance_world_space = distance * 2 * sprite_size;
	float depth_offset = sqrt(sprite_size * sprite_size - dot(distance_world_space, distance_world_space));
	float depth = linear_depth(gl_FragCoord.z) - depth_offset;

	g_col = vec4(color, 1);

	float linear_depth = linear_depth(gl_FragCoord.z) - depth_offset + eps;
	// g_depth.xy = distance_world_space;
	// g_depth.z = depth_offset;
	gl_FragDepth = depth_sample(linear_depth);
	g_depth.w = isnan(depth) ? 0 : depth;
}