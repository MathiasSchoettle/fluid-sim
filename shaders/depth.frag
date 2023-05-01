#version 440

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
	vec2 distance = gl_PointCoord - vec2(0.5, 0.5);
	float mag_sq = dot(distance, distance);
	
	if (mag_sq > 0.25)
		discard;

	vec2 distance_world_space = distance * 2 * sprite_size;
	float depth_offset = sqrt(sprite_size * sprite_size - dot(distance_world_space, distance_world_space));
	gl_FragDepth = depth_sample(linear_depth(gl_FragCoord.z) - depth_offset + eps);
}