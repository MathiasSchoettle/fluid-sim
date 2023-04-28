#version 440

in float size;
in vec4 world;

uniform mat4 view;
uniform mat4 proj;
uniform float f;
uniform float n;

out vec4 out_col;

float linear_depth(float depth) {
	depth = 2.0 * depth - 1.0;
	return 2.0 * n * f / (f + n - depth * (f - n));
}

float depth_sample(float linear_depth) {
	float non_linear_depth = (f + n - 2.0 * n * f / linear_depth) / (f - n);
	return (non_linear_depth + 1.0) / 2.0;
}

float linearize_depth(float depth) {
	return (2.0 * n) / (f + n - depth * (f - n));
}

void main() {
	vec2 distance = gl_PointCoord - vec2(0.5, 0.5);

	if (dot(distance, distance) > 0.25) {
		discard;
	}

	vec2 world_space = distance * 2 * size;
	float depth_offset = sqrt(size * size - dot(world_space, world_space));

	float epsilon = 0.125; // TODO calculate from point size
	float depth = depth_sample(linear_depth(gl_FragCoord.z) - depth_offset + epsilon);
	gl_FragDepth = depth;
	
	out_col = vec4(linearize_depth(depth));
}
