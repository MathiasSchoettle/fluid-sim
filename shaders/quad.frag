#version 440 core

in vec2 tex_coords;

uniform sampler2D g_norm;
uniform sampler2D g_col;
uniform sampler2D g_depth;
uniform float n;
uniform float f;
uniform vec3 light_dir;

uniform mat4 view;

vec3 light_color = vec3(1);
float ambient_strength = 0.3;
float specular_strength = 2;

out vec3 out_color;

float linearDepth(float depthSample)
{
	depthSample = 2.0 * depthSample - 1.0;
	float zLinear = 2.0 * n * f / (f + n - depthSample * (f - n));
	return zLinear;
}

void main()
{
	vec4 col = texture(g_col, tex_coords);
	vec3 norm = texture(g_norm, tex_coords).rgb;
	vec3 depth = texture(g_depth, tex_coords).rgb;

	out_color = vec3(1, 0, 1);
}
