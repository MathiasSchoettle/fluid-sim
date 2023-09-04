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

float depth_sample(float linear_depth)
{
	float non_linear_depth = (f + n - 2.0 * n * f / linear_depth) / (f - n);
	return (non_linear_depth + 1.0) / 2.0;
}

float linear_depth(float depth)
{
	depth = 2.0 * depth - 1.0;
	return 2.0 * n * f / (f + n - depth * (f - n));
}

void main()
{
	vec4 col = texture(g_col, tex_coords);
	vec3 norm = texture(g_norm, tex_coords).rgb;
	vec4 depth = texture(g_depth, tex_coords);

	vec3 ambient = ambient_strength * light_color;
	vec3 diffuse = max(-dot(norm, light_dir), 0.0) * light_color;
	vec3 ref = reflect(light_dir, norm);

	float spec = pow(max(ref.z, 0.0), 64);
	vec3 specular = specular_strength * spec * light_color;

	vec3 color_w = col.xyz;
	vec3 res = (diffuse + ambient + specular) * color_w;

	out_color = color_w;
	out_color = norm;
	out_color = vec3(depth.w);
	out_color = res;
}
