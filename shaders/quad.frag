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
float ambient_strength = 0.25;
float specular_strength = 1;

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

	vec3 normal = normalize(norm);
	vec3 normal_vis = (normal + vec3(1.0, 1.0, 1.0)) / 2.0;

	vec3 ambient = ambient_strength * light_color;
	vec3 diffuse = max(-dot(normal, light_dir), 0.0) * light_color;
	vec3 ref = reflect(light_dir, normal);

	float spec = pow(max(ref.z, 0.0), 64);
	vec3 specular = specular_strength * spec * light_color;

	vec3 color_w = col.xyz / col.a;
	vec3 res = (diffuse + ambient + specular) * color_w;

	out_color = normal_vis;
	out_color = res;
}
