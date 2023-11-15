#version 440 core

in vec2 tex_coords;

uniform sampler2D g_norm;
uniform sampler2D g_col;
uniform sampler2D g_col_particle;
uniform sampler2D g_depth;
uniform float n;
uniform float f;
uniform vec2 res;
uniform vec3 light_dir;

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

vec3 get_color() {
	vec3 scene_color = texture(g_col, tex_coords).rgb;
	vec4 particle_color = texture(g_col_particle, tex_coords);

	if (length(particle_color) == 0) {
		return scene_color;
	} else {

		float offset_x = 1.0f / res.x;
		float offset_y = 1.0f / res.y;
		float L, R, T, B;

		L = length(texture(g_col_particle, tex_coords - vec2(offset_x, 0)).rgba);
		R = length(texture(g_col_particle, tex_coords + vec2(offset_x, 0)).rgba);
		T = length(texture(g_col_particle, tex_coords + vec2(0, offset_y)).rgba);
		B = length(texture(g_col_particle, tex_coords - vec2(0, offset_y)).rgba);

		vec3 normal = vec3(2*(R-L), 2*(T-B), 0);

		if (length(normal) != 0) normal = normalize(normal);

		vec3 ambient = ambient_strength * light_color;
		vec3 diffuse = max(-dot(normal, light_dir), 0.0) * light_color;
		vec3 ref = reflect(light_dir, normal);
		float spec = pow(max(ref.z, 0.0), 64);
		vec3 specular = specular_strength * spec * light_color;

		vec3 res = (diffuse + ambient + specular) * vec3(1, 0.5, 0.1);

		return res;

		return vec3(normal);
		return vec3(particle_color);
		return vec3(scene_color + particle_color.xyz * particle_color.w);
	}
}

void main()
{
	vec3 col = texture(g_col, tex_coords).rgb;
	vec3 norm = texture(g_norm, tex_coords).rgb;
	vec3 depth = texture(g_depth, tex_coords).rgb;
	
	out_color = depth;
	out_color = norm;
	out_color = get_color();
}
