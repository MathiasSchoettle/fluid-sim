#version 440 core

in vec2 tex_coords;

uniform sampler2D g_col_particle;
uniform float n;
uniform float f;
uniform vec2 res;

uniform bool horizontal;
uniform float weight[7] = float[] (0.001, 0.020, 0.109, 0.172, 0.109, 0.020, 0.001);

vec3 light_color = vec3(1);
float ambient_strength = 0.3;
float specular_strength = 2;

out vec4 out_color;

void main()
{
	vec2 tex_offset = 1.0 / res;
	vec4 result = texture(g_col_particle, tex_coords).rgba * weight[0];

	if (result.w == 0) return;

	if(horizontal) {
		for(int i = 1; i < 7; ++i) {
			result += texture(g_col_particle, tex_coords + vec2(tex_offset.x * i, 0.0)).rgba * weight[i];
			result += texture(g_col_particle, tex_coords - vec2(tex_offset.x * i, 0.0)).rgba * weight[i];
		}
	}
	else {
		for(int i = 1; i < 7; ++i) {
			result += texture(g_col_particle, tex_coords + vec2(0.0, tex_offset.y * i)).rgba * weight[i];
			result += texture(g_col_particle, tex_coords - vec2(0.0, tex_offset.y * i)).rgba * weight[i];
		}
	}
	out_color = result;
}
