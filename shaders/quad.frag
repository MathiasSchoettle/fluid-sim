#version 440 core

in vec2 tex_coords;

uniform sampler2D g_pos;
uniform sampler2D g_norm;
uniform sampler2D g_col;

out vec4 out_color;

void main()
{
	vec3 col = texture(g_col, tex_coords).rgb;
	out_color = vec4(col, 1);
}
