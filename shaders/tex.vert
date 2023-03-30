#version 150

in vec3 in_pos;
in vec3 in_norm;
in vec2 in_tc;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view;
uniform mat4 view_normal;
uniform mat4 proj;

out vec3 pos_ws;
out vec3 n_ws;
out vec2 tc;


void main() {
	n_ws = normalize(mat3(model_normal) * in_norm);
	pos_ws = (model * vec4(in_pos, 1.0)).xyz;
	tc = in_tc;
	gl_Position = proj * view * model * vec4(in_pos, 1.0);
}
