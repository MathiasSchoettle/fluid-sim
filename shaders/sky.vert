#version 150

in vec3 in_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 pos_wc;

void main() {
	pos_wc = (model*vec4(in_pos,1.0)).xyz;
	gl_Position = proj * view * model * vec4(in_pos, 1.0);
}
