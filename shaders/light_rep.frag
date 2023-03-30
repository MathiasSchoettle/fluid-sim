#version 150

uniform vec3 pointlight_col;

out vec4 out_col;

void main() {
	out_col = vec4(pointlight_col,1);
}
