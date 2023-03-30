#version 150

uniform vec4 k_diff;

out vec4 out_col;

void main() {
	out_col = k_diff;
}
