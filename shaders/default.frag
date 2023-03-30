#version 150

out vec4 out_col;
uniform vec4 cols[100];

void main() {
	out_col = cols[gl_PrimitiveID % 100];
}
