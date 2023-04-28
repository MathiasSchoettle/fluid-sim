#version 150

in vec4 in_pos;

uniform mat4 view;
uniform mat4 proj;
uniform vec2 screen_size;
uniform float sprite_size;

out float size;
out vec4 world;

void main() {
	world = in_pos;
	gl_Position = proj * view * in_pos;
	gl_PointSize = screen_size.y * (1.0 / tan(45.0)) * sprite_size / gl_Position.w;
	size = sprite_size;
}
