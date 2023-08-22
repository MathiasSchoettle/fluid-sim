#version 440

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 velocity;
layout (location = 2) in vec4 col;

uniform mat4 view;
uniform mat4 proj;
uniform ivec2 screen_size;
uniform float sprite_size;

out vec3 color;

void main() {
	gl_Position = proj * view * position;
	gl_PointSize = screen_size.y * (1.0 / tan(45.0)) * sprite_size / gl_Position.w;
	color = col.xyz;
}
