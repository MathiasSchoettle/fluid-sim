#version 440

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 velocity;
layout (location = 2) in vec3 col;

uniform mat4 view;
uniform mat4 proj;
uniform ivec2 screen_size;
uniform float sprite_size;

out vec3 color;

void main() {
	gl_Position = proj * view * vec4(position, 1);
	gl_PointSize = screen_size.y * (1.0 / tan(45.0)) * sprite_size / gl_Position.w;
	color = col;
}
