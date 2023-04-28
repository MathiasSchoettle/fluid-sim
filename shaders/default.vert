#version 440

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 velocity;

uniform mat4 view;
uniform mat4 proj;
uniform ivec2 screen_size;
uniform float sprite_size;

out float size;
out vec4 world;

void main() {
	world = vec4(position, 1);
	gl_Position = proj * view * vec4(position, 1);
	gl_PointSize = screen_size.y * (1.0 / tan(45.0)) * sprite_size / gl_Position.w;
	size = sprite_size;
}
