#version 440 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 tex;

out vec2 tex_coords;

void main()
{
	tex_coords = tex;
	gl_Position = vec4(pos, 1.0);
}
