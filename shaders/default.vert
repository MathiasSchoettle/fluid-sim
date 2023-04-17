#version 150

in vec4 in_pos;

uniform mat4 view;
uniform mat4 proj;
uniform vec2 screen_size;
uniform float sprite_size;

void main() {
	vec4 eyePos = view * in_pos;
	vec4 projVoxel = proj * vec4(sprite_size,sprite_size,eyePos.z,eyePos.w);
	vec2 projSize = screen_size * projVoxel.xy / projVoxel.w;
	gl_PointSize = 0.25 * (projSize.x+projSize.y);
	gl_Position = proj * eyePos;
}
