#version 150

in vec4 in_pos;

uniform mat4 view;
uniform mat4 proj;


void main() {
	vec2 screenSize = vec2(1280, 720);
	float sprite_size = 20s;

	vec4 eyePos = view * in_pos;
	vec4 projVoxel = proj * vec4(sprite_size,sprite_size,eyePos.z,eyePos.w);
	vec2 projSize = screenSize * projVoxel.xy / projVoxel.w;
	gl_PointSize = 0.25 * (projSize.x+projSize.y);
	gl_Position = proj * eyePos;
}
