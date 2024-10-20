#version 150

in vec3 in_pos;
in vec3 in_norm;
in vec2 in_tc;
in vec3 in_tan;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view;
uniform mat4 view_normal;
uniform mat4 proj;

uniform vec3 cam_pos;
uniform vec3 dirlight_dir;

out vec3 pos_ws;
out vec3 to_dirlight;
out vec3 v;
out vec2 tc;

void main() {
	vec3 bi_tan = cross(in_norm, in_tan);
	vec3 T = normalize(vec3(model_normal * vec4(in_tan, 0)));
	vec3 B = normalize(vec3(model_normal * vec4(bi_tan, 0)));
	vec3 N = normalize(vec3(model_normal * vec4(in_norm, 0)));
	mat3 ts = transpose(mat3(T, B, N));

	pos_ws = (model * vec4(in_pos, 1.0)).xyz;

	v = normalize(ts * (cam_pos - pos_ws));
	to_dirlight = normalize(ts * -dirlight_dir);

	tc = in_tc;
	gl_Position = proj * view * vec4(pos_ws, 1);
}
