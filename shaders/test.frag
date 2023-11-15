#version 450

layout (location = 0) out vec4 g_norm;
layout (location = 1) out vec4 g_col;
layout (location = 2) out vec4 g_depth;

uniform vec3 cam_pos;
uniform float n;
uniform float f;

in vec3 pos_ws;
in vec3 n_ws;

out vec4 out_col;

const vec3 dirlight_dir = normalize(vec3(0.5,-.43,.7));
const vec3 dirlight_col = vec3(1.0,0.97,0.8);
const float dirlight_scale = 1.2f;
const vec4 k_diff = vec4(1);
const vec4 k_spec = vec4(1);

void main() {
	vec3 diff = vec3(0.1);
	vec3 spec = vec3(0);
	vec3 v = normalize(cam_pos - pos_ws);
	vec3 n = normalize(n_ws);

	float n_dot_l = dot(-dirlight_dir, n);
	if (n_dot_l > 0) {
		diff += k_diff.rgb * n_dot_l * dirlight_col * dirlight_scale;

		vec3 r = 2*dot(n,-dirlight_dir)*n + dirlight_dir;
		spec += pow(max(0, dot(r, v)), 4) * k_spec.rgb * dirlight_col * dirlight_scale;
	}

	g_norm = vec4(n_ws, 1);
	g_col = vec4(diff * k_diff.rgb, 1);
	g_depth = gl_FragCoord;
}
