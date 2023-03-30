#version 150

uniform vec4 k_diff;
uniform vec4 k_spec;
uniform vec3 dirlight_dir;
uniform vec3 dirlight_col;
uniform float dirlight_scale;
uniform vec3 pointlight_pos;
uniform vec3 pointlight_col;
uniform float pointlight_scale;
uniform vec3 cam_pos;

in vec3 pos_ws;
in vec3 n_ws;

out vec4 out_col;

void main() {
	vec3 diff = vec3(0);
	vec3 spec = vec3(0);
	vec3 v = normalize(cam_pos - pos_ws);
	vec3 n = normalize(n_ws);

	// directional light
	float n_dot_l = dot(-dirlight_dir, n);
	if (n_dot_l > 0) {
		diff += k_diff.rgb * n_dot_l * dirlight_col * dirlight_scale;

		vec3 r = 2*dot(n,-dirlight_dir)*n + dirlight_dir;
		spec += pow(max(0, dot(r, v)), 4) * k_spec.rgb * dirlight_col * dirlight_scale;
	}

	vec3 to_light = pointlight_pos - pos_ws;
	float dist = length(to_light);
	to_light = normalize(to_light);
	n_dot_l = dot(to_light, n);
	if (n_dot_l > 0) {
		diff += (n_dot_l / (dist/100)) * k_diff.rgb * pointlight_col * pointlight_scale;

		vec3 r = 2*dot(n,to_light)*n - to_light;
		spec += pow(max(0, dot(r, v)), 140) * k_spec.rgb * pointlight_col * pointlight_scale;
	}

	out_col = vec4(diff*k_diff.rgb + spec*k_spec.rgb,1);
}
