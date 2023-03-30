#version 150

uniform vec3 cam_pos;

uniform vec3 dirlight_dir;
uniform vec3 dirlight_col;
uniform float dirlight_scale;

uniform vec3 pointlight_pos;
uniform vec3 pointlight_col;
uniform float pointlight_scale;


in vec3 pos_ws;
in vec3 n_ws;

uniform sampler2D diffuse;
uniform sampler2D specular;
in vec2 tc;

out vec4 out_col;

float diff(vec3 l) {
	return max(0,dot(l, n_ws));
}
float spec(vec3 l, float n_s) {
	vec3 v = normalize(cam_pos - pos_ws);
	vec3 r = 2*dot(n_ws,l)*n_ws - l;
	return pow(max(0, dot(r, v)), n_s);
}

void main() {
	// directional lighting components
	vec3 diff_dl = diff(-dirlight_dir)    * dirlight_col * dirlight_scale;
	vec3 spec_dl = spec(-dirlight_dir, 4) * dirlight_col * dirlight_scale;

	// point lighting components
	vec3 to_light = pointlight_pos - pos_ws;
	float dist = length(to_light);
	to_light = normalize(to_light);
	vec3 diff_pl = diff(to_light)     / (dist/100) * pointlight_col * pointlight_scale;
	vec3 spec_pl = spec(to_light, 10) / (dist/100) * pointlight_col * pointlight_scale;

	// material color
	vec3 k_diff = texture(diffuse, tc).rgb;
	vec3 k_spec = texture(specular, tc).rrr;

	out_col = vec4(k_diff * (diff_pl + diff_dl) + k_spec * (spec_pl + spec_dl), 1.0);
}
