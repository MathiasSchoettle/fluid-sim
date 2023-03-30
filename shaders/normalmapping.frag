#version 150

uniform vec3 dirlight_col;
uniform float dirlight_scale;

uniform vec3 pointlight_col;
uniform float pointlight_scale;

uniform sampler2D diffuse;
uniform sampler2D specular;

uniform int has_alphamap;
uniform sampler2D alphamap;

uniform int has_normalmap;
uniform sampler2D normalmap;

in vec3 v;
in vec3 to_dirlight;
in vec3 to_pointlight;

in vec2 tc;
out vec4 out_col;

float diff(vec3 l, vec3 n) {
	return max(0,dot(l, n));
}

float spec(vec3 l, vec3 n, float n_s) {
	vec3 r = 2*dot(n,l)*n - l;
	return pow(max(0, dot(r, v)), n_s);
}

void main() {
	if (has_alphamap == 1) {
		float alpha = texture(alphamap, tc).r;
		if (alpha < 0.5) {
			discard;
			return;
		}
	}
	vec3 N = vec3(0,0,1);
	if (has_normalmap == 1) {
		N = texture(normalmap, tc).rgb * vec3(2.0) - vec3(1.0);
	}

	// directional lighting components
	vec3 diff_dl = diff(to_dirlight, N)    * dirlight_col * dirlight_scale;
	vec3 spec_dl = spec(to_dirlight, N, 4) * dirlight_col * dirlight_scale;

	// point lighting components
	float dist = length(to_pointlight);
	vec3 to_light = normalize(to_pointlight);
	vec3 diff_pl = diff(to_light, N)     / (dist/100) * pointlight_col * pointlight_scale;
	vec3 spec_pl = spec(to_light, N, 10) / (dist/100) * pointlight_col * pointlight_scale;

	// material color
	vec3 k_diff = texture(diffuse, tc).rgb;
	vec3 k_spec = texture(specular, tc).rrr;

	out_col = vec4(k_diff * (diff_pl + diff_dl) + k_spec * (spec_pl + spec_dl), 1.0);

	// Debug: So kann die normale als Farbe ausgegeben werden
	// if (has_normalmap == 1)
	// 	out_col = vec4(N,0);
}
