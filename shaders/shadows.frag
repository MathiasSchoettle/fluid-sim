#version 150

uniform vec3 dirlight_col;
uniform float dirlight_scale;

uniform vec3 heli_col;
uniform float heli_scale;

uniform sampler2D diffuse;
uniform sampler2D specular;

uniform int has_alphamap;
uniform sampler2D alphamap;

uniform int has_normalmap;
uniform sampler2D normalmap;

uniform mat4 shadow_V;
uniform mat4 shadow_P;
uniform sampler2DShadow shadowmap;

in vec3 v;
in vec3 to_dirlight;
in vec3 pos_ws;

in vec2 tc;
out vec4 out_col;

float diff(vec3 l, vec3 n) {
	return max(0,dot(l, n));
}

float spec(vec3 l, vec3 n, float n_s) {
	vec3 r = 2*dot(n,l)*n - l;
	return pow(max(0, dot(r, v)), n_s);
}

float shadow_coeff(mat4 m, sampler2DShadow s) {
	vec4 shadow_tc = m*vec4(pos_ws,1);
	shadow_tc /= shadow_tc.w;
	shadow_tc.xyz = (shadow_tc.xyz + vec3(1)) * vec3(0.5);
	float shadow = texture(s, shadow_tc.xyz);
	return shadow;
}

void main() {
	if (has_alphamap == 1) {
		float alpha = texture(alphamap, tc).r;
		if (alpha < 0.5) discard;
	}

	float shadow  = shadow_coeff(shadow_P*shadow_V, shadowmap);
	shadow = 0.2+0.8*shadow;

	vec3 N = vec3(0,0,1);
	if (has_normalmap == 1) {
		N = texture(normalmap, tc).rgb * vec3(2.0) - vec3(1.0);
	}

	vec3 diff_dl = diff(to_dirlight, N)    * dirlight_col * dirlight_scale * shadow;
	vec3 spec_dl = spec(to_dirlight, N, 40) * dirlight_col * dirlight_scale * shadow;

	vec3 k_diff = pow(texture(diffuse, tc).rgb, vec3(2.2));
	vec3 k_spec = texture(specular, tc).rrr;

	out_col = vec4(k_diff * (diff_dl) + k_spec * (spec_dl), 1.0);
	out_col = vec4(pow(out_col.rgb, vec3(1.0/2.2)), 1.0);
}
