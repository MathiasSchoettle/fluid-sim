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

uniform mat4 heli_V;
uniform mat4 heli_P;
uniform sampler2DShadow heli_shadowmap;

in vec3 v;
in vec3 to_dirlight;
in vec3 to_pointlight;
in vec3 pos_ws;
in vec3 spot_dir;

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

	// TODO Rufen Sie shadow_coeff mit passenden Parametern auf, einmal für die
	// Shadowmap für das direktionale Licht und einemal für das Helikopter-Licht.
	// 1.0 heißt: kein Schatten.
	float shadow  = shadow_coeff(shadow_P*shadow_V, shadowmap);
	float hshadow = shadow_coeff(heli_P*heli_V, heli_shadowmap);
	
	// TODO Der Shift ist nur dazu da, dass es im Schatten nicht vollkommen
	// finster ist. Welche Einstellung sieht für Sie gut aus?
	shadow = 0.2+0.8*shadow;
	hshadow = 0.2+0.8*hshadow;

	vec3 N = vec3(0,0,1);
	if (has_normalmap == 1) {
		N = texture(normalmap, tc).rgb * vec3(2.0) - vec3(1.0);
	}

	// directional lighting components
	vec3 diff_dl = diff(to_dirlight, N)    * dirlight_col * dirlight_scale * shadow;
	vec3 spec_dl = spec(to_dirlight, N, 40) * dirlight_col * dirlight_scale * shadow;

	// point lighting components
	float dist = length(to_pointlight);
	vec3 to_light = normalize(to_pointlight);
	float falloff = pow(max(0,dot(spot_dir, to_light)), 15) * hshadow;
	vec3 diff_pl = falloff * diff(to_light, N)     / (dist/100) * heli_col * heli_scale;
	vec3 spec_pl = falloff * spec(to_light, N, 40) / (dist/100) * heli_col * heli_scale;

	// material color
	vec3 k_diff = pow(texture(diffuse, tc).rgb, vec3(2.2));
	vec3 k_spec = texture(specular, tc).rrr;

	out_col = vec4(k_diff * (diff_pl + diff_dl) + k_spec * (spec_pl + spec_dl), 1.0);
	out_col = vec4(pow(out_col.rgb, vec3(1.0/2.2)), 1.0);
}
