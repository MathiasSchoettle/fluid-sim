#version 150

in vec3 pos_wc;
uniform sampler2D tex;
out vec4 out_col;

#define PI 3.1415926535897932384626433832795

void main() {
	vec3 dir = normalize(-pos_wc);

	vec2 uv = vec2(atan(dir.z, dir.x) / (2*PI) + 0.5,
				   0.5 - asin(dir.y)/PI);
	
	out_col = vec4(texture(tex, uv).rgb, 1);
	
	// Falls Sie die berechnete Textur-Koordinate ausgeben wollen:
	// out_col = vec4(uv,0,1);
}
