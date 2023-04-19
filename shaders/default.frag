#version 150

in float size;

uniform float n;
uniform float f;

out vec3 out_col;

void main() {
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if (dot(circCoord, circCoord) > 1) {
		discard;
	}

	float v = dot(circCoord, circCoord);

	float x = (1 - gl_PointCoord.x);
	float y = (1 - gl_PointCoord.y);

	float p = length(circCoord) * size;
	float height = sqrt(p * p + size * size);
	float z = gl_FragCoord.z;
	
	out_col = vec3(1, 1, 1) * x * y;
	out_col = vec3(0.2, 0.4, 0.8) * (gl_FragCoord.w * 10 - height / 3000) * 25;
}
