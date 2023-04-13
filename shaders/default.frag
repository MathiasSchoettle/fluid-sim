#version 150

out vec4 out_col;

void main() {

	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if (dot(circCoord, circCoord) > 1.0) {
		discard;
	}

	float v = dot(circCoord, circCoord);

	out_col = vec4(0.7, 0.2, 0.4, 1.0);
}
