#version 150

out vec4 out_col;

void main() {

	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if (dot(circCoord, circCoord) > 1.0) {
		discard;
	}

	float v = dot(circCoord, circCoord);

	float x = (1 - gl_PointCoord.x);
	float y = (1 - gl_PointCoord.y);



	out_col = vec4(0.75);
	out_col = vec4(1, 1, 1, 1) * x * y;
}
