#pragma once

#include <bigduckgl/bigduckgl.h>

struct quad {
	quad();
	~quad();
	void draw();
private:
	GLuint vao, vbo;
	float *data;
};