#pragma once

#include <bigduckgl/bigduckgl.h>

struct box {
	box();
	~box();
	void draw();
private:
	GLuint vao, vbo, ibo;
	float *data;
};