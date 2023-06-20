#pragma once

#include <bigduckgl/bigduckgl.h>

struct box {
	box(float size);
	~box();
	void draw();
private:
	GLuint vao, vbo, ibo;
	float *data;
	float size;
};