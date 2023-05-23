#include "box.h"

#include <iostream>

using namespace std;

box::box() {

	data = new float[72] {
		// Front face
		0.0f, 100.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		100.0f, 100.0f, 0.0f,
		100.0f, 0.0f, 0.0f,

		// Back face
		0.0f, 100.0f, 100.0f,
		0.0f, 0.0f, 100.0f,
		100.0f, 100.0f, 100.0f,
		100.0f, 0.0f, 100.0f,

		// Left face
		0.0f, 100.0f, 100.0f,
		0.0f, 0.0f, 100.0f,
		0.0f, 100.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		// Right face
		100.0f, 100.0f, 100.0f,
		100.0f, 0.0f, 100.0f,
		100.0f, 100.0f, 0.0f,
		100.0f, 0.0f, 0.0f,

		// Top face
		0.0f, 100.0f, 100.0f,
		0.0f, 100.0f, 0.0f,
		100.0f, 100.0f, 100.0f,
		100.0f, 100.0f, 0.0f,

		// Bottom face
		0.0f, 0.0f, 100.0f,
		0.0f, 0.0f, 0.0f,
		100.0f, 0.0f, 100.0f,
		100.0f, 0.0f, 0.0f
	};

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * 3 * 6, data, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glBindVertexArray(0);
}

box::~box() {
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	delete[] data;
}

void box::draw() {
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(2.0f);

	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 24);

	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}