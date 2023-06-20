#include "box.h"

#include <iostream>

using namespace std;

box::box(float size) : size(size) {

	float vertices[] = {
		// Front face
		0, 0, size,   // Vertex 0
		size, 0, size,    // Vertex 1
		size, size, size,     // Vertex 2
		0, size, size,    // Vertex 3
		
		// Back face
		0, 0, 0,  // Vertex 4
		size, 0, 0,   // Vertex 5
		size, size, 0,    // Vertex 6
		0, size, 0    // Vertex 7
	};

	// Indices of the edges (12 edges)
	GLuint indices[] = {
		0, 1,   // Front bottom edge
		1, 2,   // Front right edge
		2, 3,   // Front top edge
		3, 0,   // Front left edge
		4, 5,   // Back bottom edge
		5, 6,   // Back right edge
		6, 7,   // Back top edge
		7, 4,   // Back left edge
		0, 4,   // Left vertical edge
		1, 5,   // Right vertical edge
		2, 6,   // Top vertical edge
		3, 7    // Bottom vertical edge
	};

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Create Vertex Buffer Object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Create Element Buffer Object
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

}

box::~box() {
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ibo);
	glDeleteVertexArrays(1, &vao);
	//delete[] data;
}

void box::draw() {
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glLineWidth(2.0f);

	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	// Draw the cube using the indices
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

	glDisableVertexAttribArray(0);
	glBindVertexArray(0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}