#include <bigduckgl/bigduckgl.h>
#include <imgui/imgui.h>

#include <iostream>
#include <vector>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <random>
#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL

#include "simulation.h"
#include "quad.h"

using namespace std;
using namespace std::chrono;

int width = 1280, height = 720;
float n = 0.1, f = 50;

GLuint g_buffer;
GLuint g_pos, g_norm, g_col;

void setup_g_buffer() {

	glGenFramebuffers(1, &g_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, g_buffer);

	glGenTextures(1, &g_pos);
	glBindTexture(GL_TEXTURE_2D, g_pos);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_pos, 0);

	glGenTextures(1, &g_norm);
	glBindTexture(GL_TEXTURE_2D, g_norm);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_norm, 0);
  
	glGenTextures(1, &g_col);
	glBindTexture(GL_TEXTURE_2D, g_col);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_col, 0);

	unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);

	unsigned int rbo_depth;
	glGenRenderbuffers(1, &rbo_depth);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo_depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo_depth);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(int argc, char** argv) {

	ContextParameters params;
	params.gl_major = 4;
	params.gl_minor = 4;
	params.title = "HSP";
	params.font_ttf_filename = "render-data/fonts/DroidSansMono.ttf";
	params.font_size_pixels = 15;
	Context::init(params);

	auto cam = make_camera("cam");
	cam->pos = glm::vec3(-8.119548, 16.163836, -0.112831);
	cam->dir = glm::vec3(0.772751, -0.531802, 0.346472);
	cam->up = glm::vec3(0,1,0);
	cam->fov_degree = 90;
	cam->fix_up_vector = true;
	cam->near = n;
	cam->far = f;
	cam->make_current();
	Camera::default_camera_movement_speed = 0.02;

	shader_ptr shader_points = make_shader("mine", "shaders/default.vert", "shaders/default.frag");
	shader_ptr shader_quad = make_shader("quad", "shaders/quad.vert", "shaders/quad.frag");
	simulation sim;
	quad quad;

	glEnable(GL_PROGRAM_POINT_SIZE);

	setup_g_buffer();

	while (Context::running()) {
		Camera::default_input_handler(Context::frame_time());
		Camera::current()->update();

		ImGui::Begin("Simulation");
		ImGui::SliderFloat("size", &sim.particle_diameter, 0.2, 5);
		ImGui::Separator();
		ImGui::Text("Density relaxation");
		ImGui::SliderFloat("k", &sim.k, 0.001, 10);
		ImGui::SliderFloat("k near", &sim.k_near, 0.1, 100);
		ImGui::SliderFloat("roh", &sim.roh_0, 1, 15);
		ImGui::Separator();
		ImGui::Text("Viscosity");
		ImGui::SliderFloat("sigma", &sim.sigma, 0.0, 1.0);
		ImGui::SliderFloat("beta", &sim.beta, 0.0, 0.5);
		if (ImGui::Button("Reset")) {
			sim.set_data();
			Shader::reload();
		}
		ImGui::End();
		
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		sim.step();

		shader_points->bind();
		shader_points->uniform("view", cam->view);
		shader_points->uniform("proj", cam->proj);
		shader_points->uniform("screen_size", Context::resolution());
		shader_points->uniform("sprite_size", sim.particle_diameter);
		shader_points->uniform("n", cam->near);
		shader_points->uniform("f", cam->far);
		
		sim.draw(g_buffer);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shader_quad->bind();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_pos);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, g_norm);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, g_col);

		shader_quad->uniform("g_pos", 0);
		shader_quad->uniform("g_norm", 1);
		shader_quad->uniform("g_col", 2);
		shader_quad->uniform("n", cam->near);
		shader_quad->uniform("f", cam->far);

		quad.draw();

		glBindFramebuffer(GL_READ_FRAMEBUFFER, g_buffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		Context::swap_buffers();
	}

	return 0;
}
