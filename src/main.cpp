#include <bigduckgl/bigduckgl.h>
#include <imgui/imgui.h>

#include <iostream>
#include <vector>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <algorithm>
#include <random>
#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL

#include "simulation.h"
#include "quad.h"

using namespace std;
using namespace std::chrono;

int width = 1920, height = 1080;
float n = 0.1, f = 400;
float EPSILON_MULT = 0.5;
int grid_size = 200;

GLuint g_buffer;
GLuint g_norm, g_col, g_depth;

void setup_g_buffer() {

	glGenFramebuffers(1, &g_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, g_buffer);

	glGenTextures(1, &g_depth);
	glBindTexture(GL_TEXTURE_2D, g_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_depth, 0);

	glGenTextures(1, &g_norm);
	glBindTexture(GL_TEXTURE_2D, g_norm);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_norm, 0);
  
	glGenTextures(1, &g_col);
	glBindTexture(GL_TEXTURE_2D, g_col);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_col, 0);

	unsigned int attachments[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, attachments);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(int argc, char** argv) {

	ContextParameters params;
	params.gl_major = 4;
	params.gl_minor = 4;
	params.title = "HSP";
	params.font_size_pixels = 15;
	params.resizable = false;
	params.width = width;
	params.height = height;
	Context::init(params);

	auto cam = make_camera("cam");
	cam->pos = glm::vec3(13.8, 69.3, -37.0);
	cam->dir = glm::vec3(0.317, -0.483, 0.816);
	cam->up = glm::vec3(0,1,0);
	cam->fov_degree = 90;
	cam->fix_up_vector = true;
	cam->near = n;
	cam->far = f;
	cam->make_current();
	Camera::default_camera_movement_speed = 0.02;

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	shader_ptr shader_depth = make_shader("depth", "shaders/default.vert", "shaders/depth.frag");
	shader_ptr shader_attribs = make_shader("attribs", "shaders/default.vert", "shaders/default.frag");
	shader_ptr shader_quad = make_shader("quad", "shaders/quad.vert", "shaders/quad.frag");

	simulation sim(grid_size);
	quad quad;

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

	setup_g_buffer();

	glfwSwapInterval(0);

	static int count = 0;

	while (Context::running()) {
		Camera::default_input_handler(Context::frame_time());
		Camera::current()->update();

		ImGui::Begin("Simulation");
		ImGui::Text("Particle count: %d", sim.particle_count);
		ImGui::Text("Fps: %f", ImGui::GetIO().Framerate);

		if (ImGui::Button(sim.pause ? "Resume" : "Pause")) {
			sim.pause = !sim.pause;
		}

		ImGui::Text("Pos: (%f, %f %f)", cam->pos.x, cam->pos.y, cam->pos.z);
		ImGui::Text("Dir: (%f, %f %f)", cam->dir.x, cam->dir.y, cam->dir.z);

		ImGui::Separator();
		float sum = 0;
		for (auto [name, time] : sim.times) {
			float current = time[time.size() - 1];
			sum += current;
			ImGui::TextColored((current > 4 ? ImVec4(1, 0, 0, 1) : ImVec4(1, 1, 1, 1)), "%s: %f", name.c_str(), current);
		}

		ImGui::Separator();
		ImGui::Text("SUM: %f", sum);
		ImGui::Separator();

		ImGui::Text("Density relaxation");
		ImGui::SliderFloat("k", &sim.k, 0.001, 2);
		ImGui::SliderFloat("k near", &sim.k_near, 0.1, 40);
		ImGui::SliderFloat("roh", &sim.roh_0, 1, 45);
		
		ImGui::Separator();
		
		ImGui::Text("Viscosity");
		ImGui::SliderFloat("sigma", &sim.sigma, 0.0, 20.0);
		ImGui::SliderFloat("beta", &sim.beta, 0.0, 20.0);
		ImGui::SliderFloat3("gravity", &sim.gravity.x, -9.81, 9.81);

		ImGui::Separator();
		ImGui::SliderFloat("render size", &sim.particle_render_factor, 0.5f, 3.0f);

		if (ImGui::Button("Gravity reset")) {
			sim.gravity = glm::vec3(0);
		}

		if (ImGui::Button("Reset")) {
			sim.set_data();
		}
		
		ImGui::End();

		if (count++ > 100) {
			count = 0;
			Shader::reload();
		}

		sim.step();

		float sprite_size = sim.particle_diameter * sim.particle_render_factor;

		shader_depth->bind();
		shader_depth->uniform("view", cam->view);
		shader_depth->uniform("proj", cam->proj);
		shader_depth->uniform("screen_size", Context::resolution());
		shader_depth->uniform("sprite_size", sprite_size);
		shader_depth->uniform("n", cam->near);
		shader_depth->uniform("f", cam->far);
		shader_depth->uniform("eps", sim.particle_diameter * EPSILON_MULT);

		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer);

		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

		sim.draw();

		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

		shader_attribs->bind();
		shader_attribs->uniform("view", cam->view);
		shader_attribs->uniform("proj", cam->proj);
		shader_attribs->uniform("screen_size", Context::resolution());
		shader_attribs->uniform("sprite_size", sprite_size);
		shader_attribs->uniform("n", cam->near);
		shader_attribs->uniform("f", cam->far);
		shader_attribs->uniform("eps", sim.particle_diameter * EPSILON_MULT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_depth);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		
		sim.draw();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		shader_quad->bind();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_norm);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, g_col);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, g_depth);

		glDisable(GL_BLEND);

		auto light = glm::normalize(glm::vec3(0.5f, -1.0f, 0.4f));
		auto ld_view = cam->view * glm::vec4(light, 0.0);

		shader_quad->uniform("view", cam->view);
		shader_quad->uniform("g_norm", 0);
		shader_quad->uniform("g_col", 1);
		shader_quad->uniform("g_depth", 2);
		shader_quad->uniform("light_dir", glm::vec3(ld_view));
		shader_quad->uniform("n", cam->near);
		shader_quad->uniform("f", cam->far);

		quad.draw();

		Context::swap_buffers();
	}

	return 0;
}
