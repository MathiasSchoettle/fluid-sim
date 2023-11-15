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
float n = 0.1, f = 4000;
float EPSILON_MULT = 0.5;
int grid_size = 200;

GLuint g_buffer;
GLuint g_norm, g_col, g_depth, g_depth_real;

GLuint g_buffer_particle;
GLuint g_depth_particle, g_col_particle;

GLuint g_buffer_particle_blurred[2];
GLuint g_col_particle_blurred[2];

void imgui(simulation &sim, const std::shared_ptr<Camera> &cam);
void setup_g_buffer();
void setup_g_buffer_particle();
void setup_g_buffer_particle_blurred();

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
	cam->pos = glm::vec3(168, 157, -179);
	cam->dir = glm::vec3(-0.639, -0.3432, 0.6883);
	cam->up = glm::vec3(0,1,0);
	cam->fov_degree = 90;
	cam->fix_up_vector = true;
	cam->near = n;
	cam->far = f;
	cam->make_current();
	Camera::default_camera_movement_speed = 0.5;
	
	std::vector<drawelement_ptr> sponza = MeshLoader::load("render-data/models/box/box.obj");
	shader_ptr shader_color  = make_shader("test", "shaders/test.vert", "shaders/test.frag");
	shader_ptr shader_particle = make_shader("particle", "shaders/particle.vert", "shaders/particle.frag");
	shader_ptr shader_quad_blur = make_shader("quadblur", "shaders/quad.vert", "shaders/quad_blur.frag");
	shader_ptr shader_quad = make_shader("quad", "shaders/quad.vert", "shaders/quad.frag");

	simulation sim(grid_size);
	quad quad;

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);

	setup_g_buffer();
	setup_g_buffer_particle();
	setup_g_buffer_particle_blurred();

	static int count = 0;
	
	while (Context::running()) {
		Camera::default_input_handler(Context::frame_time());
		Camera::current()->update();

		imgui(sim, cam);

		if (count++ > 100) {
			count = 0;
			Shader::reload();
		}

		sim.step();

		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer);

		unsigned int attachments1[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, attachments1);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		unsigned int attachments2[] = { GL_COLOR_ATTACHMENT2 };
		glDrawBuffers(1, attachments2);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		unsigned int attachments3[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
		glDrawBuffers(3, attachments3);

		for (auto &de : sponza) {
			de->shader = shader_color;
			de->bind();
			shader_color->uniform("cam_pos", Camera::current()->pos);
			shader_color->uniform("n", Camera::current()->near);
			shader_color->uniform("f", Camera::current()->far);
			de->draw(glm::mat4(1));
			de->unbind();
		}

		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_particle);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_depth_real);

		const float sprite_size = sim.particle_diameter * sim.particle_render_factor;

		shader_particle->bind();
		shader_particle->uniform("g_depth", 0);
		shader_particle->uniform("view", cam->view);
		shader_particle->uniform("proj", cam->proj);
		shader_particle->uniform("screen_size", glm::ivec2(width, height));
		shader_particle->uniform("sprite_size", sprite_size);
		shader_particle->uniform("n", cam->near);
		shader_particle->uniform("f", cam->far);
		shader_particle->uniform("invProj", glm::inverse(cam->proj));
		shader_particle->uniform("invView", glm::inverse(cam->view));
		
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glBlendFunc(GL_SRC_COLOR, GL_ONE);
		sim.draw();
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);

		for (int i = 0; i < 2; ++i) {
			glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_particle_blurred[i]);
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		glActiveTexture(GL_TEXTURE0);

		shader_quad_blur->bind();
		shader_quad_blur->uniform("g_col_particle", 0);
		shader_quad_blur->uniform("n", cam->near);
		shader_quad_blur->uniform("f", cam->far);
		shader_quad_blur->uniform("res", glm::vec2(width, height));

		bool horizontal = true, first_iteration = true;
		int amount = 20;

		for (int i = 0; i < amount; ++i) {
			glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_particle_blurred[horizontal]);
			shader_quad_blur->uniform("horizontal", horizontal);
			glBindTexture(GL_TEXTURE_2D, first_iteration ? g_col_particle : g_col_particle_blurred[!horizontal]);
			quad.draw();
			horizontal = !horizontal;
			if (first_iteration) first_iteration = false;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, g_norm);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, g_col);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, g_col_particle_blurred[!horizontal]);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, g_depth_real);

		auto light = glm::normalize(glm::vec3(0.5f, -1.0f, 0.4f));
		auto ld_view = cam->view * glm::vec4(light, 0.0);

		shader_quad->bind();
		shader_quad->uniform("g_norm", 0);
		shader_quad->uniform("g_col", 1);
		shader_quad->uniform("g_col_particle", 2);
		shader_quad->uniform("g_depth", 3);
		shader_quad->uniform("n", cam->near);
		shader_quad->uniform("f", cam->far);
		shader_quad->uniform("res", glm::vec2(width, height));
		shader_quad->uniform("light_dir", glm::vec3(ld_view));

		quad.draw();

		Context::swap_buffers();
	}

	return 0;
}

void imgui(simulation &sim, const std::shared_ptr<Camera> &cam) {

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
	for (const auto [name, time] : sim.times) {
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
	ImGui::SliderFloat("render size", &sim.particle_render_factor, 0.5f, 8.0f);

	if (ImGui::Button("Gravity reset")) {
		sim.gravity = glm::vec3(0);
	}

	if (ImGui::Button("Reset")) {
		sim.reset();
	}
	
	ImGui::End();
}

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

	glGenTextures(1, &g_depth_real);
	glBindTexture(GL_TEXTURE_2D, g_depth_real);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_depth_real, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void setup_g_buffer_particle() {

	glGenFramebuffers(1, &g_buffer_particle);
	glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_particle);

	glGenTextures(1, &g_depth_particle);
	glBindTexture(GL_TEXTURE_2D, g_depth_particle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_depth_particle, 0);

	glGenTextures(1, &g_col_particle);
	glBindTexture(GL_TEXTURE_2D, g_col_particle);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_col_particle, 0);

	unsigned int attachments[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void setup_g_buffer_particle_blurred() {

	glGenFramebuffers(2, g_buffer_particle_blurred);
	glGenTextures(2, g_col_particle_blurred);

	for (int i = 0; i < 2; i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_particle_blurred[i]);
		glBindTexture(GL_TEXTURE_2D, g_col_particle_blurred[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_col_particle_blurred[i], 0);

		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}