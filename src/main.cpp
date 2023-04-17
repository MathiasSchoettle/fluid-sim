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
#include <glm/gtx/string_cast.hpp>

using namespace std;
using namespace std::chrono;

bool game_is_running = true;
int amount = 4;
int m = 16;
int width = 1280, height = 720;
float k = 100, k_near = 100, roh_0 = 1;

int last;

int particle_diameter = 10;

void resize(int w, int h) {
	width = w;
	height = h;
}

void init(GLfloat* vertices_position, std::vector<glm::vec3> &positions) {

	positions.clear();

	for (int i = 0; i < amount; ++i) {
		for (int j = 0; j < amount; ++j) {
			for (int k = 0; k < amount; ++k) {
				int index = i * amount * amount + j * amount + k;

				float x = i * (particle_diameter - 4);
				float y = j * (particle_diameter - 2);
				float z = k * (particle_diameter - 3);

				vertices_position[index * 3] = x;
				vertices_position[index * 3 + 2] = y;
				vertices_position[index * 3 + 1] = z;
				positions.push_back(glm::vec3(x, y, z));
			}
		}
	}
}

void update(GLfloat* vertices_position, std::vector<glm::vec3> &positions) {

	std::vector<glm::vec3> old_posis(positions.size());

	for (int i = 0; i < positions.size(); ++i) {
		old_posis[i] = positions[i];
	}

	int ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	float delta_t = ms - last;
	last = ms;
	delta_t /= 1000;
	delta_t = delta_t * delta_t;

	for (int i = 0; i < positions.size(); ++i) {
		auto pos = positions[i];
		float roh_i = 0;
		float roh_near = 0;

		for (int j = 0; j < positions.size(); ++j) {
			if (j == i) continue;

			auto other = positions[j];
			auto r = pos - other;
			auto distance = glm::length(r);

			auto q = distance/particle_diameter;

			if (q < 1) {
				float t = (1 - q);
				roh_i += t * t;
				roh_near += t * t * t;
			}	
		}


		float P_i = k * (roh_i - roh_0);
		float P_near = k_near * roh_near;
		glm::vec3 d_x = glm::vec3(0);

		for (int j = 0; j < positions.size(); ++j) {
			if (j == i) continue;

			auto other = positions[j];
			auto r = pos- other;
			auto distance = glm::length(r);

			auto q = distance/particle_diameter;

			if (q < 1) {
				float t = 1 - q;
				glm::vec3 D = delta_t * (P_i * t + P_near * t * t) * r;
				positions[j] = positions[j] + (D * 0.5f);


				d_x = d_x - (D * 0.5f);
			}
		}

		positions[i] = pos + d_x;
	}

	for (int i = 0; i < positions.size(); ++i) {
		auto pos = positions[i];
		vertices_position[i * 3] = pos.y;
		vertices_position[i * 3 + 2] = pos.x;
		vertices_position[i * 3 + 1] = pos.z;
	}
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
	cam->pos = glm::vec3(-80,202,-27);
	cam->dir = glm::vec3(0.778490, -0.595482, 0.198381);
	cam->up = glm::vec3(0,1,0);
	cam->fix_up_vector = true;
	cam->near = 1;
	cam->far = 12500;
	cam->make_current();
	Camera::default_camera_movement_speed = 0.4;

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLfloat vertices_position[m * m * m * 3] = {0};
	std::vector<glm::vec3> positions;

	init(vertices_position, positions);

	GLuint vbo;
	glGenBuffers(1, &vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_position), vertices_position, GL_STATIC_DRAW);

	shader_ptr shader_points = make_shader("mine", "shaders/default.vert", "shaders/default.frag");

	GLuint shader_id = shader_points.get()->id;
	glEnable(GL_PROGRAM_POINT_SIZE);

	GLint position_attribute = glGetAttribLocation(shader_id, "in_pos");
	glVertexAttribPointer(position_attribute, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(position_attribute);

	Context::set_resize_callback(resize);

	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(positions), std::end(positions), rng);

	last = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	while (Context::running() && game_is_running) {

		update(vertices_position, positions);	
		glNamedBufferSubData(vbo, 0, sizeof(vertices_position), vertices_position);

		Camera::default_input_handler(Context::frame_time());
		Camera::current()->update();
		static uint32_t counter = 0;
		if (counter++ % 100 == 0) Shader::reload();

		{
			static float pi = M_PI;
			float dt = Context::frame_time();
			float v_max = 10;
			static float v = 0;
			
			if (Context::key_pressed(GLFW_KEY_SPACE))
				v = max(0.04f, min(v_max, v*1.05f));
			if (Context::key_pressed(GLFW_KEY_BACKSPACE))
				v = max(0.0f, v*0.92f);
		}

		ImGui::Begin("Simu");

		ImGui::SliderInt("Count", &amount, 0, m);
		ImGui::Separator();
		ImGui::SliderFloat("K", &k, 1, 500);
		ImGui::SliderFloat("K near", &k_near, 1, 500);
		ImGui::SliderFloat("roh", &roh_0, 1, 500);
		ImGui::SliderInt("size", &particle_diameter, 5, 20);

		if (ImGui::Button("Reset")) {
			init(vertices_position, positions);
		}

		ImGui::End();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shader_ptr shader = shader_points;
		shader->bind();
		shader->uniform("view", cam->view);
		shader->uniform("proj", cam->proj);
		shader->uniform("screen_size", glm::vec2(width, height));
		shader->uniform("sprite_size", (float) particle_diameter);

		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, amount * amount * amount);

		Context::swap_buffers();
		
	}

	return 0;
}
