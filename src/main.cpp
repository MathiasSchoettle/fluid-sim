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
int m = 32;
int width = 1280, height = 720;
float k = 1, k_near = 50, roh_0 = 8.0f;
glm::vec3 gravity(0, 0, 0);

float alpha = 0.3, gama = 0.1;
float sigma = 0.5, betta = 0.25;

float L_frac = 0.1;
float k_spring = 0.3;

int last;

int particle_diameter = 10;

void resize(int w, int h) {
	width = w;
	height = h;
}

void init(GLfloat* vertices_position, vector<glm::vec3> &positions, vector<glm::vec3> &velocities, map<tuple<int, int>, float> springs) {

	positions.clear();
	velocities.clear();
	springs.clear();
	
	int amount = 10;

	for (int i = 0; i < amount; ++i) {
		for (int j = 0; j < amount; ++j) {
			for (int k = 0; k < amount; ++k) {
				int index = i * amount * amount + j * amount + k;

				float x = i * (particle_diameter + 5);
				float y = j * (particle_diameter + 5);
				float z = k * (particle_diameter + 5);

				vertices_position[index * 3] = x;
				vertices_position[index * 3 + 2] = y;
				vertices_position[index * 3 + 1] = z;

				positions.push_back(glm::vec3(x, y, z));
				velocities.push_back(glm::vec3(0.0));
			}
		}
	}
}

void update(GLfloat* vertices_position, vector<glm::vec3> &positions, vector<glm::vec3> &velocities, map<tuple<int, int>, float> springs) {

	std::vector<glm::vec3> old_posis(positions.size());

	int ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	float delta_t = ms - last;
	last = ms;
	delta_t /= 7000.0f;

	delta_t = 0.1;

	float h = particle_diameter * 1.5;

	// gravity
	for (int i = 0; i < positions.size(); ++i) {
		velocities[i] = velocities[i] + delta_t * gravity;
	}

	// viscosity
	for (int i = 0; i < positions.size() - 1; ++i) {
		for (int j = i + 1; j < positions.size(); ++j) {

			float q = glm::length(positions[j] - positions[i]) / h;

			if (q < 1) {
				auto u = glm::dot((velocities[i] - velocities[j]), positions[j] - positions[i]);

				if (u > 0) {
					auto I = delta_t * (1 - q) * (sigma * u + betta * u * u) * glm::normalize(positions[j] - positions[i]);
					velocities[i] -= I * 0.5f;
					velocities[j] += I * 0.5f; 
				}
			}
		}
	}

	// velocity prediction
	for (int i = 0; i < positions.size(); ++i) {
		old_posis[i] = positions[i];
		positions[i] += delta_t * velocities[i];
	}

	// //adjust springs
	// for (int j = 0; j < positions.size(); ++j) {
	// 	for (int i = 0; i < j; ++i) {

	// 		auto distance = glm::length(positions[j] - positions[i]);
	// 		float q = distance / h;

	// 		if (q < 1) {

	// 			auto spring = tuple(i, j);
	// 			if (springs.find(spring) == springs.end()) {
	// 				springs[spring] = h;
	// 			}

	// 			auto d = gama * springs[spring];

	// 			if (glm::length(positions[j] - positions[i]) > L_frac + d) {
	// 				springs[spring] += delta_t * alpha * (distance - L_frac - d);
	// 			} else if (glm::length(positions[j] - positions[i]) < L_frac + d) {
	// 				springs[spring] -= delta_t * alpha * (L_frac - d - distance);
	// 			}
	// 		}
	// 	}
	// }

	// spring deletion
	// for (auto it = springs.cbegin(); it != springs.cend();) {
	// 	if (it->second > h) {
	// 		it = springs.erase(it);
	// 	}
	// }

	// // spring adjustments
	// for (auto [key, val] : springs) {
	// 	auto r = positions[get<1>(key)] - positions[get<0>(key)];
	// 	auto distance = glm::length(r);
	// 	auto D = delta_t * delta_t * k_spring * (1 - val / h) * (val - distance) * glm::normalize(r);
	// 	positions[get<0>(key)] -= D * 0.5f;
	// 	positions[get<1>(key)] += D * 0.5f;
	// }


	// double density
	for (int i = 0; i < positions.size(); ++i) {
		auto pos = positions[i];
		float roh = 0;
		float roh_near = 0;

		for (int j = 0; j < positions.size(); ++j) {
			if (j == i) continue;

			float q = glm::length(positions[j] - pos) / h;

			if (q < 1) {
				roh += (1 - q) * (1 - q);
				roh_near += (1 - q) * (1 - q) * (1 - q);
			}	
		}

		float P = k * (roh - roh_0);
		float P_near = k_near * roh_near;
		glm::vec3 d_x = glm::vec3(0, 0, 0);

		for (int j = 0; j < positions.size(); ++j) {
			if (j == i) continue;

			float q = glm::length(positions[j] - pos) / h;

			if (q < 1) {
				glm::vec3 D = delta_t * delta_t * (P * (1 - q) + P_near * (1 - q) * (1 - q)) * glm::normalize(positions[j] - pos);
				positions[j] += D * 0.5f;
				d_x = d_x - D * 0.5f;
			}
		}

		positions[i] = pos + d_x;
	}

	// compute next velocity and set position
	for (int i = 0; i < positions.size(); ++i) {
		auto pos = positions[i];
		vertices_position[i * 3] = pos.y;
		vertices_position[i * 3 + 2] = pos.x;
		vertices_position[i * 3 + 1] = max(0.0f, pos.z);
		velocities[i] = (pos - old_posis[i]) / delta_t;
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
	std::vector<glm::vec3> velocities;
	map<tuple<int, int>, float> springs;

	init(vertices_position, positions, velocities, springs);

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

		update(vertices_position, positions, velocities, springs);


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

		ImGui::Begin("Simulation");

		ImGui::SliderInt("size", &particle_diameter, 5, 20);

		ImGui::Separator();

		ImGui::Text("Density relaxation");
		ImGui::SliderFloat("k", &k, 0.001, 1);
		ImGui::SliderFloat("k near", &k_near, 1, 100);
		ImGui::SliderFloat("roh", &roh_0, 1, 10);

		ImGui::Separator();

		ImGui::Text("Viscosity");
		ImGui::SliderFloat("omega", &sigma, 0.0, 1.0);
		ImGui::SliderFloat("beta", &betta, 0.0, 0.5);

		// ImGui::Text("Springs");
		// ImGui::SliderFloat("alpha", &alpha, 0.001, 1.0);
		// ImGui::SliderFloat("gamma", &gama, 0.001, 1);
		// ImGui::SliderFloat("L frac", &L_frac, 0.0, 1.0);
		// ImGui::SliderFloat("k spring", &k_spring, 0.0, 1);
		// ImGui::Text("Spring count: %ld", springs.size());

		if (ImGui::Button("Reset")) {
			init(vertices_position, positions, velocities, springs);
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
		glDrawArrays(GL_POINTS, 0, 1000);

		Context::swap_buffers();
		
	}

	return 0;
}
