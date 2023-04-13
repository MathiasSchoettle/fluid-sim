#include <bigduckgl/bigduckgl.h>
#include <imgui/imgui.h>

#include <iostream>
#include <vector>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

using namespace std;

bool game_is_running = true;

int main(int argc, char** argv) {

	ContextParameters params;
	params.gl_major = 4;
	params.gl_minor = 4;
	params.title = "HSP";
	params.font_ttf_filename = "render-data/fonts/DroidSansMono.ttf";
	params.font_size_pixels = 15;
	Context::init(params);

	auto cam = make_camera("cam");
	cam->pos = glm::vec3(-270,131,-82);
	cam->dir = glm::vec3(1,0,0);
	cam->up = glm::vec3(0,1,0);
	cam->fix_up_vector = true;
	cam->near = 1;
	cam->far = 12500;
	cam->make_current();
	Camera::default_camera_movement_speed = 0.4;

	auto shadow_cam = make_camera("shadowcam");
	shadow_cam->perspective = false;
	shadow_cam->fix_up_vector = false;
	shadow_cam->up = glm::vec3(0,0,1);
	shadow_cam->near = 1;
	shadow_cam->far = 10000;
	shadow_cam->left = -1000;
	shadow_cam->right = 1000;
	shadow_cam->top = 1000;
	shadow_cam->bottom = -1000;

	std::vector<drawelement_ptr> sponza = MeshLoader::load("render-data/models/plane.obj");
	shader_ptr shader_normalmapped = make_shader("a6", "shaders/normalmapping.vert", "shaders/normalmapping.frag");
	shader_ptr shader_shadows      = make_shader("a7", "shaders/shadows.vert", "shaders/shadows.frag");

	shader_ptr light_rep_shader = make_shader("light-rep", "shaders/light_rep.vert", "shaders/light_rep.frag");
	std::vector<drawelement_ptr> light_rep = MeshLoader::load("render-data/models/sphere.obj", false, [&](const material_ptr &) { return light_rep_shader; });
	
	shader_ptr sky_shader = make_shader("sky", "shaders/sky.vert", "shaders/sky.frag");
	material_ptr sky_mat = make_material("sky");
	sky_mat->k_diff = glm::vec4(.3,.3,1,1);
	shared_ptr<Texture2D> sky_tex = make_texture("sky", "render-data/cgskies-0319-free.jpg", false);
	sky_mat->add_texture("tex", sky_tex);
	drawelement_ptr sky = make_drawelement("sky", sky_shader, sky_mat, light_rep[0]->meshes);

	int sm_w = 1024;
	auto shadowmap = make_framebuffer("shadowmap", sm_w, sm_w);
	shadowmap->attach_depthbuffer(make_texture("shadowmap", sm_w, sm_w, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT));
	shadowmap->check();

	glBindTexture(GL_TEXTURE_2D, shadowmap->depth_texture->id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	TimerQuery input_timer("input");
	TimerQuery update_timer("update");
	TimerQueryGL render_timer("render");
	TimerQueryGL render_sm_timer("render shadowmap");

	while (Context::running() && game_is_running) {

		input_timer.start();
		Camera::default_input_handler(Context::frame_time());
		Camera::current()->update();
		static uint32_t counter = 0;
		if (counter++ % 100 == 0) Shader::reload();
		input_timer.end();

		static glm::vec3 dirlight_dir = glm::vec3(0.25,-.93,-.25);
		static glm::vec3 dirlight_col = glm::vec3(1.0,0.97,0.97);
		static float     dirlight_scale = 1.2f;

		update_timer.start();
		{
			static float pi = M_PI;
			float dt = Context::frame_time();
			float v_max = 10;
			static float v = 0;
			
			if (Context::key_pressed(GLFW_KEY_SPACE))
				v = max(0.04f, min(v_max, v*1.05f));
			if (Context::key_pressed(GLFW_KEY_BACKSPACE))
				v = max(0.0f, v*0.92f);

			shadow_cam->pos = -dirlight_dir*3000.0f;
			shadow_cam->dir = dirlight_dir;
			shadow_cam->update();
		}
		update_timer.end();

		render_sm_timer.start();
		auto render = [&](std::shared_ptr<Camera> cam) {
			cam->make_current();
			glClear(GL_DEPTH_BUFFER_BIT);
			for (auto &de : sponza) {
				de->shader = shader_normalmapped;
				de->bind();
				de->shader->uniform("cam_pos", Camera::current()->pos);
				de->shader->uniform("dirlight_dir", dirlight_dir);
				de->shader->uniform("dirlight_col", dirlight_col);
				de->shader->uniform("dirlight_scale", dirlight_scale);
				de->shader->uniform("has_alphamap", de->material->has_texture("alphamap") ? 1 : 0);
				de->shader->uniform("has_normalmap", de->material->has_texture("normalmap") ? 1 : 0);
				de->draw(glm::mat4(1));
				de->unbind();
			}
		};

		glCullFace(GL_FRONT);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1,1);
		shadowmap->bind();
		render(shadow_cam);
		shadowmap->unbind();
		glDisable(GL_POLYGON_OFFSET_FILL);
		glCullFace(GL_BACK);
		cam->make_current();

		render_sm_timer.end();

		render_timer.start();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		for (auto &de : sponza) {
			shader_ptr shader = shader_shadows;
			de->shader = shader_shadows;
			de->bind();
			shader->uniform("cam_pos", Camera::current()->pos);
			shader->uniform("dirlight_dir", dirlight_dir);
			shader->uniform("dirlight_col", glm::pow(dirlight_col, glm::vec3(2.2f)));
			shader->uniform("dirlight_scale", dirlight_scale);
			shader->uniform("has_alphamap", de->material->has_texture("alphamap") ? 1 : 0);
			shader->uniform("has_normalmap", de->material->has_texture("normalmap") ? 1 : 0);
			shader->uniform("shadowmap", shadowmap->depth_texture, 5);
			shader->uniform("shadow_V", shadow_cam->view);
			shader->uniform("shadow_P", shadow_cam->proj);
			de->draw(glm::mat4(1));
			de->unbind();
		}

		glDisable(GL_CULL_FACE);
		float n = Camera::current()->near;
		float f = Camera::current()->far;
		Camera::current()->near = 10;
		Camera::current()->far = 20000;
		Camera::current()->update();
		sky->bind();
		sky->draw(glm::scale(glm::mat4(1), glm::vec3(15000,15000,15000)));
		sky->unbind();
		Camera::current()->near = n;
		Camera::current()->far = f;
		glEnable(GL_CULL_FACE);

		render_timer.end();

		Context::swap_buffers();
	}

	return 0;
}
