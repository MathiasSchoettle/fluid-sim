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
//     parse_cmdline(argc, argv);

    // init context and set parameters
    ContextParameters params;
	params.gl_major = 3;
	params.gl_minor = 3;
    params.title = "CG'21";
    params.font_ttf_filename = "render-data/fonts/DroidSansMono.ttf";
    params.font_size_pixels = 15;
    Context::init(params);

    auto cam = make_camera("cam");
	cam->pos = glm::vec3(-516,584,-138);
	cam->dir = glm::vec3(1,0,0);
	cam->up = glm::vec3(0,1,0);
	cam->fix_up_vector = true;
    cam->near = 1;
    cam->far = 12500;
    cam->make_current();
	Camera::default_camera_movement_speed = 0.1;

	auto heli_cam = make_camera("heli");
	heli_cam->pos = glm::vec3(0,500,0);
	heli_cam->dir = glm::vec3(1,0,0);
	heli_cam->up = glm::vec3(0,1,0);
	heli_cam->near = 1;
	heli_cam->far = 5000;

	glm::mat4 heli_mat = glm::inverse(glm::lookAt(glm::vec3(0,500,0), glm::vec3(0,500,1), glm::vec3(0,1,0)));
	heli_cam->view = glm::inverse(heli_mat);
	glm::mat4 rotor(1);


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
	std::vector<drawelement_ptr> sponza = MeshLoader::load("render-data/models/sponza/sponza.fixed.obj");
	shader_ptr shader_plain_color  = make_shader("a1", "shaders/default.vert", "shaders/default.frag");
	shader_ptr shader_material     = make_shader("a2", "shaders/material.vert", "shaders/material.frag");
	shader_ptr shader_lighting     = make_shader("a3", "shaders/shading.vert", "shaders/shading.frag");
	shader_ptr shader_textured     = make_shader("a4", "shaders/tex.vert", "shaders/tex.frag");
	shader_ptr shader_masked       = make_shader("a5", "shaders/mask.vert", "shaders/mask.frag");
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

	std::vector<drawelement_ptr> heli = MeshLoader::load("render-data/models/heli2.obj", false, [&](const material_ptr &) { return shader_lighting; });

	int sm_w = 1024;
	auto shadowmap = make_framebuffer("shadowmap", sm_w, sm_w);
	shadowmap->attach_depthbuffer(make_texture("shadowmap", sm_w, sm_w, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT));
	shadowmap->check();

	auto heli_shadowmap = make_framebuffer("heli_shadowmap", sm_w, sm_w);
	heli_shadowmap->attach_depthbuffer(make_texture("heli_shadowmap", sm_w, sm_w, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT));
	heli_shadowmap->attach_colorbuffer(make_texture("heli0", sm_w, sm_w, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE));
	heli_shadowmap->check();

	glBindTexture(GL_TEXTURE_2D, shadowmap->depth_texture->id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindTexture(GL_TEXTURE_2D, heli_shadowmap->depth_texture->id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	TimerQuery input_timer("input");
	TimerQuery update_timer("update");
	TimerQueryGL render_timer("render");
	TimerQueryGL render_sm_timer("render shadowmap");

	glm::vec4 cols[100];
	for (int i = 0; i < 100; ++i)
		cols[i] = glm::vec4(glm::linearRand(0.0f,1.0f), glm::linearRand(0.0f,1.0f), glm::linearRand(0.0f,1.0f), 1.0f);

    while (Context::running() && game_is_running) {
        // input handling
        input_timer.start();
		Camera::default_input_handler(Context::frame_time());
        Camera::current()->update();
        static uint32_t counter = 0;
        if (counter++ % 100 == 0) Shader::reload();
        input_timer.end();

		// UI
		ImGui::Begin("Lighting");
		static const char *items[] = { "Plain Color",
		                               "Material Color",
									   "Phong Lighting",
									   "Textured",
									   "Masked",
									   "Normalmapped",
									   "With Sky",
									   "With Shadows",
		};
		enum { 
			PlainColor,
			MaterialColor,
			PhongLighting,
			Textured,
			Masked,
			Normalmapped,
			Sky,
			Shadows,
			N
		};
		static int active_mode = Sky;
		if (ImGui::BeginCombo("Assignment", items[active_mode])) {
			for (int i = 0; i < N; ++i) {
				if (ImGui::Selectable(items[i], i==active_mode))
					active_mode = i;
				if (active_mode == i)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		static glm::vec3 dirlight_dir = glm::vec3(0.25,-.93,-.25);
		static glm::vec3 dirlight_col = glm::vec3(1.0,0.97,0.8);
		static float     dirlight_scale = 1.2f;
		static glm::vec3 pointlight_pos = glm::vec3(75,500,500);
		static glm::vec3 pointlight_col = glm::vec3(1.0,0.58,0.16);
		static float     pointlight_scale = 1.5f;
		ImGui::Separator();
		if (ImGui::SliderFloat3("Directional Light", &dirlight_dir.x, -1, 1))
			if (dirlight_dir == glm::vec3(0,0,0))
				dirlight_dir = glm::vec3(0,-1,0);
			else
				dirlight_dir = glm::normalize(dirlight_dir);
		ImGui::ColorEdit3("Directional Light Color", &dirlight_col.x);
		ImGui::SliderFloat("Directional Light Intensity Scale", &dirlight_scale, 0, 10);
		ImGui::Separator();
		ImGui::SliderFloat3("Point Light Position", &pointlight_pos.x, -500, 500);
		ImGui::ColorEdit3("Point Light Color", &pointlight_col.x);
		ImGui::SliderFloat("Point Light Intensity Scale", &pointlight_scale, 0, 20);
		ImGui::End();

        // update
        update_timer.start();
		{
			static float pi = M_PI;
			float dt = Context::frame_time();
			float v_max = 10;
			static float v = 0;
			if (Context::key_pressed(GLFW_KEY_H))
				heli_mat = glm::rotate(heli_mat, 0.001f*dt*pi, glm::vec3(0,1,0));
			if (Context::key_pressed(GLFW_KEY_L))
				heli_mat = glm::rotate(heli_mat, -0.001f*dt*pi, glm::vec3(0,1,0));
			if (Context::key_pressed(GLFW_KEY_K))
				heli_mat = glm::translate(heli_mat, glm::vec3(0, 1,0));
			if (Context::key_pressed(GLFW_KEY_J))
				heli_mat = glm::translate(heli_mat, glm::vec3(0,-1,0));
			if (Context::key_pressed(GLFW_KEY_SPACE))
				v = max(0.04f, min(v_max, v*1.05f));
			if (Context::key_pressed(GLFW_KEY_BACKSPACE))
				v = max(0.0f, v*0.92f);
			heli_mat = glm::translate(heli_mat, v*glm::vec3(0,0,-1));
			rotor = glm::rotate(rotor, 0.005f*dt*pi, glm::vec3(0,1,0));

			heli_cam->view = glm::inverse(heli_mat);
			
			shadow_cam->pos = -dirlight_dir*3000.0f;
			shadow_cam->dir = dirlight_dir;
			shadow_cam->update();
		}
		update_timer.end();

		// render sm
        render_sm_timer.start();
		if (active_mode == Shadows) {
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
					de->shader->uniform("pointlight_pos", pointlight_pos);
					de->shader->uniform("pointlight_col", pointlight_col);
					de->shader->uniform("pointlight_scale", pointlight_scale);
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

			heli_shadowmap->bind();
			render(heli_cam);
			heli_shadowmap->unbind();
			glDisable(GL_POLYGON_OFFSET_FILL);
			glCullFace(GL_BACK);

			cam->make_current();
		}
        render_sm_timer.end();

        // render
        render_timer.start();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if (active_mode == PlainColor) {
			shader_plain_color->bind();
			shader_plain_color->uniform("model", glm::mat4(1));
			shader_plain_color->uniform("view", Camera::current()->view);
			shader_plain_color->uniform("proj", Camera::current()->proj);
			int loc = glGetUniformLocation(shader_plain_color->id, "cols");
			glUniform4fv(loc, 100, (float*)&cols[0]);
			for (auto &de : sponza) {
				de->material->bind(shader_plain_color);
				de->draw(glm::mat4(1));
				de->material->unbind();
			}
			shader_plain_color->unbind();
		}
		else {
			for (auto &de : sponza) {
				shader_ptr shader;
				if (active_mode == MaterialColor)      shader = shader_material;
				else if (active_mode == PhongLighting) shader = shader_lighting;
				else if (active_mode == Textured)      shader = shader_textured;
				else if (active_mode == Masked)        shader = shader_masked;
				else if (active_mode == Normalmapped)  shader = shader_normalmapped;
				else if (active_mode == Sky)           shader = sky_shader;
				else if (active_mode >= Shadows)       shader = shader_shadows;

				de->shader = shader;
				de->bind();
				shader->uniform("cam_pos", Camera::current()->pos);
				shader->uniform("pointlight_pos", pointlight_pos);
				shader->uniform("pointlight_col", glm::pow(pointlight_col, glm::vec3(2.2f)));
				shader->uniform("pointlight_scale", pointlight_scale);
				shader->uniform("dirlight_dir", dirlight_dir);
				shader->uniform("dirlight_col", glm::pow(dirlight_col, glm::vec3(2.2f)));
				shader->uniform("dirlight_scale", dirlight_scale);
				shader->uniform("has_alphamap", de->material->has_texture("alphamap") ? 1 : 0);
				shader->uniform("has_normalmap", de->material->has_texture("normalmap") ? 1 : 0);
				shader->uniform("heli_col", glm::pow(pointlight_col, glm::vec3(2.2f)));
				shader->uniform("heli_scale", pointlight_scale);
				shader->uniform("shadowmap", shadowmap->depth_texture, 5);
				shader->uniform("heli_shadowmap", heli_shadowmap->depth_texture, 6);
				shader->uniform("shadow_V", shadow_cam->view);
				shader->uniform("shadow_P", shadow_cam->proj);
				shader->uniform("heli_V", heli_cam->view);
				shader->uniform("heli_P", heli_cam->proj);
				de->draw(glm::mat4(1));
				de->unbind();
			}
		}


		if ((int)active_mode >= PhongLighting) {
			for (auto &de : light_rep) {
				de->bind();
				light_rep_shader->uniform("pointlight_col", pointlight_col);
				de->draw(glm::scale(glm::translate(glm::mat4(1), pointlight_pos), glm::vec3(1,1,1)));
				de->unbind();
			}
			for (auto &de : heli) {
				de->bind();
				glm::mat4 model = glm::scale(glm::translate(glm::mat4(1), pointlight_pos), glm::vec3(8,8,8));
				model = glm::scale(heli_mat, glm::vec3(8,8,8));
				if (de->name.find("Primary_Rotor") != std::string::npos) {
					model = model * rotor;
				}
				de->draw(model);
				de->unbind();
			}
		}
		if ((int)active_mode >= Sky) {
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
		}
        render_timer.end();

        // finish frame
        Context::swap_buffers();
    }

    return 0;
}
