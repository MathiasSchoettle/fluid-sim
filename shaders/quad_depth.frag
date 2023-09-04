#version 440 core

layout (location = 3) out vec4 g_depth_new;

in vec2 tex_coords;

uniform sampler2D g_depth;
uniform float n;
uniform float f;

const float u_ScreenWidth = 1920;
const float u_ScreenHeight = 1080;
const float thresholdRatio = 20.0;
const float u_ParticleRadius = 1;
const int u_FilterSize = 5;
const int u_MaxFilterSize = 15;

float compute_weight2D(vec2 r, float two_sigma2) {
	return exp(-dot(r, r) / two_sigma2);
}

float compute_weight1D(float r, float two_sigma2) {
	return exp(-r * r / two_sigma2);
}

void main()
{
	vec2 f_TexCoord = tex_coords;
	vec2 blurRadius = vec2(1.0 / float(u_ScreenWidth), 1.0 / float(u_ScreenHeight));

	vec4 pixelDepth = texture(g_depth, f_TexCoord);
	vec4 finalDepth = pixelDepth;

	if(pixelDepth.w > f || pixelDepth.w < n) {
		finalDepth = pixelDepth;
	} else {
		float ratio      = u_ScreenHeight / 2.0 / tan(45.0 / 2.0);
		float K          = u_FilterSize * ratio * u_ParticleRadius * 0.1f;
		int   filterSize = min(u_MaxFilterSize, int(ceil(K / pixelDepth.w)));
		float sigma      = filterSize / 3.0f;
		float two_sigma2 = 2.0f * sigma * sigma;

		float threshold       = u_ParticleRadius * thresholdRatio;
		float sigmaDepth      = threshold / 3.0f;
		float two_sigmaDepth2 = 2.0f * sigmaDepth * sigmaDepth;

		vec4 f_tex = f_TexCoord.xyxy;
		vec2 r     = vec2(0, 0);
		vec4 sum4  = vec4(pixelDepth.w, 0, 0, 0);
		vec4 wsum4 = vec4(1, 0, 0, 0);
		vec4 sampleDepth;
		vec4 w4_r;
		vec4 w4_depth;
		vec4 rDepth;

		for(int x = 1; x <= filterSize; ++x) {
			r.x     += blurRadius.x;
			f_tex.x += blurRadius.x;
			f_tex.z -= blurRadius.x;
			vec4 f_tex1 = f_tex.xyxy;
			vec4 f_tex2 = f_tex.zwzw;

			for(int y = 1; y <= filterSize; ++y) {
				r.y += blurRadius.y;

				f_tex1.y += blurRadius.y;
				f_tex1.w -= blurRadius.y;
				f_tex2.y += blurRadius.y;
				f_tex2.w -= blurRadius.y;

				sampleDepth.x = texture(g_depth, f_tex1.xy).w;
				sampleDepth.y = texture(g_depth, f_tex1.zw).w;
				sampleDepth.z = texture(g_depth, f_tex2.xy).w;
				sampleDepth.w = texture(g_depth, f_tex2.zw).w;

				rDepth     = sampleDepth - vec4(pixelDepth.w);
				w4_r       = vec4(compute_weight2D(blurRadius * r, two_sigma2));
				w4_depth.x = compute_weight1D(rDepth.x, two_sigmaDepth2);
				w4_depth.y = compute_weight1D(rDepth.y, two_sigmaDepth2);
				w4_depth.z = compute_weight1D(rDepth.z, two_sigmaDepth2);
				w4_depth.w = compute_weight1D(rDepth.w, two_sigmaDepth2);

				sum4  += sampleDepth * w4_r * w4_depth;
				wsum4 += w4_r * w4_depth;
			}
		}
		
		finalDepth.w = length(sum4) / length(wsum4);
	}

	g_depth_new = finalDepth;
}
