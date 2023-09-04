#version 440 core

layout (location = 0) out vec4 g_norm;

in vec2 tex_coords;

uniform sampler2D g_depth;
uniform float n;
uniform float f;
uniform mat4 proj;

int u_ScreenWidth = 1920;
int u_ScreenHeight = 1080;

vec3 uvToEye(vec2 texCoord, float depth)
{
	float x  = texCoord.x * 2.0 - 1.0;
	float y  = texCoord.y * 2.0 - 1.0;
	float zn = ((f + n) / (f - n) * depth + 2 * f * n / (f - n)) / depth;

	vec4 clipPos = vec4(x, y, zn, 1.0f);
	vec4 viewPos = inverse(proj) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main()
{
	vec2 f_TexCoord = tex_coords;

	float pixelWidth  = 1 / float(u_ScreenWidth);
	float pixelHeight = 1 / float(u_ScreenHeight);
	float x           = f_TexCoord.x;
	float y           = f_TexCoord.y;

	float depth   = texture(g_depth, vec2(x, y)).w;

	if(depth > f || depth < n) {
		g_norm = vec4(0, 0, 0, 1);
		return;
	}

	float depthxp = texture(g_depth, vec2(x+pixelWidth, y)).w;
	float depthxn = texture(g_depth, vec2(x-pixelWidth, y)).w;
	float dzdx = (depthxp - depthxn) / 2.0f;

	float depthyp = texture(g_depth, vec2(x, y+pixelWidth)).w;
	float depthyn = texture(g_depth, vec2(x, y-pixelWidth)).w;
	float dzdy = (depthyp - depthyn) / 2.0f;

	vec3 n = normalize(vec3(dzdx, dzdy, 1));

	g_norm = vec4(n, 1);
}
