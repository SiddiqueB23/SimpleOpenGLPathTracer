#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <openglDebug.h>
#include <demoShaderLoader.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <stack>
//#include "imgui.h"
//#include "backends/imgui_impl_opengl3.h"
//#include "backends/imgui_impl_glfw.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include <tiny_obj_loader.h>

#define USE_GPU_ENGINE 0
extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = USE_GPU_ENGINE;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = USE_GPU_ENGINE;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int WIN_WIDTH = 1280;
int WIN_HEIGHT = 720;
int TEXTURE_WIDTH = 1280;
int TEXTURE_HEIGHT = 720;
std::vector<int>WindowSizeUniforms;

float vfov = 90; //vertical fov of camera
float lookfrom[3] = { 0, 0, -1 }; // Point camera is looking from
float lookat[3] = { 0, 0, 0 }; // Point camera is looking at
float defocus_angle = 0.5;

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
	WIN_WIDTH = width;
	WIN_HEIGHT = height;
	for (auto i : WindowSizeUniforms)
		glUniform2f(i, (float)WIN_WIDTH, (float)WIN_HEIGHT);
}

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
	if (quadVAO == 0)
	{
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

/*
struct Sphere {
	alignas(16) float pos[3];
	float r;
	int materialIndex;
};
const int sphereCount = 5;
Sphere S[sphereCount];
*/
struct Material {
	float albedo[4];
	float emissive_colour[4];
	float emission_strength;
	int type;
	float smoothness;
	float ri;
};
const int materialCount = 5;
Material M[materialCount];

struct AABB {
	alignas(16)glm::vec3 min_v;
	alignas(16)glm::vec3 max_v;
};

AABB combineAABB(const AABB& aabb1, const AABB& aabb2)
{
	return AABB{
		glm::min(aabb1.min_v,aabb2.min_v),
		glm::max(aabb1.max_v,aabb2.max_v),
	};
}

struct BVHNode {
	AABB aabb;
	int start, end;
	int lc = -1,rc = -1;
};

struct triangle_index {
	tinyobj::index_t v0, v1, v2;
};

std::vector<BVHNode> generateBVH(std::vector<triangle_index>&VI,const std::vector<tinyobj::real_t>&VB,int start,int end,int max_depth) 
{
	std::vector<BVHNode>BVH;
	std::stack<std::pair<int,int>>stk;

	AABB aabb;
	aabb.min_v = glm::vec3(1000.0f, 1000.0f, 1000.0f);
	aabb.max_v = glm::vec3(-1000.0f, -1000.0f, -1000.0f);
	std::vector<AABB>aabb_cache(VI.size());
	for (int i = 0; i < VI.size(); i++)
	{
		glm::vec3 v0 = glm::vec3(VB[VI[i].v0.vertex_index * 3], VB[VI[i].v0.vertex_index * 3 + 1], VB[VI[i].v0.vertex_index * 3 + 2]);
		glm::vec3 v1 = glm::vec3(VB[VI[i].v1.vertex_index * 3], VB[VI[i].v1.vertex_index * 3 + 1], VB[VI[i].v1.vertex_index * 3 + 2]);
		glm::vec3 v2 = glm::vec3(VB[VI[i].v2.vertex_index * 3], VB[VI[i].v2.vertex_index * 3 + 1], VB[VI[i].v2.vertex_index * 3 + 2]);
		aabb_cache[i] = AABB{
			glm::min(v0,glm::min(v1,v2)),
			glm::max(v0,glm::max(v1,v2))
		};
		aabb = combineAABB(aabb, aabb_cache[i]);
	}

	std::vector<int>VI_index(VI.size());
	for (int i = 0; i < VI.size(); i++)VI_index[i] = i;

	BVH.push_back(BVHNode{ aabb,start,end,-1,-1 });
	int curr_depth = 1;
	stk.push({0,1});
	int max_stk_size = 1;
	while (!stk.empty()) {
		max_stk_size = std::max(max_stk_size, (int)stk.size());
		BVHNode& bvhnode = BVH[stk.top().first]; 
		curr_depth = stk.top().second;
		stk.pop();
		if (curr_depth < max_depth && bvhnode.end-bvhnode.start>4) 
		{
			glm::vec3 aabbdiff = bvhnode.aabb.max_v - bvhnode.aabb.min_v;
			if (aabbdiff.x > aabbdiff.y && aabbdiff.x > aabbdiff.z)
				std::sort(
					VI_index.begin()+ bvhnode.start,
					VI_index.begin() + bvhnode.end+1, 
					[&aabb_cache](const int& a, const int& b) {return aabb_cache[a].max_v.x < aabb_cache[b].max_v.x; }
				);
			else if (aabbdiff.y > aabbdiff.x && aabbdiff.y > aabbdiff.z)
				std::sort(VI_index.begin() + bvhnode.start,
					VI_index.begin() + bvhnode.end + 1,
					[&aabb_cache](const int& a, const int& b) {return aabb_cache[a].max_v.y < aabb_cache[b].max_v.y; }
				);
			else 
				std::sort(VI_index.begin() + bvhnode.start,
					VI_index.begin() + bvhnode.end + 1,
					[&aabb_cache](const int& a, const int& b) {return aabb_cache[a].max_v.z < aabb_cache[b].max_v.z; }
				);
			//for (int i = 0; i < 100; i++)std::cout << aabb_cache[VI_index[i]].min_v.x << ' '; std::cout << "\n";
			int mid = (bvhnode.start + bvhnode.end) / 2;
			AABB aabbl = aabb_cache[VI_index[bvhnode.start]], aabbr = aabb_cache[VI_index[mid]];
			for (int i = bvhnode.start; i < mid; i++)
				aabbl = combineAABB(aabbl, aabb_cache[VI_index[i]]);
			for (int i = mid; i <= bvhnode.end; i++)
				aabbr = combineAABB(aabbr, aabb_cache[VI_index[i]]);
			bvhnode.lc = BVH.size();
			bvhnode.rc = BVH.size()+1;
			BVH.push_back(BVHNode{ aabbl,bvhnode.start,mid - 1,-1,-1 });
			BVH.push_back(BVHNode{ aabbr,mid,bvhnode.end,-1,-1 });
			stk.push({ bvhnode.lc , curr_depth + 1 });
			stk.push({ bvhnode.rc , curr_depth + 1 });
		}

	}
	std::cout << max_stk_size << std::endl;
	std::vector<triangle_index>temp_VI = VI;
	for (int i = 0; i < VI.size(); i++)
		VI[i] = temp_VI[VI_index[i]];
		//VI[VI_index[i]] = temp_VI[i];
	/*
	for (int i = 0; i < 100; i++)std::cout << std::min({ VB[VI[i].v0.vertex_index * 3], VB[VI[i].v1.vertex_index * 3] ,VB[VI[i].v2.vertex_index * 3] }) << " "; std::cout << "\n";
	for (int i = 0; i < BVH.size(); i++) {
		std::cout << "BVHNode" << i << "\n";
		std::cout << glm::to_string(BVH[i].aabb.min_v) << ' ' << glm::to_string(BVH[i].aabb.max_v) << '\n';
		std::cout << BVH[i].lc << " " << BVH[i].rc << "\n";
		std::cout << BVH[i].start << " " << BVH[i].end << "\n\n";
	}
	std::cout << "\n";
	*/
	return BVH;
}

int main(void)
{

	if (!glfwInit())
		return -1;


#pragma region report opengl errors to std
	//enable opengl debugging output.
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#pragma endregion


	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //you might want to do this when testing the game for shipping


	GLFWwindow* window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "RayTracer", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	glfwSwapInterval(0);
	glfwSetWindowSizeCallback(window, glfw_window_size_callback);

	//Setup ImGui
	//ImGui::CreateContext();
	//ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//ImGui::StyleColorsDark();
	//ImGui_ImplGlfw_InitForOpenGL(window, true);
	//ImGui_ImplOpenGL3_Init("#version 330");

#pragma region report opengl errors to std
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glDebugOutput, 0);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#pragma endregion

	Shader screenQuad, computeShader;
	screenQuad.loadShaderProgramFromFile(RESOURCES_PATH "screenQuad_vert.glsl", RESOURCES_PATH "screenQuad_frag.glsl");
	//computeShader.loadComputeShaderProgramFromFile(RESOURCES_PATH "computeShader.glsl");
	computeShader.loadComputeShaderProgramFromFile(RESOURCES_PATH "computeShaderTri.glsl");
	/*
	// Get the binary length
	GLint length = 0;
	glGetProgramiv(computeShader.id, GL_PROGRAM_BINARY_LENGTH, &length);

	// Retrieve the binary code
	std::vector<GLubyte> buffer(length);
	GLenum format = 0;
	glGetProgramBinary(computeShader.id, length, NULL, &format, buffer.data());

	// Write the binary to a file.
	std::string fName(RESOURCES_PATH "shader.bin");
	std::cout << "Writing to " << fName << ", binary format = " << format << std::endl;
	std::ofstream out(fName.c_str(), std::ios::binary);
	out.write(reinterpret_cast<char*>(buffer.data()), length);
	out.close();
	*/

	screenQuad.bind();
	glUniform1i(screenQuad.getUniform("tex"), 0);

	unsigned int texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	/*
	S[0] = Sphere{ {0.0, 0.0, 3.0}, 1.0, 4 };
	S[1] = Sphere{ {2.1, 0.0, 3.0}, 1.0, 1 };
	S[2] = Sphere{ {-2.1, 0.0, 3.0}, 1.0, 2 };
	S[3] = Sphere{ {0.0, -101.0, 3.0}, 100.0, 0 };
	S[4] = Sphere{ {-2.1, 0.0, 3.0}, 0.8, 3 };

	GLuint ssbo1;
	glGenBuffers(1, &ssbo1);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo1);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(S), S, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo1);
	*/

	//objloader
	std::string inputfile = RESOURCES_PATH "dragon05.obj";
	tinyobj::ObjReaderConfig reader_config;
	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(inputfile, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	std::cout << attrib.vertices.size() << std::endl;
	float* vertices = (float*)attrib.vertices.data();
	/*float vertices[] =
	{
			3.0, 0.0, 6.0,
			-3.0, 0.0, 6.0,
			0.0, 3.0, 6.0,
			0.0, -3.0, 6.0,
	};*/
	GLuint VB;
	glGenBuffers(1, &VB);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, VB);
	glBufferData(GL_SHADER_STORAGE_BUFFER, attrib.vertices.size() * sizeof(float), vertices, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, VB);

	/*
	int triCount = 2;
	int vertex_indices[] =
	{
		0,1,2,
		0,1,3
	};*/
	std::cout << shapes[0].mesh.num_face_vertices.size() * 3 << " " << shapes[0].mesh.indices.size() << std::endl;
	int triCount = shapes[0].mesh.num_face_vertices.size();
	std::vector<triangle_index>vertex_indices(triCount);
	for (size_t i = 0; i < triCount; i++)
	{
		vertex_indices[i] = triangle_index{
			shapes[0].mesh.indices[i * 3],
			shapes[0].mesh.indices[i * 3 + 1],
			shapes[0].mesh.indices[i * 3 + 2]
		};
	}
	std::vector<BVHNode>BVH = generateBVH(vertex_indices, attrib.vertices, 0, triCount - 1, 15);
	//triangle_index* vertex_indices = (triangle_index*)shapes[0].mesh.indices.data();
	std::cout << "Number of faces: " << vertex_indices.size() << std::endl;
	std::cout << "BVH size: " << BVH.size() << "\n";
	GLuint VI;
	glGenBuffers(1, &VI);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, VI);
	glBufferData(GL_SHADER_STORAGE_BUFFER, vertex_indices.size() * sizeof(triangle_index), vertex_indices.data(), GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, VI);


	std::cout << glm::to_string(BVH[0].aabb.min_v) << " " << glm::to_string(BVH[0].aabb.max_v) << std::endl;
	GLuint BVHBuffer;
	glGenBuffers(1, &BVHBuffer);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, BVHBuffer);
	glBufferData(GL_SHADER_STORAGE_BUFFER, BVH.size()*sizeof(BVHNode), BVH.data(), GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, BVHBuffer);

	M[0] = Material{ {0.5, 0.5, 0.5, 1.0}, {0.0, 0.0, 0.0, 0.0}, 0, 0, 0.0, 0.0 };
	M[1] = Material{ {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, 0, 0, 0.3, 0.1 };
	M[2] = Material{ {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, 0, 1, 0.0, 1.5 };
	M[3] = Material{ {1.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, 0, 1, 0.0, 1.0 / 1.5 };
	M[4] = Material{ {0.0, 0.0, 0.0, 1.0}, {1.0, 1.0, 1.0, 1.0}, 1, 0, 0.0, 0.0 };
	GLuint ssbo2;
	glGenBuffers(1, &ssbo2);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo2);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(M), M, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo2);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


	float deltaTime = 0.0f, currentFrame = 0.0f, lastFrame = 0.0f;
	int fCounter = 0;
	float frameCount = 0;
	char inp;
	//std::cin >> inp;
	computeShader.bind();
	glm::vec3 lookfromvec = glm::rotateY(glm::vec3(lookfrom[0], lookfrom[1], lookfrom[2] ), 1.0f);
	lookfrom[0] = lookfromvec.x; lookfrom[1] = lookfromvec.y; lookfrom[2] = lookfromvec.z;
	glm::vec3 lookatvec = glm::rotateY(glm::vec3(lookat[0], lookat[1], lookat[2] ), 1.0f);
	lookat[0] = lookatvec.x; lookat[1] = lookatvec.y; lookat[2] = lookatvec.z;
	glUniform1f(computeShader.getUniform("vfov"), vfov);
	glUniform1f(computeShader.getUniform("defocus_angle"), defocus_angle);
	glUniform3f(computeShader.getUniform("lookfrom"), lookfrom[0], lookfrom[1], lookfrom[2]);
	glUniform3f(computeShader.getUniform("lookat"), lookat[0], lookat[1], lookat[2]);
	//glUniform1i(computeShader.getUniform("triCount"), triCount);

	while (!glfwWindowShouldClose(window))
	{
		/*
		if (fCounter > 200) {
			std::cout << "FPS: " << 1 / deltaTime << std::endl;
			fCounter = 1;
		}
		else {
			fCounter++;
		}
		std::cout << deltaTime*1000 << "\n";
		*/
		currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		computeShader.bind();
		glUniform1f(computeShader.getUniform("frameCount"), frameCount);
		//glUniform3f(computeShader.getUniform("lookfrom"), lookfrom[0], lookfrom[1], lookfrom[2]);
		//glUniform3f(computeShader.getUniform("lookat"), lookat[0], lookat[1], lookat[2]);
		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo1);
		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo1);
		//glUniform1i(computeShader.getUniform("sphereCount"), sphereCount);
		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, VB);
		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, VB);
		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, VI);
		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, VI);
		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo2);
		//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo2);
		glDispatchCompute(	 (unsigned int)TEXTURE_WIDTH/8,
							(unsigned int)TEXTURE_HEIGHT/8, 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		glClear(GL_COLOR_BUFFER_BIT);
		screenQuad.bind();
		renderQuad();
		glMemoryBarrier(GL_ALL_BARRIER_BITS);
		frameCount++;
		/*
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("Settings");
		if (ImGui::Button("Clear Render")) frameCount = 0.0;
		if (ImGui::SliderFloat("vfov", &vfov, 1.0, 179.0)) frameCount = 0.0;
		if (ImGui::DragFloat("defocus_angle", &defocus_angle, 0.1, 0.0, 179.0)) frameCount = 0.0;
		if (ImGui::DragFloat3("lookfrom", lookfrom, 0.5, -10.0, 10.0)) frameCount = 0.0;
		if (ImGui::DragFloat3("lookat", lookat, 0.5, -10.0, 10.0)) frameCount = 0.0;
		ImGui::End();
		
		ImGui::Begin("Spheres");
		for (int i = 0; i < sphereCount; i++) {
			{
				ImGui::BeginChild("##scrolling");
				ImGui::Text(std::to_string(i).c_str());
				ImGui::PushID(i);
				if (ImGui::DragFloat3("pos", S[i].pos, 0.5, -10.0, 10.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Sphere) * i, sizeof(Sphere), S + i);
					frameCount = 0.0;
				}
				if (ImGui::DragFloat("radius", &S[i].r, 0.5, 0.0, 10.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Sphere) * i, sizeof(Sphere), S + i);
					frameCount = 0.0;
				}
				if (ImGui::DragInt("materialindex", &S[i].materialIndex, 0.5, 0.0, 10.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Sphere) * i, sizeof(Sphere), S + i);
					frameCount = 0.0;
				}
				ImGui::PopID();
				ImGui::EndChild();
			}
		}
		ImGui::End();
		
		ImGui::Begin("Materials");
		for (int i = 0; i < materialCount; i++) {
			{
				ImGui::BeginChild("##scrolling");
				ImGui::Text(std::to_string(i).c_str());
				ImGui::PushID(i);
				if (ImGui::DragInt("type", &M[i].type, 1.0, 0.0, 10.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				if (ImGui::ColorEdit4("albedo", M[i].albedo)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				if (ImGui::ColorEdit4("emissive_colour", M[i].emissive_colour)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				if (ImGui::DragFloat("emission_strength", &M[i].emission_strength, 0.1, 0.0, 10.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				if (ImGui::DragFloat("ri/specular_chance", &M[i].ri, 0.05, 0.0, 1.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				if (ImGui::DragFloat("smoothness", &M[i].smoothness, 0.05, 0.0, 1.0)) {
					glBufferSubData(GL_SHADER_STORAGE_BUFFER, sizeof(Material) * i, sizeof(Material), M + i);
					frameCount = 0.0;
				}
				ImGui::PopID();
				ImGui::EndChild();
			}
		}
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		*/
		glfwSwapBuffers(window);
		glfwPollEvents();
		std::cin >> inp;	
	}

	//ImGui_ImplOpenGL3_Shutdown();
	//ImGui_ImplGlfw_Shutdown();
	//ImGui::DestroyContext();

	return 0;
}
