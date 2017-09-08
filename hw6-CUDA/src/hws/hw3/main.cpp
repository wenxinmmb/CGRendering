//============================================================
//
// CS 148 (Summer 2016) - Assignment 3 - Meshes & Smoothing
//
//============================================================

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// GLM Mathematics
#define GLM_FORCE_RADIANS // force everything in radian
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

// Other includes
#include "Shader.h"
#include "STLib.h"

const int WIDTH = 768;
const int HEIGHT = 512;
const unsigned int g_k_initial_num_iterations = 0;

// The meshes
Mesh * g_unaltered_mesh;
Mesh * g_smoothed_mesh;
Mesh * g_bunny_1;
Mesh * g_bunny_2;

// Inspection controls
bool g_just_clicked = false;
double g_mouse_x = 0.0;
double g_mouse_y = 0.0;
double g_mouse_x_prev = 0.0;
double g_mouse_y_prev = 0.0;
double g_rot_angle_x = 0.0;
double g_rot_angle_y = 0.0;

GLfloat g_last_x  =  WIDTH  / 2.0;
GLfloat g_last_y  =  HEIGHT / 2.0;

GLfloat deltaTime = 0.0f; // Time between current frame and last frame
GLfloat lastFrame = 0.0f; // Time of last frame
bool g_keys[1024];

// Function prototypes
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void handleInput();

World * g_world;

//--------------------------------------------------------- 
// Your task is to implement this method by setting the
// current mesh to a bunny which models "numIterations" 
// of Laplacian smoothing.
//---------------------------------------------------------
void computeLaplacianSmoothedMesh(unsigned int numIterations)
{
    // We'll double-buffer with two shapes
    if (g_bunny_1) delete g_bunny_1;
    if (g_bunny_2) delete g_bunny_2;

#ifdef WIN32
	g_bunny_1 = new Mesh("../bunny.obj", g_world->m_shader, glm::vec3(0));
	g_bunny_2 = new Mesh("../bunny.obj", g_world->m_shader, glm::vec3(0));
#else
    g_bunny_1 = new Mesh("bunny.obj", g_world->m_shader, glm::vec3(0));
    g_bunny_2 = new Mesh("bunny.obj", g_world->m_shader, glm::vec3(0));
#endif

    if (numIterations == 0) {
        g_smoothed_mesh = g_bunny_1;
    }

    // Assignment Task 3: Smooth the g_smoothed_mesh.

    // Using the two above bunnies, g_bunny_1 and
    // g_bunny_2, along with g_unaltered_mesh,
    // to set the final g_smoothed_mesh according to
    // "numIterations" iterations of laplacian smoothing

    g_smoothed_mesh = g_bunny_1; // replace this line using your algorithm

    g_world->setNewSmoothedMesh(g_smoothed_mesh);
}

//--------------------------------------------------------- 
// Handle Key Presses
//---------------------------------------------------------
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (key >= 0 && key < 1024)
    {
        if (action == GLFW_RELEASE) {
          switch(key) {
            case GLFW_KEY_1: { computeLaplacianSmoothedMesh(1); break; }
            case GLFW_KEY_2: { computeLaplacianSmoothedMesh(2); break; }
            case GLFW_KEY_3: { computeLaplacianSmoothedMesh(4); break; }
            case GLFW_KEY_4: { computeLaplacianSmoothedMesh(8); break; }
            case GLFW_KEY_5: { computeLaplacianSmoothedMesh(16); break; }
            case GLFW_KEY_6: { computeLaplacianSmoothedMesh(32); break; }
            case GLFW_KEY_7: { computeLaplacianSmoothedMesh(64); break; }
            default: break;
          }
        }

        if (action == GLFW_PRESS) {
            g_keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            g_keys[key] = false;
        }
    }
}

//--------------------------------------------------------- 
// Setup interface with the OS's windowing system
//---------------------------------------------------------
GLFWwindow * setupWindow() 
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "HW3 - Meshes & Smoothing", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // don't show the cursor
    glewExperimental = GL_TRUE;
    glewInit();

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);

    return window;
}

//--------------------------------------------------------- 
// Setup methods which handle input from hardware
//---------------------------------------------------------
void setupInputHandlers(GLFWwindow * window) {
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
}

//--------------------------------------------------------- 
// Cleanup before application exits
//---------------------------------------------------------
void cleanup() {
    glfwTerminate();
    if (g_world) delete g_world;
}

//--------------------------------------------------------- 
// Control camera movement with mouse motion
//---------------------------------------------------------
bool firstMouse = true;
void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        g_last_x = xpos;
        g_last_y = ypos;
        firstMouse = false;
    }

    GLfloat xoffset = xpos - g_last_x;
    GLfloat yoffset = g_last_y - ypos;  // Reversed since y-coordinates go from bottom to top

    g_last_x = xpos;
    g_last_y = ypos;

    if (g_world->m_camera) g_world->m_camera->ProcessMouseMovement(xoffset, yoffset);
}

//--------------------------------------------------------- 
// Setup our world with the light and rabbit!
//---------------------------------------------------------
void setupWorld(GLFWwindow * window) {
    g_world = new World(window);
    g_world->setLight(new Light(g_world->m_shader, glm::vec3(1.0f, 0.75f, 1.5f)));

    computeLaplacianSmoothedMesh(g_k_initial_num_iterations);
}

//--------------------------------------------------------- 
// Control camera movement with keyboard keys
//---------------------------------------------------------
void handleInput()
{
    glfwPollEvents();

    if (!g_world || !g_world->m_camera) {
        return;
    }

    if (g_keys[GLFW_KEY_W]) g_world->m_camera->ProcessKeyboard(FORWARD, deltaTime);
    if (g_keys[GLFW_KEY_S]) g_world->m_camera->ProcessKeyboard(BACKWARD, deltaTime);
    if (g_keys[GLFW_KEY_A]) g_world->m_camera->ProcessKeyboard(LEFT, deltaTime);
    if (g_keys[GLFW_KEY_D]) g_world->m_camera->ProcessKeyboard(RIGHT, deltaTime);
}

//--------------------------------------------------------- 
// Entry point
//---------------------------------------------------------
int main(int argc, char ** argv)
{
    srand(time(NULL));

    GLFWwindow * window = setupWindow();
    setupInputHandlers(window);
    setupWorld(window);

    while (!glfwWindowShouldClose(window))
    {
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        handleInput();
        g_world->update(deltaTime);
        g_world->render();
    }

    cleanup();

    return 0;
}
