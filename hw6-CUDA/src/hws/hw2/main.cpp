// CS148 Summer 2017 Homework 2 
// Real-Time Graphics Boot Camp

#include <iostream>
#include <cmath>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
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

using namespace std;

// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void handleInput();

const glm::vec3 kLightPos = glm::vec3(0.0f, 0.0f, 9.0f);

// Window dimensions
const GLuint WIDTH = 800, HEIGHT = 600;

GLfloat lastX  =  WIDTH  / 2.0;
GLfloat lastY  =  HEIGHT / 2.0;
bool    keys[1024];

// Deltatime
GLfloat deltaTime = 0.0f;   // Time between current frame and last frame
GLfloat timeOfLastFrame = 0.0f;   // Time of last frame

class World; // forward declaration
World * g_world;

GLFWwindow* setupWindow() 
{
    // Init GLFW
    glfwInit();
    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "HW2 - OpenGL Boot Camp", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    // GLFW Optional settings
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // don't show the cursor

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions
    glewExperimental = GL_TRUE;

    // Initialize GLEW to setup the OpenGL Function pointers
    glewInit();

    // Define the viewport dimensions
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);

    return window;
}

void setupInputHandlers(GLFWwindow * window) {
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (key >= 0 && key < 1024)
    {
        
        // TODO: you might check here for key releases
        // for the various parts of this assignment.
        // E.g. if "p" is released, key == GLFW_KEY_P
        // should be true, and action == GLFW_RELEASE
        // should also be true.

        if (action == GLFW_PRESS) {
            keys[key] = true;
        } else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }
}

// Control camera movement with keyboard keys
void handleInput()
{
    glfwPollEvents();

    if (!g_world || !g_world->m_camera) {
        return;
    }

    if (keys[GLFW_KEY_W]) g_world->m_camera->ProcessKeyboard(FORWARD, deltaTime);
    if (keys[GLFW_KEY_S]) g_world->m_camera->ProcessKeyboard(BACKWARD, deltaTime);
    if (keys[GLFW_KEY_A]) g_world->m_camera->ProcessKeyboard(LEFT, deltaTime);
    if (keys[GLFW_KEY_D]) g_world->m_camera->ProcessKeyboard(RIGHT, deltaTime);
}

// Control camera movement with mouse motion
bool firstMouse = true;
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset = lastY - ypos;  // Reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    g_world->m_camera->ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_world->m_camera->ProcessMouseScroll(yoffset);
}

void cleanup() {
    // Terminate GLFW, clearing any resources it allocated (e.g. the window).
    glfwTerminate();

    if (g_world) delete g_world;
}

// Create the universe!
void setupWorld(GLFWwindow * window) {
    g_world = new World(window);
    g_world->addEntity(new Light(g_world->m_shader, kLightPos));

    // Create a single cube and add it to the center of the world
    BreathingCube * c = new BreathingCube(g_world->m_shader, glm::vec3(0,0,0), glm::vec4(0.7f, 0.17f, 0.17f, 1.0f));
    g_world->addEntity(c);
}

// Main entry point
int main()
{
    srand(time(NULL)); // seed random number generation

    GLFWwindow * window = setupWindow(); // GLFW creates a window using whichever OS you're on
    setupInputHandlers(window); // GLFW also handles input from keyboard, mouse, etc.
    setupWorld(window); // Create our universe

    // The "run loop" (exit when "esc" is pressed)
    // Every properly architechted real-time graphics app uses this loop -
    // You should Understand the design of this loop as deeply as possible,
    // Think: why and how does it leverage MVC? What does it enforce?
    // What problems does it prevent?
    while (!glfwWindowShouldClose(window))
    {
        // Calculate deltatime of current frame
        GLfloat timeOfCurrentFrame = glfwGetTime();
        deltaTime = timeOfCurrentFrame - timeOfLastFrame;
        timeOfLastFrame = timeOfCurrentFrame;

        handleInput(); // Check latest input state
        g_world->update(deltaTime); // Update the world's state (the "truth" - alter entity locations, transformation values, physics calculations, etc.)
        g_world->render(); // Draw the world in its current state (render from the model in a read-only manner)
    }

    cleanup();

    return 0;
}
