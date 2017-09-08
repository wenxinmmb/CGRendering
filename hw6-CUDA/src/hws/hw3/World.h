#include <vector>

#include <GLFW/glfw3.h>

// GLM Mathematics
#define GLM_FORCE_RADIANS // force everything in radian
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include "Camera.h"

class Entity;

using namespace std;

//===========================================================
// Class : World
// The entire world in which all entities exist
//===========================================================

class World
{

public:

    World(GLFWwindow * window) {
        m_window = window;
#ifdef WIN32
        m_shader = new Shader("../phong.vs", "../phong.frag");
#else
		m_shader = new Shader("./phong.vs", "./phong.frag");
#endif
        m_camera = new Camera(glm::vec3(-0.01f, 0.53f, 1.0f));
    }

    ~World() {
        if (m_light)  delete m_light;
        if (m_mesh)   delete m_mesh;
        if (m_shader) delete m_shader;
        if (m_camera) delete m_camera;
    }

    void setLight(Light * light) {
        m_light = light;
    }

    void setNewSmoothedMesh(Mesh * mesh) {
        if (!m_mesh) {
            printf("World.h:: Warning - Attempt to set new mesh to null");
            return;
        }
        m_mesh = mesh;
        m_mesh->refresh();
    }

    void render() const {
        glEnable(GL_DEPTH_TEST);

        glClearColor(0.08f, 0.08f, 0.16f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GLint viewPosLoc = glGetUniformLocation(m_shader->Program, "viewPos");
        glUniform3f(viewPosLoc, m_camera->Position.x, m_camera->Position.y, m_camera->Position.z);

        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);

        glm::mat4 view;
        view = m_camera->GetViewMatrix();
        glm::mat4 projection = glm::perspective(m_camera->Zoom, (GLfloat)w / (GLfloat)h, 0.1f, 100.0f);

        GLint viewLoc  = glGetUniformLocation(m_shader->Program, "view");
        GLint projLoc  = glGetUniformLocation(m_shader->Program, "projection");

        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        m_light->render();
        m_mesh->render();

        glfwSwapBuffers(m_window);
    }

    void update(double time_since_last_update) {
        if (m_light) m_light->update(time_since_last_update);
        if (m_mesh) m_mesh->update(time_since_last_update);
    }

    Shader * m_shader;
    Camera * m_camera;

private:

    GLFWwindow * m_window;
    
    Light * m_light;
    Mesh * m_mesh;
};
