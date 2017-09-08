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
        m_camera = new Camera(glm::vec3(0.0f, 0.0f, 14.0f));
    }

    ~World() {
        for(size_t i = 0; i < m_entities.size(); i++) {
            Entity * curr = m_entities.at(i);
            if(curr) delete curr;
        }

        if (m_shader) delete m_shader;
        if (m_camera) delete m_camera;
    }

    void render() const {

        // OpenGL options
        glEnable(GL_DEPTH_TEST);

        // Clear the colorbuffer
        glClearColor(0.08f, 0.08f, 0.16f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup the shader based upon camera
        GLint viewPosLoc = glGetUniformLocation(m_shader->Program, "viewPos");
        glUniform3f(viewPosLoc, m_camera->Position.x, m_camera->Position.y, m_camera->Position.z);

        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);

        // Create m_camera transformations
        glm::mat4 view;
        view = m_camera->GetViewMatrix();
        glm::mat4 projection = glm::perspective(m_camera->Zoom, (GLfloat)w / (GLfloat)h, 0.1f, 100.0f);

        // Get the uniform locations
        GLint viewLoc  = glGetUniformLocation(m_shader->Program, "view");
        GLint projLoc  = glGetUniformLocation(m_shader->Program, "projection");

        // Pass view+projection matrices to the shader. This is done inside the World
        // because these matrices are the same for all entities. The World controls the 
        // m_camera which then navigates the global scene of all entities.
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Draw ALL the things!
        for(size_t i = 0; i < m_entities.size(); i++) {
            Entity * curr = m_entities.at(i);
            if(curr) curr->render();
        }

        // Swap the screen buffers using a double-buffer:
        // Just before this line, we rendered the world into the back 
        // (hidden) buffer, so we now swap that buffer into the front 
        // while moving the currently showing frame's buffer to the back.
        // Any OpenGL rendering after this line will render to the new
        // back (newly hidden) buffer and will not show up onscreen until
        // the next time we swap the buffers.
        glfwSwapBuffers(m_window);
    }

    void update(double time_since_last_update) {

        for(size_t i = 0; i < m_entities.size(); i++) {
            Entity * curr = m_entities.at(i);
            
            if(!curr) continue;
            
            handleCollisions(curr);
            curr->update(time_since_last_update);
        }
    }
    
    void addEntity(Entity * entity) {
        m_entities.push_back(entity);
    }

    Shader * m_shader;
    Camera * m_camera;

private:

    GLFWwindow * m_window;

    void handleCollisions(Entity * entity) {
        // The state engine might do things here like handle 
        // physical collisions between the world's entities 
    }
    
    vector<Entity *> m_entities;
};
