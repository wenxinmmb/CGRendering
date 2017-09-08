class Entity;

class BreathingCube : public Entity
{

public:
    
    BreathingCube(Shader * shader, glm::vec3 initial_pos, glm::vec4 initial_color, bool should_breathe) {
        m_type = ET_CUBE;

        m_breathing_enabled = should_breathe;

        m_total_time = 0.0;
        m_shader = shader;
        m_initial_pos = initial_pos;
        m_pos = initial_pos;

        m_initial_color = initial_color;

        m_random_delta = 0.5 * (m_pos.x + m_pos.y + m_pos.z);

        // Set up vertex data (and buffer(s)) and attribute pointers
        // Use below for starter code
        GLfloat vertices[] = {
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
            -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
            -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
            -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
             0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
            -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
             0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
            -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
        };

        // First, set the container's VAO (and VBO)
        glGenVertexArrays(1, &m_VAO);
        glGenBuffers(1, &m_VBO);

        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindVertexArray(m_VAO);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
        glEnableVertexAttribArray(0);

        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        
    }

    ~BreathingCube() {
        glDeleteBuffers(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
    }
    
    void render() const {

        // Make sure always to set the current shader before setting uniforms/drawing objects
        if (m_shader) { 
            m_shader->Use();

            // set cube's color
            GLint objectColorLoc = glGetUniformLocation(m_shader->Program, "objectColor");
            glUniform3f(objectColorLoc, m_color[0], m_color[1], m_color[2]);

            // compute model matrix
            glm::mat4 translatedCubeModelMat4 = glm::translate(glm::mat4(), glm::vec3(m_pos));
            glm::mat4 translatedScaledCubeModelMat4 = glm::scale(translatedCubeModelMat4, glm::vec3(m_current_breath_amt));
    
            GLint modelLoc = glGetUniformLocation(m_shader->Program, "model"); // Get the uniform locations
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(translatedScaledCubeModelMat4)); // Pass the transformed model matrix to the shader
        }

        // Draw the cube from its VAO
        //glBindVertexArray(containerVAO);
        glBindVertexArray(m_VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
    }

    void update(double time_since_last_update) {
        m_current_breath_amt = m_breathing_enabled ? 0.56 + 0.36 * sin(m_total_time + m_random_delta) + 0.001 : 0.95;

        // base the cube's color upon its current amount of breath
        // red == max inhale
        // blue == max exhale
        m_color[0] = 0.8 * m_current_breath_amt;
        m_color[1] = m_initial_color[1];
        m_color[2] = 1 - m_color[0];

        m_pos.z = m_initial_pos.z + (m_current_breath_amt * 1.28);

        m_rotated_angle_in_degrees = m_total_time;
        if (m_rotated_angle_in_degrees > 360.0) m_rotated_angle_in_degrees -= 360.0;

        m_total_time += time_since_last_update;
    }
    
protected:

    float m_rotated_angle_in_degrees;
    float m_current_breath_amt;
    glm::vec4 m_initial_color;
    glm::vec3 m_initial_pos;

    bool m_breathing_enabled;

    // Necessary only for starter code:
    GLuint m_VBO;
    GLuint m_VAO;
    GLfloat m_random_delta;
};
