class Entity;

//===========================================================
// Class : Light
//===========================================================

class Light : public Entity {

public:

    Light(Shader * shader, glm::vec3 initial_pos) {
        m_type = ET_LIGHT;
        m_shader = shader;
        m_initial_pos = initial_pos;
        m_pos = initial_pos;

        m_color = glm::vec4(1.0, 1.0, 1.0, 1.0);
    }

    ~Light() {
        
    }

    void render() const {
        if (m_shader) { 
            m_shader->Use();

            GLint lightColorLoc = glGetUniformLocation(m_shader->Program, "lightColor");
            GLint lightPosLoc   = glGetUniformLocation(m_shader->Program, "lightPos");

            // pass data to the shader's variables
            glUniform3f(lightColorLoc, m_color[0], m_color[1], m_color[2]);
            glUniform3f(lightPosLoc, m_pos.x, m_pos.y, m_pos.z);
        }
    }

    void update(double time_since_last_update) {

    }

protected:

    glm::vec3 m_initial_pos;

};