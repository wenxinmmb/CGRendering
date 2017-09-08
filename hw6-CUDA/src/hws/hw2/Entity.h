// GLM Mathematics
#define GLM_FORCE_RADIANS // force everything in radians
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

//===========================================================
// Class : Entity
// Abstract base class for all entities in the world
//===========================================================

// We define these types in order to do easy introspection
// Introspection allows us to check what type of subclass we're
// looking at if we have some handle on an instance of a base
// class. E.g. we have a pointer to an Entity e, but we want to know
// whether it's actually a cube or a light. Checing e->m_type will
// tell us so long as the cube's and light's constructors each set
// themselves as their corresponding type.
typedef enum et {
    ET_NONE = 0,
    ET_CUBE,
    ET_LIGHT,
} EntityType;

class Entity {

public:

    Entity() {}
    virtual ~Entity() {}
    
    // remember: setting a C++ function equal to 0 means it's 
    // "pure abstract". 
    // Because Entity contains at least one pure abstract method, 
    // the entire class therefore becomes abstract. 
    // i.e. You cannot create an instance of  "Entity" - you 
    // must create a sub-class to instantiate it because it 
    // contains at least one pure abstract method.
    virtual void render() const = 0;
    virtual void update(double time_since_last_update) = 0;
    
    // member vars
    bool m_active;
    float m_total_time; // time elapsed since creation

    glm::vec3 m_pos;
    glm::vec4 m_color;
    
    EntityType m_type;

    Shader * m_shader;
    
};