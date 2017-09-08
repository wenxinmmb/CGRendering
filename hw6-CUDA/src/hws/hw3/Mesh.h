class Entity;

#include <string>
#include <vector>
#include <set>
#include <map>

// The Mesh class represents a geometric shape and 
// uses an list of vertices and a list of faces 
// which connect them.
class Mesh : public Entity
{

public:

    // Vertex struct represents positions and normals
    // for each Vertex across the mesh.
    typedef struct {
        GLfloat x, y, z;
        GLfloat nx, ny, nz;
    }
    Vertex;

    // Type used for indices into the vertex array
    typedef size_t Index;

    // An inner class which manages each face
    class Face
    {
    public:

        // Construct a face from three vertex indices
        Face(Index i0, Index i1, Index i2) {
            mIndices[0] = i0;
            mIndices[1] = i1;
            mIndices[2] = i2;
        }

        // Get an index from this face.
        inline Index getIndex(int which) const { return mIndices[which]; }

        // Reverse the order of indices in this face. This changes
        // right-handed winding to left-handed and vice versa.
        void ReverseWinding() {
            std::swap(mIndices[1], mIndices[2]);
        }

    private:
        // Index array storage.
        Index mIndices[3];
    };

    typedef std::vector<Face> FaceArray;

    // Meshes must be constructed by passing in the name of an .obj file.
    // That .obj file must be of a particular format. This Mesh work for only
    // a small subset of .obj file formats. See loadOBJ method for more info
    // about the specification of the .obj format this Mesh will accept.
    Mesh(const std::string& filename, Shader * shader, glm::vec3 initial_pos) {
        m_type = ET_MESH;

        m_total_time = 0.0;
        m_shader = shader;
        m_pos = initial_pos;

        m_color = glm::vec4(0.7, 0.25, 0.25, 1.0);

        if (loadOBJ(filename) == -1) {
            printf("%s", "Mesh.h: ERROR: failed to load OBJ file into Mesh data structure.\n");
            exit(1);
        }
    }

    ~Mesh() {
        glDeleteBuffers(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);
    }

    void refresh() {
        reorderVerticesForDrawing();
        recreateGPUDataStructure();
    }

    void recreateGPUDataStructure() {

        // Delete previous GPU data
        glDeleteBuffers(1, &m_VAO);
        glDeleteBuffers(1, &m_VBO);

        // Set the container VAO
        glGenVertexArrays(1, &m_VAO);
        glBindVertexArray(m_VAO);

        // Create the VBO
        glGenBuffers(1, &m_VBO);
        // Place mesh data into the VBO on the GPU
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        glBufferData(GL_ARRAY_BUFFER, m_vertices_to_draw.size() * sizeof(Vertex), &m_vertices_to_draw.front(), GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid *)0);
        glEnableVertexAttribArray(0);

        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid *)(3 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);

        // Reset vertex array to nothing
        glBindVertexArray(0);
    }
    
    void render() const {

        // Make sure always to set the current shader before setting uniforms/drawing objects
        if (m_shader) { 
            m_shader->Use();

            // set mesh's color
            GLint objectColorLoc = glGetUniformLocation(m_shader->Program, "objectColor");
            glUniform3f(objectColorLoc, m_color[0], m_color[1], m_color[2]);

            GLint modelLoc = glGetUniformLocation(m_shader->Program, "model"); // Get the uniform locations
            glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(5.0));
            glm::mat4 translate = glm::translate(scale, m_pos);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(translate)); // Pass the transformed model matrix to the shader
        }

        // Draw the mesh from its VAO
        glBindVertexArray(m_VAO);
        glDrawArrays(GL_TRIANGLES, 0, getNumVertices() * (sizeof(Vertex)/sizeof(float)));

        // Reset state
        glBindVertexArray(0);
    }

    void update(double time_since_last_update) {
        // nothing to update per frame :(
    }

    typedef std::vector<Vertex> VertexArray;
    typedef std::set<Index> VertexIndexSet;

    // Vertex helpers
    inline size_t getNumVertices() const { return m_vertices.size(); } // Return the number of vertices in the mesh
    inline Vertex getVertex(size_t idx) const { return m_vertices[idx]; } // Read a vertex of this shape by index

    // Write a vertex to this shape by index
    inline void setVertex(size_t idx, const Vertex& vertex) { m_vertices[idx] = vertex; }
    // Add a new vertex to the end of the vertex array.
    // Returns the index of the new vertex, which will
    // always be equal to the previous number of vertices.
    size_t addVertex(const Vertex & vertex) {
        size_t result = getNumVertices();
        m_vertices.push_back(vertex);
        return result;
    }

    // Face helpers
    inline size_t getNumFaces() const { return m_faces.size(); } // Get the number of faces in this shape
    inline Face getFace(size_t idx) const { return m_faces[idx]; } // Read a face of this shape by index
    inline void setFace(size_t idx, const Face& face) { m_faces[idx] = face; } // Write a face of this shape by index
    void addFace(const Face& face) { 
        m_faces.push_back(face); 
    }

    // Helper function to deal with the particular way
    // that indices must be interpreted in an .obj file
    static int OBJIndexing(int input, size_t numValues) {
        if (input > 0) return input - 1;
        return numValues + input;
    }

    // Orders the vertcies into a vector for drawing
    // the mesh out of GL_TRIANGLES
    void reorderVerticesForDrawing() {
        m_vertices_to_draw.clear();
        size_t numFaces = getNumFaces();
        for (size_t ii = 0; ii < numFaces; ++ii) {
            const Face& face = getFace(ii);
            for (size_t jj = 0; jj < 3; ++jj) {
                Index index = face.getIndex(jj);
                const Vertex v = m_vertices[index];
                m_vertices_to_draw.push_back(v);
            }
        }
    }

    // debug helper
    void printDataStructure(bool printPositions, bool printNormals, bool printFaces) {
        for (size_t ii = 0; ii < getNumVertices(); ++ii) {
            if (printPositions) printf("Vertex at: %f, %f, %f\n", m_vertices[ii].x, m_vertices[ii].y, m_vertices[ii].z);
            if (printNormals) printf("Vertex normal: %f, %f, %f\n", m_vertices[ii].nx, m_vertices[ii].ny, m_vertices[ii].nz);
        }
        
        if (printFaces) {
            for (size_t ii = 0; ii < getNumFaces(); ++ii) {
                printf("Face with indices: %u, %u, %u\n", 
                    (unsigned int)m_faces[ii].getIndex(0), 
                    (unsigned int)m_faces[ii].getIndex(1), 
                    (unsigned int)m_faces[ii].getIndex(2));
            }
        }
    }
    
protected:

    // Assignment Task 2:
    
    // You'll need to create an additional data structure and
    // corresponding algorithms to enable lookup of neighboring
    // vertices for any given vertex. This lookup needs to happen
    // as fast as possible, so think hard about which data structure
    // makes the most sense for this task.
    // Hint: some of your algorithm may need to alter some of the above, 
    // public methods. You are also free to design and implement 
    // your own protected methods here.

    float m_total_time;

    GLuint m_VBO;
    GLuint m_VAO;

    VertexArray m_vertices;         // The mesh of this shape as loaded from disk
    VertexArray m_vertices_to_draw; // The mesh ordered for drawing as triangular faces
    FaceArray m_faces;              // The faces of this mesh

    // Compute or re-compute the per-vertex normals for this shape
    // using the normals of adjoining faces.
    // We compute the normals for the mesh as the area-weighted average of
    // the normals of incident faces. This is a simple technique to 
    // implement, but more advanced techniques are possible as well.
    void generateNormals() {

        // Initialize all the normals to zero
        size_t numVertices = getNumVertices();
        for (size_t ii = 0; ii < numVertices; ++ii) {
            m_vertices[ii].nx = 0;
            m_vertices[ii].ny = 0;
            m_vertices[ii].nz = 0;
        }

        // Assignment Task 1:

        // Loop over faces, adding the normal of each face
        // to the vertices that use it.

        // For each face:

            // We compute a cross-product of two triangle edges.
            // This direction of this vector will be the normal
            // direction, while the magnitude will be twice
            // the triangle area. We can therefore use the result 
            // as a weighted normal.


            // We now add the face normal to the normal stored
            // in each of the vertices using the face

        // After that, each vertex now should have an area-weighted normal. 
        // We need to normalize them to turn them into correct unit-length
        // normal vectors for rendering. Do that here.
    }

    // Loads a .obj file into the mesh's data structure
    // Returns -1 on failure, 0 on success
    int loadOBJ(const std::string& filename) {

        static const int kMaxLineLen = 256;

        // The subset of the OBJ format that we handle has
        // the following commands:
        //
        // v    <x> <y> <z>         Define a vertex position.
        // f    <i1> <i2> <i3>      Define a face from previous vertex indices
        //
        // Every face in an OBJ file refers to previously-defines
        // positions by index.
        //

        // Open the file
        FILE * file = fopen(filename.c_str(), "r");
        if (!file) {
            fprintf(stderr,
                    "Mesh::loadOBJ() - Could not open shape file '%s'.\n", filename.c_str());
            return -1;
        }

        char lineBuffer[kMaxLineLen];
        int lineIndex = 0; // for printing error messages

        // Vector to collect the position of each vertex
        std::vector<Vertex> positions;

        // Map to point us to previously-created vertices. This maps
        // position indices to a single index in the new mesh.
        typedef std::map<int, size_t> IndexMap;
        IndexMap indexMap;

        // Read the file line-by-line
        while (fgets(lineBuffer, kMaxLineLen, file)) {
            ++lineIndex;
            char * str = strtok(lineBuffer, " \t\n\r");

            // Skip empty or comment lines
            if (!str || str[0] == '\0' || str[0] == '#') {
                continue;
            }
            if (str[0] == 'g' || str[0] == 's' || strcmp(str, "usemtl") == 0 || strcmp(str, "mtllib") == 0) {
                continue;
            }

            // Process other lines based on their commands
            if (strcmp(str, "v") == 0) {
                // It's a vertex position line, so we
                // read the position data into the Vertex's (x, y, z)
                str = strtok(NULL, "");
                Vertex position;
                sscanf(str, "%f %f %f\n", &position.x, &position.y, &position.z);
                positions.push_back(position);
            }
            else if (strcmp(str, "f") == 0) {
                // It's a face line.
                // Each vertex in the face will be defined by
                // the indices of its position.

                std::vector<Index> faceIndices;

                // Read each vertex entry.
                int curIndex = 0;
                Index indices[3];

                int positionIdx;

                while ((str = strtok(NULL, " \t\n\r")) != NULL) {

                    sscanf(str, "%d", &positionIdx);

                    // We look to see if we have already created a vertex
                    // based on this position, and reuse it
                    // if possible. Otherwise we add a new vertex.
                    positionIdx = OBJIndexing(positionIdx, positions.size());
                    size_t newIndex;
                    IndexMap::const_iterator ii = indexMap.find(positionIdx);
                    if (ii != indexMap.end()) {
                        newIndex = ii->second;
                    } else {
                        // We didn't find an existing vertex, 
                        // so we create a new one

                        Vertex position = positions[positionIdx];

                        Vertex newVertex;
                        newVertex.x = position.x;
                        newVertex.y = position.y;
                        newVertex.z = position.z;

                        newIndex = addVertex(newVertex);
                        indexMap[positionIdx] = newIndex;
                    }

                    indices[curIndex++] = newIndex;

                    // Keep fanning the triangle
                    if (curIndex == 3) {
                      addFace(Face(indices[0], indices[1], indices[2]));
                      indices[1] = indices[2];
                      curIndex = 2;
                    }
                }
            } else {
                // Unknown line in obj file - ignore and print a warning.
                fprintf(stderr, "Mesh::loadOBJ() - "
                        "Unable to parse line %d: %s (continuing)\n",
                        lineIndex, lineBuffer);
            }
        }

        fclose(file);

        // The .obj file didn't already have normals, so we generate them
        // using an algorithm which intuits what they could be
        generateNormals();

        refresh();

        // Success!
        return 0;
    }

};
