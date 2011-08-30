import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from ctypes import c_void_p, c_char, c_char_p, POINTER

class VertexBuffer:
    def __init__(self, data, format_description):
        '''
        Sample usage: geometry = VertexBuffer(data, size, [(3, GL_FLOAT), (4, GL_FLOAT)], GL_STATIC_DRAW)

        Uses Vertex Arrays Object (OpenGL 3.0) if possible. Vertex Buffer Objects were introduced in 1.5 (2003).
        '''
        pass

class OWOpenGLRenderer:
    '''OpenGL 3 deprecated a lot of old (1.x) functions, particulary, it removed
       immediate mode (glBegin, glEnd, glVertex paradigm). Vertex buffer objects and similar
       (through glDrawArrays for example) should be used instead. This class simplifies
       the usage of that functionality by providing methods which resemble immediate mode.'''
    def __init__(self):
        self._projection = QMatrix4x4()
        self._modelview = QMatrix4x4()

        vertex_shader_source = '''
            in vec3 position; // TODO: research qualifiers

            uniform mat4 projection;
            uniform mat4 modelview;

            void main(void)
            {
                gl_Position = projection * modelview * vec4(position, 1.);
            }
            '''

        fragment_shader_source = '''
            uniform vec4 color;

            void main(void)
            {
                gl_FragColor = color;
            }
            '''

        self.grid_shader = QtOpenGL.QGLShaderProgram()
        self.grid_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.grid_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.grid_shader.bindAttributeLocation('position', 0)

        if not self.grid_shader.link():
            print('Failed to link grid shader!')

    def set_transform(self, projection, modelview, viewport=None):
        self._projection = projection
        self._modelview = modelview
        if viewport:
            glViewport(*viewport)

    def draw_point(self, location):
        pass

    def draw_line(self, beginning, end):
        '''Draws a line. beginnig and end must be instances of QVector3D.'''
        # TODO
        glBegin(GL_LINES)
        glVertex3f(beginning.x(), beginning.y(), beginning.z())
        glVertex3f(end.x(), end.y(), end.z())
        glEnd()

    def draw_rectangle(self, vertex_min, vertex_max):
        if len(vertex_min) == 2:
            glBegin(GL_QUADS)
            glVertex2f(vertex_min[0], vertex_min[1])
            glVertex2f(vertex_max[0], vertex_min[1])
            glVertex2f(vertex_max[0], vertex_max[1])
            glVertex2f(vertex_min[0], vertex_max[1])
            glEnd()

    def draw_triangle(self, vertex0, vertex1, vertex2):
        # TODO
        if len(vertex0) == 2:
            glBegin(GL_TRIANGLES)
            glVertex2f(*vertex0)
            glVertex2f(*vertex1)
            glVertex2f(*vertex2)
            glEnd()
        else:
            glBegin(GL_TRIANGLES)
            glVertex3f(*vertex0)
            glVertex3f(*vertex1)
            glVertex3f(*vertex2)
            glEnd()
