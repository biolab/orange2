from ctypes import c_void_p, c_char, c_char_p, POINTER

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtOpenGL

import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.FULL_LOGGING = False
OpenGL.ERROR_ON_COPY = False
from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy

class VertexBuffer:
    def __init__(self, data, format_description, usage=GL_STATIC_DRAW):
        '''
        Sample usage: geometry = VertexBuffer(data, size, [(3, GL_FLOAT), (4, GL_FLOAT)], GL_STATIC_DRAW)

        Currently available attribute types: GL_FLOAT # TODO

        Uses Vertex Arrays Object (OpenGL 3.0) if possible. Vertex Buffer Objects were introduced in 1.5 (2003).
        '''
        self._format_description = format_description

        if glGenVertexArrays:
            self._vao = GLuint(42)
            glGenVertexArrays(1, self._vao)
            glBindVertexArray(self._vao)
            vertex_buffer_id = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
            glBufferData(GL_ARRAY_BUFFER, data, usage)

            vertex_size = sum(attribute[0]*4 for attribute in format_description)
            current_size = 0
            for i, (num_components, type) in enumerate(format_description):
                glVertexAttribPointer(i, num_components, type, GL_FALSE, vertex_size, c_void_p(current_size))
                glEnableVertexAttribArray(i)
                current_size += num_components*4

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        else:
            self._vbo_id = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_id)
            glBufferData(GL_ARRAY_BUFFER, data, usage)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, primitives, first, count):
        '''glDrawArrays'''
        if hasattr(self, '_vao'):
            glBindVertexArray(self._vao)
            glDrawArrays(primitives, first, count)
            glBindVertexArray(0)
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_id)

            vertex_size = sum(attribute[0]*4 for attribute in self._format_description)
            current_size = 0
            for i, (num_components, type) in enumerate(self._format_description):
                glVertexAttribPointer(i, num_components, type, GL_FALSE, vertex_size, c_void_p(current_size))
                glEnableVertexAttribArray(i)
                current_size += num_components*4

            glDrawArrays(primitives, first, count)
            for i in range(len(self._format_description)):
                glDisableVertexAttribArray(i)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

class OWOpenGLRenderer:
    '''OpenGL 3 deprecated a lot of old (1.x) functions, particulary, it removed
       immediate mode (glBegin, glEnd, glVertex paradigm). Vertex buffer objects and similar
       (through glDrawArrays for example) should be used instead. This class simplifies
       the usage of that functionality by providing methods which resemble immediate mode.'''
    def __init__(self):
        self._projection = QMatrix4x4()
        self._modelview = QMatrix4x4()

        ## Shader used to draw primitives. Position and color of vertices specified through uniforms. Nothing fancy.
        vertex_shader_source = '''
            in float index;
            varying vec4 color;

            uniform vec3 positions[6]; // 6 vertices for quad
            uniform vec4 colors[6];

            uniform mat4 projection;
            uniform mat4 modelview;

            void main(void)
            {
                int i = int(index);
                gl_Position = projection * modelview * vec4(positions[i], 1.);
                color = colors[i];
            }
            '''

        fragment_shader_source = '''
            in vec4 color;

            void main(void)
            {
                gl_FragColor = color;
            }
            '''

        self._shader = QtOpenGL.QGLShaderProgram()
        self._shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self._shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self._shader.bindAttributeLocation('index', 0)

        if not self._shader.link():
            print('Failed to link dummy renderer shader!')

        indices = numpy.array(range(6), dtype=numpy.float32) 
        self._vertex_buffer = VertexBuffer(indices, [(1, GL_FLOAT)])

    def set_transform(self, projection, modelview, viewport=None):
        self._projection = projection
        self._modelview = modelview
        if viewport:
            glViewport(*viewport)
        self._shader.bind()
        self._shader.setUniformValue('projection', projection)
        self._shader.setUniformValue('modelview', modelview)
        self._shader.release()

    def draw_line(self, position0, position1, color0=QColor(0, 0, 0), color1=QColor(0, 0 ,0), color=None):
        '''Draws a line. position0 and position1 must be instances of QVector3D. colors are specified with QColor'''

        if color:
            colors = [color.redF(), color.greenF(), color.blueF(), color.alphaF()] * 2
        else:
            colors = [color0.redF(), color0.greenF(), color0.blueF(), color0.alphaF(),
                      color1.redF(), color1.greenF(), color1.blueF(), color1.alphaF()]

        positions = [position0.x(), position0.y(), position0.z(),
                     position1.x(), position1.y(), position1.z()]

        self._shader.bind()
        glUniform4fv(glGetUniformLocation(self._shader.programId(), 'colors'), len(colors)/4, numpy.array(colors, numpy.float32))
        glUniform3fv(glGetUniformLocation(self._shader.programId(), 'positions'), len(positions)/3, numpy.array(positions, numpy.float32))
        self._vertex_buffer.draw(GL_LINES, 0, 2)
        self._shader.release()

    def draw_rectangle(self, position0, position1, position2, position3,
                       color0=QColor(0, 0, 0), color1=QColor(0, 0, 0), color2=QColor(0, 0, 0), color3=QColor(0, 0, 0), color=None):
        if color:
            colors = [color.redF(), color.greenF(), color.blueF(), color.alphaF()] * 6
        else:
            colors = [color0.redF(), color0.greenF(), color0.blueF(), color0.alphaF(),
                      color1.redF(), color1.greenF(), color1.blueF(), color1.alphaF(),
                      color3.redF(), color3.greenF(), color3.blueF(), color3.alphaF(),

                      color3.redF(), color3.greenF(), color3.blueF(), color3.alphaF(),
                      color1.redF(), color1.greenF(), color1.blueF(), color1.alphaF(),
                      color2.redF(), color2.greenF(), color2.blueF(), color2.alphaF()]

        positions = [position0.x(), position0.y(), position0.z(),
                     position1.x(), position1.y(), position1.z(),
                     position3.x(), position3.y(), position3.z(),

                     position3.x(), position3.y(), position3.z(),
                     position1.x(), position1.y(), position1.z(),
                     position2.x(), position2.y(), position2.z()]

        self._shader.bind()
        glUniform4fv(glGetUniformLocation(self._shader.programId(), 'colors'), len(colors)/4, numpy.array(colors, numpy.float32))
        glUniform3fv(glGetUniformLocation(self._shader.programId(), 'positions'), len(positions)/3, numpy.array(positions, numpy.float32))
        self._vertex_buffer.draw(GL_TRIANGLES, 0, 6)
        self._shader.release()

    def draw_triangle(self, position0, position1, position2,
                       color0=QColor(0, 0, 0), color1=QColor(0, 0, 0), color2=QColor(0, 0, 0), color=None):
        if color:
            colors = [color.redF(), color.greenF(), color.blueF(), color.alphaF()] * 3
        else:
            colors = [color0.redF(), color0.greenF(), color0.blueF(), color0.alphaF(),
                      color1.redF(), color1.greenF(), color1.blueF(), color1.alphaF(),
                      color2.redF(), color2.greenF(), color2.blueF(), color2.alphaF()]

        positions = [position0.x(), position0.y(), position0.z(),
                     position1.x(), position1.y(), position1.z(),
                     position2.x(), position2.y(), position2.z()]

        self._shader.bind()
        glUniform4fv(glGetUniformLocation(self._shader.programId(), 'colors'), len(colors)/4, numpy.array(colors, numpy.float32))
        glUniform3fv(glGetUniformLocation(self._shader.programId(), 'positions'), len(positions)/3, numpy.array(positions, numpy.float32))
        self._vertex_buffer.draw(GL_TRIANGLES, 0, 3)
        self._shader.release()
