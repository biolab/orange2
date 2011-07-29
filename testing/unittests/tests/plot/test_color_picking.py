import unittest
import struct
from ctypes import c_void_p

import numpy
from PyQt4 import QtOpenGL
from PyQt4.QtGui import *

from OpenGL.GL import *
from OpenGL.GL.ARB.vertex_array_object import *
from OpenGL.GL.ARB.vertex_buffer_object import *

class TestWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        QtOpenGL.QGLWidget.__init__(self, QtOpenGL.QGLFormat(), parent)
        self.supported_hardware = True
        self.info = '\n'
        self.resize(500, 500)

    def initializeGL(self):
        vertex_shader_source = '''
            #extension GL_EXT_gpu_shader4 : enable

            attribute vec4 position;

            varying vec4 var_color;

            void main(void) {
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(position.xyz, 1.);
                gl_PointSize = 2.;

                // We've packed example index into .w component of this vertex,
                // to output it to the screen, it has to be broken down into RGBA.
                uint index = uint(position.w);
                var_color = vec4(float((index & 0xFF)) / 255.,
                                        float((index & 0xFF00) >> 8) / 255.,
                                        float((index & 0xFF0000) >> 16) / 255.,
                                        float((index & 0xFF000000) >> 24) / 255.);
            }
            '''

        fragment_shader_source = '''
            varying vec4 var_color;

            void main(void) {
                gl_FragColor = var_color;
            }
            '''

        self.shader = QtOpenGL.QGLShaderProgram()
        self.shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.shader.bindAttributeLocation('position', 0)

        if not self.shader.link():
            self.supported_hardware = False
            self.info += '\nShader doesnt work: ' + self.shader.log()

        format = QtOpenGL.QGLFramebufferObjectFormat()
        format.setAttachment(QtOpenGL.QGLFramebufferObject.CombinedDepthStencil)
        self.picking_fbo = QtOpenGL.QGLFramebufferObject(1024, 1024, format)
        if not self.picking_fbo.isValid():
            self.supported_hardware = False
            self.info += '\nFBO is not supported!'

        # (x,y,z, index)
        vertices = [[i*5+1, i*5+1, 0, 1000*1000+i] for i in range(100)]

        self.vao_id = GLuint(0)
        glGenVertexArrays(1, self.vao_id)
        glBindVertexArray(self.vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(vertices, 'f'), GL_STATIC_DRAW)

        vertex_size = 4*4
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def paintGL(self):
        glDisable(GL_CULL_FACE)
        glViewport(0, 0, self.width(), self.height())
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width(), self.height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.picking_fbo.bind()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        self.shader.bind()

        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        glBindVertexArray(self.vao_id)
        glDrawArrays(GL_POINTS, 0, 100)
        glBindVertexArray(0)

        self.shader.release()

        # Draw stencil mask.
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        glStencilMask(0x01)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT)
        glStencilFunc(GL_ALWAYS, 0, ~0)
        glEnable(GL_STENCIL_TEST)

        glBegin(GL_QUADS)
        glVertex2f(100, 100)
        glVertex2f(150, 100)
        glVertex2f(150, 150)
        glVertex2f(100, 150)
        glEnd()

        glDisable(GL_STENCIL_TEST)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)
        self.picking_fbo.release()

    def get_selected_indices(self):
        width, height = self.width(), self.height()
        self.picking_fbo.bind()
        color_pixels = glReadPixels(0, 0,
                                    width, height,
                                    GL_RGBA,
                                    GL_UNSIGNED_BYTE)
        stencil_pixels = glReadPixels(0, 0,
                                      width, height,
                                      GL_STENCIL_INDEX,
                                      GL_FLOAT)
        self.picking_fbo.release()
        stencils = struct.unpack('f'*width*height, stencil_pixels)
        colors = struct.unpack('I'*width*height, color_pixels)
        indices = set([])
        for stencil, color in zip(stencils, colors):
            if stencil > 0. and color < 4294967295:
                indices.add(color)

        return indices

class TestPicking(unittest.TestCase):
    '''
    Tests unique-color-encoding picking algorithm.
    '''
    def setUp(self):
        self.app = QApplication([])
        self.test_widget = TestWidget()
        self.test_widget.updateGL()

    def tearDown(self):
        del self.test_widget

    def test_picking(self):
        if self.test_widget.supported_hardware:
            indices = self.test_widget.get_selected_indices()
            self.assertEqual(indices, set(range(1000*1000+20, 1000*1000+30)))
        else:
            # TODO: what to do on unsupported hardware?
            print('Test ran on unsupported hardware: ' + self.test_widget.info)

if __name__ == '__main__':
    unittest.main()
