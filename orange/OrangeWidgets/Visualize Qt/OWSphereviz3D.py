'''
<name>Sphereviz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>2000</priority>
'''

from plot.owplot3d import *
from plot.primitives import parse_obj
from OWLinProjQt import *
from OWLinProj3DPlot import OWLinProj3DPlot

import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

class OWSphereviz3DPlot(OWLinProj3DPlot):
    def __init__(self, widget, parent=None, name='SpherevizPlot'):
        OWLinProj3DPlot.__init__(self, widget, parent, name)

    def _build_anchor_grid(self):
        lines = []
        num_parts = 30
        anchors = array([a[:3] for a in self.anchor_data])
        for anchor in self.anchor_data:
            a0 = array(anchor[:3])
            neighbours = anchors.copy()
            neighbours = [(((n-a0)**2).sum(), n)  for n in neighbours]
            neighbours.sort(key=lambda e: e[0])
            for i in range(1, min(len(anchors), 4)): 
                difference = neighbours[i][1]-a0
                for j in range(num_parts):
                    lines.extend(normalize(a0 + difference*j/float(num_parts)))
                    lines.extend(normalize(a0 + difference*(j+1)/float(num_parts)))

        self.grid_vao_id = GLuint(0)
        glGenVertexArrays(1, self.grid_vao_id)
        glBindVertexArray(self.grid_vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(lines, numpy.float32), GL_STATIC_DRAW)

        vertex_size = 3*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.grid_vao_id.num_vertices = len(lines) / 3

    def updateData(self, labels=None, setAnchors=0, **args):
        OWLinProj3DPlot.updateData(self, labels, setAnchors, **args)

        if self.anchor_data:
            self._build_anchor_grid()

        self.updateGL()

    def setData(self, data, subsetData=None, **args):
        OWLinProj3DPlot.setData(self, data, subsetData, **args)

        # No need to generate backgroud grid sphere geometry more than once
        if hasattr(self, 'sphere_vao_id'):
            return

        lines = []
        num_parts = 30
        num_horizontal_rings = 20
        num_vertical_rings = 24
        r = 1.

        for i in range(num_horizontal_rings):
            z_offset = float(i) * 2 / num_horizontal_rings - 1.
            r = (1. - z_offset**2)**0.5
            for j in range(num_parts):
                angle_z_0 = float(j) * 2 * pi / num_parts
                angle_z_1 = float(j+1) * 2 * pi / num_parts
                lines.extend([sin(angle_z_0)*r, z_offset, cos(angle_z_0)*r,
                              sin(angle_z_1)*r, z_offset, cos(angle_z_1)*r])

        for i in range(num_vertical_rings):
            r = 1.
            phi = 2 * i * pi / num_vertical_rings
            for j in range(num_parts):
                theta_0 = (j) * pi / num_parts
                theta_1 = (j+1) * pi / num_parts
                lines.extend([sin(theta_0)*cos(phi)*r, cos(theta_0)*r, sin(theta_0)*sin(phi)*r,
                              sin(theta_1)*cos(phi)*r, cos(theta_1)*r, sin(theta_1)*sin(phi)*r])

        self.sphere_vao_id = GLuint(0)
        glGenVertexArrays(1, self.sphere_vao_id)
        glBindVertexArray(self.sphere_vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(lines, numpy.float32), GL_STATIC_DRAW)

        vertex_size = 3*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.sphere_vao_id.num_vertices = len(lines) / 3

        vertex_shader_source = '''#version 150
            in vec3 position;
            out float transparency;

            uniform mat4 projection;
            uniform mat4 modelview;
            uniform vec3 cam_position;

            void main(void)
            {
                transparency = clamp(dot(normalize(cam_position-position), normalize(position)), 0., 1.);
                gl_Position = projection * modelview * vec4(position, 1.);
            }
            '''

        fragment_shader_source = '''#version 150
            in float transparency;
            uniform bool invert_transparency;

            void main(void)
            {
                gl_FragColor = vec4(0.5, 0.5, 0.5, 1. - transparency - 0.6);
                if (invert_transparency)
                    gl_FragColor.a = transparency;
            }
            '''

        self.sphere_shader = QtOpenGL.QGLShaderProgram()
        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.sphere_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.sphere_shader.bindAttributeLocation('position', 0)

        if not self.sphere_shader.link():
            print('Failed to link sphere shader!')

        # Another dummy shader (anchor grid)
        vertex_shader_source = '''#version 150
            in vec3 position;

            uniform mat4 projection;
            uniform mat4 modelview;

            void main(void)
            {
                gl_Position = projection * modelview * vec4(position, 1.);
            }
            '''

        fragment_shader_source = '''#version 150
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

        self.before_draw_callback = lambda: self.before_draw()

    def before_draw(self):
        # Override modelview (scatterplot points camera somewhat below the center, which doesn't
        # look good with sphere)
        modelview = QMatrix4x4()
        if self.camera_in_center:
            modelview.lookAt(
                QVector3D(0, 0, 0),
                QVector3D(self.camera[0]*self.camera_distance,
                          self.camera[1]*self.camera_distance,
                          self.camera[2]*self.camera_distance),
                QVector3D(0, 1, 0))
            projection = QMatrix4x4()
            projection.perspective(90., float(self.width()) / self.height(),
                                   self.perspective_near, self.perspective_far)
            self.projection = projection
        else:
            modelview.lookAt(
                QVector3D(self.camera[0]*self.camera_distance,
                          self.camera[1]*self.camera_distance,
                          self.camera[2]*self.camera_distance),
                QVector3D(0, 0, 0),
                QVector3D(0, 1, 0))
        self.modelview = modelview

        self.draw_sphere()

        # Qt text rendering classes still use deprecated OpenGL functionality
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.modelview.data(), dtype=float))

        if self.showAnchors:
            for anchor in self.anchor_data:
                x, y, z, label = anchor

                direction = QVector3D(x, y, z)
                up = QVector3D(0, 1, 0)
                right = QVector3D.crossProduct(direction, up).normalized()
                up = QVector3D.crossProduct(right, direction)
                rotation = QMatrix4x4()
                rotation.setColumn(0, QVector4D(right, 0))
                rotation.setColumn(1, QVector4D(up, 0))
                rotation.setColumn(2, QVector4D(direction, 0))

                self.cone_shader.bind()
                self.cone_shader.setUniformValue('projection', self.projection)
                modelview = QMatrix4x4(self.modelview)
                modelview.translate(x, y, z)
                modelview = modelview * rotation
                modelview.rotate(-90, 1, 0, 0)
                modelview.translate(0, -0.02, 0)
                modelview.scale(-0.02, -0.02, -0.02)
                self.cone_shader.setUniformValue('modelview', modelview)

                glBindVertexArray(self.cone_vao_id)
                glDrawArrays(GL_TRIANGLES, 0, self.cone_vao_id.num_vertices)
                glBindVertexArray(0)

                self.cone_shader.release()

                self.qglColor(self._theme.axis_values_color)
                self.renderText(x*1.2, y*1.2, z*1.2, label)

            if self.anchor_data and not hasattr(self, 'grid_vao_id'):
                self._build_anchor_grid()

            # Draw grid between anchors
            self.grid_shader.bind()
            self.grid_shader.setUniformValue('projection', self.projection)
            self.grid_shader.setUniformValue('modelview', self.modelview)
            self.grid_shader.setUniformValue('color', self._theme.axis_color)
            glBindVertexArray(self.grid_vao_id)

            glDrawArrays(GL_LINES, 0, self.grid_vao_id.num_vertices)

            glBindVertexArray(0)
            self.grid_shader.release()

        self._draw_value_lines()

    def draw_sphere(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.sphere_shader.bind()
        self.sphere_shader.setUniformValue('projection', self.projection)
        self.sphere_shader.setUniformValue('modelview', self.modelview)
        self.sphere_shader.setUniformValue('cam_position', QVector3D(*self.camera)*self.camera_distance)
        self.sphere_shader.setUniformValue('invert_transparency', self.camera_in_center)
        glBindVertexArray(self.sphere_vao_id)

        glDrawArrays(GL_LINES, 0, self.sphere_vao_id.num_vertices)

        glBindVertexArray(0)
        self.sphere_shader.release()

    def mouseMoveEvent(self, event):
        self.invert_mouse_x = self.camera_in_center
        OWLinProj3DPlot.mouseMoveEvent(self, event)

class OWSphereviz3D(OWLinProjQt):
    settingsList = ['showAllAttributes']

    def __init__(self, parent=None, signalManager=None):
        OWLinProjQt.__init__(self, parent, signalManager, "Sphereviz 3D", graphClass=OWSphereviz3DPlot)

        self.inputs = [("Examples", ExampleTable, self.setData, Default),
                       ("Example Subset", ExampleTable, self.setSubsetData),
                       ("Attribute Selection List", AttributeList, self.setShownAttributes),
                       ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults),
                       ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable),
                        ("Unselected Examples", ExampleTable),
                        ("Attribute Selection List", AttributeList)]
        self.resize(1000, 600)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viz = OWSphereviz3D()
    viz.show()
    data = orange.ExampleTable('../../doc/datasets/iris')
    viz.setData(data)
    viz.handleNewSignals()
    app.exec_()
