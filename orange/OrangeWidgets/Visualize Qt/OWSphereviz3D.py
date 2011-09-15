'''
<name>Sphereviz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>2000</priority>
'''

import os
from math import pi
from random import choice

from plot.owplot3d import *
from plot.owopenglrenderer import VertexBuffer
from plot.primitives import parse_obj
from OWLinProjQt import *
from OWLinProj3DPlot import OWLinProj3DPlot

import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

class OWSphereviz3DPlot(OWLinProj3DPlot):
    def __init__(self, widget, parent=None, name='SpherevizPlot'):
        OWLinProj3DPlot.__init__(self, widget, parent, name)

        self.camera_angle = 90
        self.camera_type = 0 # Default, center, attribute

    def _build_anchor_grid(self):
        lines = []
        num_parts = 30
        anchors = numpy.array([a[:3] for a in self.anchor_data])
        for anchor in self.anchor_data:
            a0 = numpy.array(anchor[:3])
            neighbours = anchors.copy()
            neighbours = [(((n-a0)**2).sum(), n)  for n in neighbours]
            neighbours.sort(key=lambda e: e[0])
            for i in range(1, min(len(anchors), 4)): 
                difference = neighbours[i][1]-a0
                for j in range(num_parts):
                    lines.extend(normalize(a0 + difference*j/float(num_parts)))
                    lines.extend(normalize(a0 + difference*(j+1)/float(num_parts)))

        self._grid_buffer = VertexBuffer(numpy.array(lines, numpy.float32), [(3, GL_FLOAT)])

    def update_data(self, labels=None, setAnchors=0, **args):
        OWLinProj3DPlot.updateData(self, labels, setAnchors, **args)

        if self.anchor_data:
            self._build_anchor_grid()

        self.updateGL()

    updateData = update_data

    def setData(self, data, subsetData=None, **args):
        OWLinProj3DPlot.set_data(self, data, subsetData, **args)

        # No need to generate backgroud grid sphere geometry more than once
        if hasattr(self, '_sphere_buffer'):
            return

        self.makeCurrent()

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

        self._sphere_buffer = VertexBuffer(numpy.array(lines, numpy.float32), [(3, GL_FLOAT)])

        self._sphere_shader = QtOpenGL.QGLShaderProgram()
        self._sphere_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
            os.path.join(os.path.dirname(__file__), 'sphere.vs'))
        self._sphere_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
            os.path.join(os.path.dirname(__file__), 'sphere.fs'))

        self._sphere_shader.bindAttributeLocation('position', 0)

        if not self._sphere_shader.link():
            print('Failed to link sphere shader!')

        ## Cones
        cone_data = parse_obj('cone_hq.obj')
        vertices = []
        for v0, v1, v2, n0, n1, n2 in cone_data:
            vertices.extend([v0[0],v0[1],v0[2], n0[0],n0[1],n0[2],
                             v1[0],v1[1],v1[2], n1[0],n1[1],n1[2],
                             v2[0],v2[1],v2[2], n2[0],n2[1],n2[2]])

        self._cone_buffer = VertexBuffer(numpy.array(vertices, numpy.float32),
            [(3, GL_FLOAT),
             (3, GL_FLOAT)])

        self._cone_shader = QtOpenGL.QGLShaderProgram()
        self._cone_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
            os.path.join(os.path.dirname(__file__), 'cone.vs'))
        self._cone_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
            os.path.join(os.path.dirname(__file__), 'cone.fs'))

        self._cone_shader.bindAttributeLocation('position', 0)
        self._cone_shader.bindAttributeLocation('normal', 1)

        if not self._cone_shader.link():
            print('Failed to link cone shader!')

        ## Another dummy shader (anchor grid)
        self._grid_shader = QtOpenGL.QGLShaderProgram()
        self._grid_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex,
            os.path.join(os.path.dirname(__file__), 'grid.vs'))
        self._grid_shader.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment,
            os.path.join(os.path.dirname(__file__), 'grid.fs'))

        self._grid_shader.bindAttributeLocation('position', 0)

        if not self._grid_shader.link():
            print('Failed to link grid shader!')

        self.before_draw_callback = self.before_draw

    def update_camera_type(self):
        if self.camera_type == 2:
            self._random_anchor = choice(self.anchor_data)
        self.update()

    def before_draw(self):
        view = QMatrix4x4()
        if self.camera_type == 2:
            view.lookAt(
                QVector3D(self._random_anchor[0], self._random_anchor[1], self._random_anchor[2]),
                self.camera,
                QVector3D(0, 1, 0))
            projection = QMatrix4x4()
            projection.perspective(self.camera_angle, float(self.width()) / self.height(),
                                   0.01, 5.)
            self.projection = projection
        elif self.camera_type == 1:
            view.lookAt(
                QVector3D(0, 0, 0),
                self.camera * self.camera_distance,
                QVector3D(0, 1, 0))
            projection = QMatrix4x4()
            projection.perspective(self.camera_angle, float(self.width()) / self.height(),
                                   0.01, 5.)
            self.projection = projection
        else:
            view.lookAt(
                self.camera * self.camera_distance,
                QVector3D(0, 0, 0),
                QVector3D(0, 1, 0))
        self.view = view

        self._draw_sphere()

        # Qt text rendering classes still use deprecated OpenGL functionality
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.view.data(), dtype=float))
        glMultMatrixd(numpy.array(self.model.data(), dtype=float))

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

                model = QMatrix4x4()
                model.translate(x, y, z)
                model = model * rotation
                model.rotate(-90, 1, 0, 0)
                model.translate(0, -0.05, 0)
                model.scale(0.02, 0.02, 0.02)

                self._cone_shader.bind()
                self._cone_shader.setUniformValue('projection', self.projection)
                self._cone_shader.setUniformValue('modelview', self.view * model)
                self._cone_buffer.draw()
                self._cone_shader.release()

                self.qglColor(self._theme.axis_values_color)
                self.renderText(x*1.2, y*1.2, z*1.2, label)

            if self.anchor_data and not hasattr(self, '_grid_buffer'):
                self._build_anchor_grid()

            # Draw grid between anchors
            self._grid_shader.bind()
            self._grid_shader.setUniformValue('projection', self.projection)
            self._grid_shader.setUniformValue('modelview', self.view * self.model)
            self._grid_shader.setUniformValue('color', self._theme.axis_color)
            self._grid_buffer.draw(GL_LINES)
            self._grid_shader.release()

        self._draw_value_lines()

    def _draw_sphere(self):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._sphere_shader.bind()
        self._sphere_shader.setUniformValue('projection', self.projection)
        self._sphere_shader.setUniformValue('modelview', self.view * self.model)
        self._sphere_shader.setUniformValue('cam_position', self.camera * self.camera_distance)
        self._sphere_shader.setUniformValue('use_transparency', self.camera_type == 0)
        self._sphere_buffer.draw(GL_LINES)
        self._sphere_shader.release()

    def mouseMoveEvent(self, event):
        self.invert_mouse_x = self.camera_type != 0
        OWLinProj3DPlot.mouseMoveEvent(self, event)

class OWSphereviz3D(OWLinProjQt):
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
