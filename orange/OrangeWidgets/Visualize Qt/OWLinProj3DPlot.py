from plot.owplot3d import *
from plot.owplotgui import *
from plot.primitives import parse_obj
from plot import OWPoint

from Orange.preprocess.scaling import ScaleLinProjData3D, get_variable_values_sorted
import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

class OWLinProj3DPlot(OWPlot3D, ScaleLinProjData3D):
    def __init__(self, widget, parent=None, name='None'):
        self.name = name
        OWPlot3D.__init__(self, parent)
        ScaleLinProjData3D.__init__(self)

        self.camera_fov = 22.
        self.camera_in_center = False
        self.show_axes = self.show_chassis = self.show_grid = False

        self.point_width = 6
        self.animate_plot = False
        self.animate_points = False
        self.antialias_plot = False
        self.antialias_points = False
        self.antialias_lines = False
        self.auto_adjust_performance = False
        self.show_filled_symbols = True
        self.use_antialiasing = True
        self.sendSelectionOnUpdate = False
        self.setCanvasBackground = self.setCanvasColor

        self._point_width_to_symbol_scale = 1.5

        if 'linear' in self.name.lower():
            self._arrow_lines = []
            self.mouseover_callback = self._update_arrow_values

        self.gui = OWPlotGUI(self)

    def setData(self, data, subsetData=None, **args):
        if data == None:
            return
        ScaleLinProjData3D.setData(self, data, subsetData, **args)
        OWPlot3D.initializeGL(self)

        if hasattr(self, 'cone_vao_id'):
            return

        self.makeCurrent()
        self.state = PlotState.IDLE # Override for now, apparently this is modified by OWPlotGUI
        self.before_draw_callback = lambda: self.before_draw()

        cone_data = parse_obj('cone_hq.obj')
        vertices = []
        for v0, v1, v2, n0, n1, n2 in cone_data:
            vertices.extend([v0[0],v0[1],v0[2], n0[0],n0[1],n0[2],
                             v1[0],v1[1],v1[2], n1[0],n1[1],n1[2],
                             v2[0],v2[1],v2[2], n2[0],n2[1],n2[2]])

        self.cone_vao_id = GLuint(0)
        glGenVertexArrays(1, self.cone_vao_id)
        glBindVertexArray(self.cone_vao_id)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(vertices, 'f'), GL_STATIC_DRAW)

        vertex_size = (3+3)*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(3*4))
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.cone_vao_id.num_vertices = len(vertices) / (vertex_size / 4)

        vertex_shader_source = '''
            in vec3 position;
            in vec3 normal;

            varying vec4 color;

            uniform mat4 projection;
            uniform mat4 modelview;

            const vec3 light_direction = normalize(vec3(-0.7, 0.42, 0.21));

            void main(void)
            {
                gl_Position = projection * modelview * vec4(position, 1.);
                float diffuse = clamp(dot(light_direction, normalize((modelview * vec4(normal, 0.)).xyz)), 0., 1.);
                color = vec4(vec3(1., 1., 1.) * diffuse + 0.1, 1.);
            }
            '''

        fragment_shader_source = '''
            varying vec4 color;

            void main(void)
            {
                gl_FragColor = color;
            }
            '''

        self.cone_shader = QtOpenGL.QGLShaderProgram()
        self.cone_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.cone_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.cone_shader.bindAttributeLocation('position', 0)
        self.cone_shader.bindAttributeLocation('normal', 1)

        if not self.cone_shader.link():
            print('Failed to link cone shader!')

        ## Value lines shader
        vertex_shader_source = '''
            in vec3 position;
            in vec3 color;
            in vec3 normal;

            varying vec4 var_color;

            uniform mat4 projection;
            uniform mat4 modelview;
            uniform float value_line_length;
            uniform vec3 plot_scale;

            void main(void)
            {
                gl_Position = projection * modelview * vec4(position*plot_scale + normal*value_line_length, 1.);
                var_color = vec4(color, 1.);
            }
            '''

        fragment_shader_source = '''
            varying vec4 var_color;

            void main(void)
            {
                gl_FragColor = var_color;
            }
            '''

        self.value_lines_shader = QtOpenGL.QGLShaderProgram()
        self.value_lines_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, vertex_shader_source)
        self.value_lines_shader.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, fragment_shader_source)

        self.value_lines_shader.bindAttributeLocation('position', 0)
        self.value_lines_shader.bindAttributeLocation('color', 1)
        self.value_lines_shader.bindAttributeLocation('normal', 2)

        if not self.value_lines_shader.link():
            print('Failed to link value-lines shader!')

    set_data = setData

    def before_draw(self):
        modelview = QMatrix4x4()
        modelview.lookAt(
            QVector3D(self.camera[0]*self.camera_distance,
                      self.camera[1]*self.camera_distance,
                      self.camera[2]*self.camera_distance),
            QVector3D(0, 0, 0),
            QVector3D(0, 1, 0))
        self.modelview = modelview

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.modelview.data(), dtype=float))

        self.renderer.set_transform(self.projection, self.modelview)

        if self.showAnchors:
            for anchor in self.anchor_data:
                x, y, z, label = anchor

                direction = QVector3D(x, y, z).normalized()
                up = QVector3D(0, 1, 0)
                right = QVector3D.crossProduct(direction, up).normalized()
                up = QVector3D.crossProduct(right, direction).normalized()
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

                glDepthMask(GL_TRUE)
                glBindVertexArray(self.cone_vao_id)
                glDrawArrays(GL_TRIANGLES, 0, self.cone_vao_id.num_vertices)
                glBindVertexArray(0)

                self.cone_shader.release()

                glDepthMask(GL_FALSE)
                self.qglColor(self._theme.axis_values_color)
                self.renderText(x*1.2, y*1.2, z*1.2, label)

                self.renderer.draw_line(
                    QVector3D(0, 0, 0),
                    QVector3D(x, y, z),
                    color=self._theme.axis_color)

        glDepthMask(GL_TRUE)

        if self.tooltipKind == 0:
            glEnable(GL_DEPTH_TEST)
            if self._arrow_lines:
                # TODO: thick lines
                glLineWidth(3)
                for x, y, z, value, factor, color in self._arrow_lines:
                    glColor3f(*color)
                    glBegin(GL_LINES)
                    glVertex3f(0, 0, 0)
                    glVertex3f(x, y, z)
                    glEnd()

                    self.qglColor(self._theme.axis_color)
                    # TODO: discrete
                    self.renderText(x,y,z, ('%f' % (value if self.tooltipValue == 0 else factor)).rstrip('0').rstrip('.'),
                                    font=self._theme.labels_font)

                glLineWidth(1)

        self._draw_value_lines()

    def _draw_value_lines(self):
        if self.showValueLines:
            self.value_lines_shader.bind()
            self.value_lines_shader.setUniformValue('projection', self.projection)
            self.value_lines_shader.setUniformValue('modelview', self.modelview)
            self.value_lines_shader.setUniformValue('value_line_length', float(self.valueLineLength))
            self.value_lines_shader.setUniformValue('plot_scale', self.plot_scale[0], self.plot_scale[1], self.plot_scale[2])

            glBindVertexArray(self.value_lines_vao)
            glDrawArrays(GL_LINES, 0, self.value_lines_vao.num_vertices)
            glBindVertexArray(0)

            self.value_lines_shader.release()

    def updateData(self, labels=None, setAnchors=0, **args):
        self.clear()
        self.clear_plot_transformations()
        self.value_lines = []

        if labels == None:
            labels = [anchor[3] for anchor in self.anchor_data]

        if not self.have_data or len(labels) < 3:
            self.anchor_data = []
            self.update()
            return

        if setAnchors or (args.has_key('XAnchors') and args.has_key('YAnchors') and args.has_key('ZAnchors')):
            self.setAnchors(args.get('XAnchors'), args.get('YAnchors'), args.get('ZAnchors'), labels)

        indices = [self.attribute_name_index[anchor[3]] for anchor in self.anchor_data]
        valid_data = self.getValidList(indices)
        trans_proj_data = self.create_projection_as_numeric_array(indices, validData=valid_data,
            scaleFactor=1.0, normalize=self.normalize_examples, jitterSize=-1,
            useAnchorData=1, removeMissingData=0)
        if trans_proj_data == None:
            return

        proj_data = trans_proj_data.T
        proj_data[0:3] += 0.5
        if self.data_has_discrete_class:
            proj_data[3] = self.no_jittering_scaled_data[self.attribute_name_index[self.data_domain.classVar.name]]
        self.set_plot_data(proj_data, None)
        self.symbol_scale = self.point_width*self._point_width_to_symbol_scale
        self.hide_outside = False
        self.fade_outside = False

        color_index = symbol_index = size_index = label_index = -1
        color_discrete = False
        x_discrete = self.data_domain[self.anchor_data[0][3]].varType == Discrete
        y_discrete = self.data_domain[self.anchor_data[1][3]].varType == Discrete
        z_discrete = self.data_domain[self.anchor_data[2][3]].varType == Discrete

        if self.data_has_discrete_class:
            self.discrete_palette.setNumberOfColors(len(self.data_domain.classVar.values))

        use_different_symbols = self.useDifferentSymbols and self.data_has_discrete_class and\
            len(self.data_domain.classVar.values) <= len(Symbol)

        if use_different_symbols:
            symbol_index = 3
            num_symbols_used = len(self.data_domain.classVar.values)
        else:
            num_symbols_used = -1

        if self.useDifferentColors and self.data_has_discrete_class:
            color_discrete = True
            color_index = 3

        colors = []
        if color_discrete:
            for i in range(len(self.data_domain.classVar.values)):
                c = self.discrete_palette[i]
                colors.append(c)

        self.set_shown_attributes(0, 1, 2, color_index, symbol_index, size_index, label_index,
                                  colors, num_symbols_used,
                                  x_discrete, y_discrete, z_discrete,
                                  numpy.array([1., 1., 1.]), numpy.array([0., 0., 0.]))

        def_color = QColor(150, 150, 150)
        def_symbol = 0
        def_size = 10

        if color_discrete:
            num = len(self.data_domain.classVar.values)
            values = get_variable_values_sorted(self.data_domain.classVar)
            for ind in range(num):
                symbol = ind if use_different_symbols else def_symbol
                self.legend().add_item(self.data_domain.classVar.name, values[ind], OWPoint(symbol, self.discrete_palette[ind], def_size))

        if use_different_symbols and not color_discrete:
            num = len(self.data_domain.classVar.values)
            values = get_variable_values_sorted(self.data_domain.classVar)
            for ind in range(num):
                self.legend().add_item(self.data_domain.classVar.name, values[ind], OWPoint(ind, def_color, def_size))

        self.legend().set_orientation(Qt.Vertical)
        self.legend().max_size = QSize(400, 400)
        if self.legend().pos().x() == 0:
            self.legend().setPos(QPointF(100, 100))
        self.legend().update_items()

        x_positions = proj_data[0]-0.5
        y_positions = proj_data[1]-0.5
        z_positions = proj_data[2]-0.5
        XAnchors = [anchor[0] for anchor in self.anchor_data]
        YAnchors = [anchor[1] for anchor in self.anchor_data]
        ZAnchors = [anchor[2] for anchor in self.anchor_data]
        data_size = len(self.raw_data)

        for i in range(data_size):
            if not valid_data[i]:
                continue
            if self.useDifferentColors:
                color = self.discrete_palette.getRGB(self.original_data[self.data_class_index][i])
            else:
                color = (0, 0, 0)

            len_anchor_data = len(self.anchor_data)
            x = array([x_positions[i]] * len_anchor_data)
            y = array([y_positions[i]] * len_anchor_data)
            z = array([z_positions[i]] * len_anchor_data)
            dists = numpy.sqrt((XAnchors - x)**2 + (YAnchors - y)**2 + (ZAnchors - z)**2)
            x_directions = 0.03 * (XAnchors - x) / dists
            y_directions = 0.03 * (YAnchors - y) / dists
            z_directions = 0.03 * (ZAnchors - z) / dists
            example_values = [self.no_jittering_scaled_data[attr_ind, i] for attr_ind in indices]

            for j in range(len_anchor_data):
                self.value_lines.extend([x_positions[i], y_positions[i], z_positions[i],
                                         color[0]/255.,
                                         color[1]/255.,
                                         color[2]/255.,
                                         0., 0., 0.,
                                         x_positions[i], y_positions[i], z_positions[i],
                                         color[0]/255.,
                                         color[1]/255.,
                                         color[2]/255.,
                                         x_directions[j]*example_values[j],
                                         y_directions[j]*example_values[j],
                                         z_directions[j]*example_values[j]])

        self.value_lines_vao = GLuint(0)
        glGenVertexArrays(1, self.value_lines_vao)
        glBindVertexArray(self.value_lines_vao)

        vertex_buffer_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id)
        glBufferData(GL_ARRAY_BUFFER, numpy.array(self.value_lines, 'f'), GL_STATIC_DRAW)

        vertex_size = (3+3+3)*4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(0))
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(3*4))
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, c_void_p(6*4))
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.value_lines_vao.num_vertices = len(self.value_lines) / (vertex_size / 4)

        self.update()

    def updateGraph(self, attrList=None, setAnchors=0, insideColors=None, **args):
        pass

    def setCanvasColor(self, c):
        pass

    def getSelectionsAsExampleTables(self, attrList, useAnchorData=1, addProjectedPositions=0):
        return (None, None) # TODO: this is disabled for now

        if not self.have_data:
            return (None, None)

        selected = self.get_selected_indices()

        if addProjectedPositions == 0 and not numpy.any(selected):
            return (None, self.raw_data)
        if (useAnchorData and len(self.anchor_data) < 3) or len(attrList) < 3:
            return (None, None)

        x_attr = orange.FloatVariable("X Positions")
        y_attr = orange.FloatVariable("Y Positions")
        z_attr = orange.FloatVariable("Z Positions")

        if addProjectedPositions == 1:
            domain = orange.Domain([x_attr, y_attr, z_attr] + [v for v in self.data_domain.variables])
        elif addProjectedPositions == 2:
            domain = orange.Domain(self.data_domain)
            domain.addmeta(orange.newmetaid(), x_attr)
            domain.addmeta(orange.newmetaid(), y_attr)
            domain.addmeta(orange.newmetaid(), z_attr)
        else:
            domain = orange.Domain(self.data_domain)

        domain.addmetas(self.data_domain.getmetas())

        if useAnchorData:
            indices = [self.attribute_name_index[val[3]] for val in self.anchor_data]
        else:
            indices = [self.attribute_name_index[label] for label in attrList]
        valid_data = self.getValidList(indices)
        if len(valid_data) == 0:
            return (None, None)

        array = self.create_projection_as_numeric_array(attrList, scaleFactor=self.scaleFactor, useAnchorData=useAnchorData, removeMissingData=0)
        if array == None:
            return (None, None)

        unselected = numpy.logical_not(selected)
        selected_indices, unselected_indices = list(selected), list(unselected)

        if addProjectedPositions:
            selected = orange.ExampleTable(domain, self.raw_data.selectref(selected_indices))
            unselected = orange.ExampleTable(domain, self.raw_data.selectref(unselected_indices))
            selected_index = 0
            unselected_index = 0
            for i in range(len(selected_indices)):
                if selected_indices[i]:
                    selected[selected_index][x_attr] = array[i][0]
                    selected[selected_index][y_attr] = array[i][1]
                    selected[selected_index][z_attr] = array[i][2]
                    selected_index += 1
                else:
                    unselected[unselected_index][x_attr] = array[i][0]
                    unselected[unselected_index][y_attr] = array[i][1]
                    unselected[unselected_index][z_attr] = array[i][2]
                    unselIndex += 1
        else:
            selected = self.raw_data.selectref(selected_indices)
            unselected = self.raw_data.selectref(unselected_indices)

        if len(selected) == 0:
            selected = None
        if len(unselected) == 0:
            unselected = None
        return (selected, unselected)

    def removeAllSelections(self):
        pass

    def update_point_size(self):
        self.symbol_scale = self.point_width*self._point_width_to_symbol_scale
        self.update()

    def update_alpha_value(self):
        self.update()

    def update_legend(self):
        self.update()

    def replot(self):
        pass

    def saveToFile(self):
        pass

    def _update_arrow_values(self, index):
        if not self.have_data:
            return

        if self.tooltipKind == 1:
            shown_attrs = [anchor[3] for anchor in self.anchor_data]
            self.show_tooltip(self.get_example_tooltip_text(self.raw_data[index], shown_attrs))
            return
        elif self.tooltipKind == 2:
            self.show_tooltip(self.get_example_tooltip_text(self.raw_data[index]))
            return

        if index == self._last_index:
            return
        self._last_index = index
        self._arrow_lines = []
        example = self.original_data.T[index]
        for x, y, z, attribute in self.anchor_data:
            value = example[self.attribute_name_index[attribute]]
            if value < 0:
                x = y = z = 0
            max_value = self.attr_values[attribute][1]
            factor = value / max_value
            if self.useDifferentColors:
                color = self.discrete_palette.getRGB(example[self.data_class_index])
            else:
                color = (0, 0, 0)
            self._arrow_lines.append([x*factor, y*factor, z*factor, value, factor, color])
        self._mouseover_called = True
        self.update()

    def get_example_tooltip_text(self, example, indices=None, maxIndices=20):
        if indices and type(indices[0]) == str:
            indices = [self.attributeNameIndex[i] for i in indices]
        if not indices: 
            indices = range(len(self.dataDomain.attributes))

        # don't show the class value twice
        if example.domain.classVar:
            classIndex = self.attributeNameIndex[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)      

        text = "<b>Attributes:</b><br>"
        for index in indices[:maxIndices]:
            attr = self.attributeNames[index]
            if attr not in example.domain:  text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            elif example[attr].isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(attr))
            else:                           text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(attr), Qt.escape(str(example[attr])))
        if len(indices) > maxIndices:
            text += "&nbsp;"*4 + " ... <br>"

        if example.domain.classVar:
            text = text[:-4]
            text += "<hr><b>Class:</b><br>"
            if example.getclass().isSpecial(): text += "&nbsp;"*4 + "%s = ?<br>" % (Qt.escape(example.domain.classVar.name))
            else:                              text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain.classVar.name), Qt.escape(str(example.getclass())))

        if len(example.domain.getmetas()) != 0:
            text = text[:-4]
            text += "<hr><b>Meta attributes:</b><br>"
            # show values of meta attributes
            for key in example.domain.getmetas():
                try: text += "&nbsp;"*4 + "%s = %s<br>" % (Qt.escape(example.domain[key].name), Qt.escape(str(example[key])))
                except: pass
        return text[:-4]        # remove the last <br>

    def mousePressEvent(self, event):
        if not self.have_data:
            return

        # Filter events (no panning or scaling)
        if event.buttons() & Qt.LeftButton or\
           (event.buttons() & Qt.RightButton and not QApplication.keyboardModifiers() & Qt.ShiftModifier) or\
           (event.buttons() & Qt.MiddleButton and not QApplication.keyboardModifiers() & Qt.ShiftModifier):
            OWPlot3D.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if not self.have_data:
            return

        pos = event.pos()

        self._last_index = -1
        self._mouseover_called = False
        self._check_mouseover(pos)
        if not self._mouseover_called and 'linear' in self.name.lower():
            before = len(self._arrow_lines)
            self._arrow_lines = []
            if before > 0:
                self.update()

        OWPlot3D.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if not self.have_data:
            return

        OWPlot3D.mouseReleaseEvent(self, event)
