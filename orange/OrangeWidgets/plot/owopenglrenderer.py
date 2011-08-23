
class OWOpenGLRenderer:
    '''OpenGL 3 deprecated a lot of old (1.x) functions, particulary, it removed
       immediate mode (glBegin, glEnd, glVertex paradigm). Vertex buffer objects and similar
       (through glDrawArrays for example) should be used instead. This class simplifies
       the usage of that functionality by providing methods which resemble immediate mode.'''
    def __init__(self):
        pass

    def draw_point(self, location):
        pass

    def draw_line(self, beginning, end):
        pass

    def draw_rectangle(self, vertex_min, vertex_max):
        pass

    def draw_triangle(self, vertex0, vertex1, vertex2):
        pass
