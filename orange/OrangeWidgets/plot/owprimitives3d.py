import os
from owplot3d import Symbol, normal_from_points

symbol_map = {
    Symbol.RECT:      'primitives/cube.obj',
    Symbol.TRIANGLE:  'primitives/pyramid.obj',
    Symbol.DTRIANGLE: 'primitives/dpyramid.obj',
    Symbol.CIRCLE:    'primitives/sphere.obj',
    Symbol.LTRIANGLE: 'primitives/lpyramid.obj',
    Symbol.DIAMOND:   'primitives/diamond.obj',
    Symbol.WEDGE:     'primitives/wedge.obj',
    Symbol.LWEDGE:    'primitives/lwedge.obj',
    Symbol.CROSS:     'primitives/cross.obj',
    Symbol.XCROSS:    'primitives/xcross.obj'
}

symbol_data = {} # Cache: contains triangles + their normals for each needed symbol.

def get_symbol_data(symbol):
    if not Symbol.is_valid(symbol):
        return []
    if symbol in symbol_data:
        return symbol_data[symbol]
    file_name = symbol_map[symbol]
    lines = open(os.path.join(os.path.dirname(__file__), file_name)).readlines()
    vertices_lines = filter(lambda line: line.startswith('v'), lines)
    vertices = [map(float, line.split()[1:]) for line in vertices_lines]
    faces_lines = filter(lambda line: line.startswith('f'), lines)
    faces = [map(int, line.split()[1:]) for line in faces_lines]
    triangles = [[vertices[face[0]-1],
                  vertices[face[1]-1],
                  vertices[face[2]-1]] for face in faces]
    for triangle in triangles:
        triangle.extend(normal_from_points(*triangle))
    symbol_data[symbol] = triangles
    return triangles
