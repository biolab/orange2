import os
import re
from owplot3d import Symbol, normal_from_points

symbol_map = {
    Symbol.RECT:      'primitives/sphere.obj',
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

_symbol_data = {} # Cache: contains triangles + their normals for each needed symbol.

def parse_obj(file_name):
    lines = open(file_name).readlines()
    normals_lines =  filter(lambda line: line.startswith('vn'), lines)
    vertices_lines = filter(lambda line: line.startswith('v'), lines)
    faces_lines =    filter(lambda line: line.startswith('f'), lines)
    normals =  [map(float, line.split()[1:]) for line in normals_lines]
    vertices = [map(float, line.split()[1:]) for line in vertices_lines]
    if len(normals) > 0:
        pattern = r'f (\d+)//(\d+) (\d+)//(\d+) (\d+)//(\d+)'
        faces = [map(int, re.match(pattern, line).groups()) for line in faces_lines]
        triangles = [[vertices[face[0]-1],
                      vertices[face[2]-1],
                      vertices[face[4]-1],
                      normals[face[1]-1],
                      normals[face[3]-1],
                      normals[face[5]-1]] for face in faces]
    else:
        faces = [map(int, line.split()[1:]) for line in faces_lines]
        triangles = []
        for face in faces:
            v0 = vertices[face[0]-1]
            v1 = vertices[face[1]-1]
            v2 = vertices[face[2]-1]
            normal = normal_from_points(v0, v1, v2)
            triangles.append([v0, v1, v2, normal, normal, normal])
    return triangles

def get_symbol_data(symbol):
    if not Symbol.is_valid(symbol):
        return []
    if symbol in _symbol_data:
        return _symbol_data[symbol]
    file_name = symbol_map[symbol]
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    triangles = parse_obj(file_name)
    _symbol_data[symbol] = triangles
    return triangles
