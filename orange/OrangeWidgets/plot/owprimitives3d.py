import os
import re
from owplot3d import Symbol
import numpy

def normalize(vec):
    return vec / numpy.sqrt(numpy.sum(vec**2))

def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value

def normal_from_points(p1, p2, p3):
    if isinstance(p1, (list, tuple)):
        v1 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
        v2 = [p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]]
    else:
        v1 = p2 - p1
        v2 = p3 - p1
    return normalize(numpy.cross(v1, v2))

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

symbol_map_2d = {
    Symbol.RECT:      'primitives/rect.obj',
    Symbol.TRIANGLE:  'primitives/triangle.obj',
    Symbol.DTRIANGLE: 'primitives/dtriangle.obj',
    Symbol.CIRCLE:    'primitives/circle.obj',
    Symbol.LTRIANGLE: 'primitives/ltriangle.obj',
    Symbol.DIAMOND:   'primitives/diamond_2d.obj',
    Symbol.WEDGE:     'primitives/wedge_2d.obj',
    Symbol.LWEDGE:    'primitives/lwedge_2d.obj',
    Symbol.CROSS:     'primitives/cross_2d.obj',
    Symbol.XCROSS:    'primitives/xcross_2d.obj'
}

symbol_edge_map = {
    Symbol.RECT:      'primitives/rect_edges.obj',
    Symbol.TRIANGLE:  'primitives/triangle_edges.obj',
    Symbol.DTRIANGLE: 'primitives/dtriangle_edges.obj',
    Symbol.CIRCLE:    'primitives/circle_edges.obj',
    Symbol.LTRIANGLE: 'primitives/ltriangle_edges.obj',
    Symbol.DIAMOND:   'primitives/diamond_edges.obj',
    Symbol.WEDGE:     'primitives/wedge_edges.obj',
    Symbol.LWEDGE:    'primitives/lwedge_edges.obj',
    Symbol.CROSS:     'primitives/cross_edges.obj',
    Symbol.XCROSS:    'primitives/xcross_edges.obj'
}

_symbol_data = {} # Cache: contains triangles + their normals for each needed symbol.
_symbol_data_2d = {}
_symbol_edges = {}

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

def get_2d_symbol_data(symbol):
    if not Symbol.is_valid(symbol):
        return []
    if symbol in _symbol_data_2d:
        return _symbol_data_2d[symbol]
    file_name = symbol_map_2d[symbol]
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    triangles = parse_obj(file_name)
    _symbol_data_2d[symbol] = triangles
    return triangles

def get_2d_symbol_edges(symbol):
    if not Symbol.is_valid(symbol):
        return []
    if symbol in _symbol_edges:
        return _symbol_edges[symbol]
    file_name = symbol_edge_map[symbol]
    file_name = os.path.join(os.path.dirname(__file__), file_name)
    lines = open(file_name).readlines()
    vertices_lines = filter(lambda line: line.startswith('v'), lines)
    edges_lines =    filter(lambda line: line.startswith('f'), lines)
    vertices = [map(float, line.split()[1:]) for line in vertices_lines]
    edges_indices = [map(int, line.split()[1:]) for line in edges_lines]
    edges = []
    for i0, i1 in edges_indices:
        v0 = vertices[i0-1]
        v1 = vertices[i1-1]
        edges.append([v0, v1])
    _symbol_edges[symbol] = edges
    return edges
