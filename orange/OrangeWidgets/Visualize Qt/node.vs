/*
 * Used by Canvas3D to render nodes.
 */

attribute vec3 position;
attribute vec3 offset;
attribute vec3 color;
attribute vec2 selected_marked;

const float MODE_NORMAL = 0.;
const float MODE_SELECTED = 1.;
const float MODE_MARKED = 2.;

uniform float mode;
uniform mat4 projection, model, view;
uniform vec3 translation, scale;

varying vec4 var_color;

void main()
{
    vec3 offset_rotated = offset.xyz;
    offset_rotated *= 0.01;

    // Calculate inverse of rotations (in this case, inverse
    // is actually just transpose), so that polygons face
    // camera all the time.
    mat3 invs;
    mat4 modelview = view * model;

    invs[0][0] = modelview[0][0];
    invs[0][1] = modelview[1][0];
    invs[0][2] = modelview[2][0];

    invs[1][0] = modelview[0][1];
    invs[1][1] = modelview[1][1];
    invs[1][2] = modelview[2][1];

    invs[2][0] = modelview[0][2];
    invs[2][1] = modelview[1][2];
    invs[2][2] = modelview[2][2];

    offset_rotated = invs * offset_rotated;

    vec3 pos = position.xyz;
    pos += translation;
    pos *= scale;
    vec4 off_pos = vec4(pos+offset_rotated, 1.);

    // Hide when unwanted
    if (mode == MODE_SELECTED && selected_marked.x == 0.)
        off_pos = vec4(0., 0., 0., 0.);

    if (mode == MODE_MARKED && selected_marked.y == 0.)
        off_pos = vec4(0., 0., 0., 0.);

    gl_Position = projection * model * view * off_pos;
    var_color = vec4(color, 1.);
}
