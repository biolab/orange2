/*
 * Used by Canvas3D to render nodes.
 */

attribute vec3 position;

uniform mat4 projection, model, view;
uniform vec3 translation, scale;

void main()
{
    gl_PointSize = 7.;
    gl_Position = projection * model * view * vec4((position+translation)*scale, 1.);
}
