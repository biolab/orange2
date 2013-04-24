/*
 * Used by Canvas3D to render edges between nodes.
 */

attribute vec3 position;

uniform mat4 projection, model, view;
uniform vec3 translation, scale;

void main()
{
    gl_Position = projection * model * view * vec4((position+translation)*scale, 1.);
}
