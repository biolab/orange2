attribute vec3 position;
varying float transparency;

uniform mat4 projection;
uniform mat4 modelview;
uniform vec3 cam_position;

void main(void)
{
    transparency = clamp(dot(normalize(cam_position-position), normalize(position)), 0., 1.);
    gl_Position = projection * modelview * vec4(position, 1.);
}
