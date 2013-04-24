attribute vec3 position;

uniform mat4 projection;
uniform mat4 modelview;

void main(void)
{
    gl_Position = projection * modelview * vec4(position, 1.);
}
