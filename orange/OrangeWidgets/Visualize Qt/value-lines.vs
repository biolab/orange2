attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;

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
