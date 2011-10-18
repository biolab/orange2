attribute vec3 position;
attribute vec3 normal;

varying vec4 out_color;

uniform mat4 projection;
uniform mat4 modelview;

const vec3 light_direction = vec3(-0.8305, 0.4983, 0.2491);

void main(void)
{
    gl_Position = projection * modelview * vec4(position, 1.);
    float diffuse = clamp(dot(light_direction, normalize((modelview * vec4(normal, 0.)).xyz)), 0., 1.);
    out_color = vec4(vec3(1., 1., 1.) * diffuse + 0.1, 1.);
}
