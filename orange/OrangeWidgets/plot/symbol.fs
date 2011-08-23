#version 150

in vec4 var_color;
uniform bool apply_texture;
uniform sampler2D texture;
uniform vec2 screen_size;

void main(void)
{
    if (apply_texture)
    {
        gl_FragColor = mix(var_color, texture2D(texture, vec2(10., 10.)*gl_FragCoord.xy/screen_size), 0.3);
    }
    else
    {
        gl_FragColor = var_color;
    }
}
