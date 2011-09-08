varying float transparency;
uniform bool use_transparency;

void main(void)
{
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1. - transparency - 0.6);
    if (!use_transparency)
        gl_FragColor.a = 1.;
}
