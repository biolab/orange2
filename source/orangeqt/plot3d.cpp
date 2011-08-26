#include "plot3d.h"
#include <iostream>
#include <QGLFormat>
#include <GL/glx.h>
#include <GL/glxext.h> // TODO: Windows?
#include <GL/glext.h>

PFNGLGENBUFFERSARBPROC glGenBuffers = NULL;
PFNGLBINDBUFFERPROC glBindBuffer = NULL;
PFNGLBUFFERDATAPROC glBufferData = NULL;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC glEnableVertexAttribArray = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC glDisableVertexAttribArray = NULL;
typedef void (APIENTRYP PFNGLGETVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
PFNGLGETVERTEXATTRIBPOINTERPROC glVertexAttribPointer = NULL;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

Plot3D::Plot3D(QWidget* parent) : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    vbo_generated = false;
}

Plot3D::~Plot3D()
{
}

template<class T>
inline T clamp(T value, T min, T max)
{
    if (value > max)
        return max;
    if (value < min)
        return min;
    return value;
}

void Plot3D::set_symbol_geometry(int symbol, int type, const QList<QVector3D>& geometry)
{
    if (type == 0)
        geometry_data_2d[symbol] = geometry;
    else if (type == 1)
        geometry_data_3d[symbol] = geometry;
    else
        geometry_data_edges[symbol] = geometry;
}

void Plot3D::set_data(quint64 array_address, int num_examples, int example_size)
{
    this->data_array = reinterpret_cast<float*>(array_address); // 32-bit systems, endianness?
    this->num_examples = num_examples;
    this->example_size = example_size;

    // Load required extensions (OpenGL context should be up by now).
#ifdef _WIN32
    // TODO: wglGetProcAddress
#else
    glGenBuffers = (PFNGLGENBUFFERSARBPROC)glXGetProcAddress((const GLubyte*)"glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)glXGetProcAddress((const GLubyte*)"glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)glXGetProcAddress((const GLubyte*)"glBufferData");
    glVertexAttribPointer = (PFNGLGETVERTEXATTRIBPOINTERPROC)glXGetProcAddress((const GLubyte*)"glVertexAttribPointer");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC)glXGetProcAddress((const GLubyte*)"glEnableVertexAttribArray");
    glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC)glXGetProcAddress((const GLubyte*)"glDisableVertexAttribArray");
#endif
}

void Plot3D::update_data(int x_index, int y_index, int z_index,
                         int color_index, int symbol_index, int size_index, int label_index,
                         const QList<QColor>& colors, int num_symbols_used,
                         bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols)
{
    const float scale = 0.001;
    float* vbo_data = new float[num_examples * 144 * 13];
    float* dest = vbo_data;
    int size_in_bytes = 0;
    QMap<int, QList<QVector3D> >& geometry = use_2d_symbols ? geometry_data_2d : geometry_data_3d;

    for (int index = 0; index < num_examples; ++index)
    {
        float* example = data_array + index*example_size;
        float x_pos = *(example + x_index);
        float y_pos = *(example + y_index);
        float z_pos = *(example + z_index);

        int symbol = 0;
        if (num_symbols_used > 1 && symbol_index > -1)
            symbol = *(example + symbol_index) * num_symbols_used;

        float size = *(example + size_index);
        if (size_index < 0 || size < 0.)
            size = 1.;

        float color_value = *(example + color_index);
        int num_colors = colors.count();
        QColor color;

        if (num_colors > 0)
            color = colors[clamp(int(color_value * num_colors), 0, num_colors-1)]; // TODO: garbage values sometimes?
        else if (color_index > -1)
            color = QColor(0., 0., color_value);
        else
            color = QColor(0., 0., 0.8);

        for (int i = 0; i < geometry[symbol].count(); i += 6) {
            size_in_bytes += 3*13*4;

            for (int j = 0; j < 3; ++j)
            {
                // position
                *dest = x_pos; dest++; 
                *dest = y_pos; dest++; 
                *dest = z_pos; dest++; 

                // offset
                *dest = geometry[symbol][i+j].x()*size*scale; dest++;
                *dest = geometry[symbol][i+j].y()*size*scale; dest++;
                *dest = geometry[symbol][i+j].z()*size*scale; dest++;

                // color
                *dest = color.redF(); dest++;
                *dest = color.greenF(); dest++;
                *dest = color.blueF(); dest++;

                // normal
                *dest = geometry[symbol][i+3+j].x(); dest++;
                *dest = geometry[symbol][i+3+j].y(); dest++;
                *dest = geometry[symbol][i+3+j].z(); dest++;

                // index
                *dest = index; dest++;
            }
        }
    }

    glGenBuffers(1, &vbo_id);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
    glBufferData(GL_ARRAY_BUFFER, size_in_bytes, vbo_data, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vbo_data;

    num_vertices = size_in_bytes / (13*4);
    vbo_generated = true;
}

void Plot3D::draw_data()
{
    if (!vbo_generated)
        return;

    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 13*4, BUFFER_OFFSET(0));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 13*4, BUFFER_OFFSET(3*4));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 13*4, BUFFER_OFFSET(6*4));
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 13*4, BUFFER_OFFSET(9*4));
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 13*4, BUFFER_OFFSET(12*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);

    glDrawArrays(GL_TRIANGLES, 0, num_vertices);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

#include "plot3d.moc"
