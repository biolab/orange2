#ifndef PLOT_3D_H
#define PLOT_3D_H

#include <QtOpenGL/QGLWidget>
#include <QVector3D>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h> // TODO: Windows?
#include <GL/glext.h>

class Plot3D : public QGLWidget {
  Q_OBJECT

public:
    explicit Plot3D(QWidget* parent = 0);
    virtual ~Plot3D();

    void set_symbol_geometry(int symbol, int type, const QList<QVector3D>& geometry);

    void set_data(quint64 array_address, int num_examples, int example_size);
    void update_data(int x_index, int y_index, int z_index,
                int color_index, int symbol_index, int size_index, int label_index,
                const QList<QColor>& colors, int num_symbols_used,
                bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols);
    void draw_data();

private:
    float* data_array;
    int num_examples;
    int example_size;
    int num_vertices;
    GLuint vbo_id;
    bool vbo_generated;

    QMap<int, QList<QVector3D> > geometry_data_2d;
    QMap<int, QList<QVector3D> > geometry_data_3d;
    QMap<int, QList<QVector3D> > geometry_data_edges;
};

#endif
