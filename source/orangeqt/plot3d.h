#ifndef PLOT_3D_H
#define PLOT_3D_H

#include "plot.h"
#include <QtOpenGL/QGLWidget>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>
#include <GL/gl.h>

class Plot3D : public QGLWidget {
  Q_OBJECT

public:
    explicit Plot3D(QWidget* parent = 0);
    virtual ~Plot3D();

    void set_symbol_geometry(int symbol,
                             int type,
                             const QList<QVector3D>& geometry);

    void set_data(quint64 array_address,
                  int num_examples,
                  int example_size);

    void set_valid_data(quint64 valid_data_address); // Until I get QList<bool> to work, address of numpy array will have to do

    void update_data(int x_index, int y_index, int z_index,
                     int color_index, int symbol_index, int size_index, int label_index,
                     const QList<QColor>& colors, int num_symbols_used,
                     bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols);

    void draw_data(GLuint shader_id, float alpha_value);
    void draw_data_solid(); // (only draws solid geometry as a performance optimization)

    QList<double> get_min_max_selected(const QList<int>& area,
                                      const QMatrix4x4& mvp,
                                      const QList<int>& viewport,
                                      const QVector3D& plot_scale,
                                      const QVector3D& plot_translation);

    void select_points(const QList<int>& area,
                       const QMatrix4x4& mvp,
                       const QList<int>& viewport,
                       const QVector3D& plot_scale,
                       const QVector3D& plot_translation,
                       Plot::SelectionBehavior behavior = Plot::AddSelection);

    void unselect_all_points();

    QList<bool> get_selected_indices();

private:
    float* data_array;
    bool* valid_data;
    QVector<bool> selected_indices; // Array of length num_examples
    int num_examples;
    int example_size;
    int num_selected_vertices;
    int num_unselected_vertices;
    int num_edges_vertices;
    GLuint vbo_selected_id;   // Triangles (drawn opaque)
    GLuint vbo_unselected_id; // Triangles (drawn slightly transparent)
    GLuint vbo_edges_id;      // Edges are in a separated VBO (but drawn together with vbo_unselected)
    bool vbos_generated;

    int x_index;
    int y_index;
    int z_index;

    QMap<int, QList<QVector3D> > geometry_data_2d;
    QMap<int, QList<QVector3D> > geometry_data_3d;
    QMap<int, QList<QVector3D> > geometry_data_edges_2d;
    QMap<int, QList<QVector3D> > geometry_data_edges_3d;
};

#endif
