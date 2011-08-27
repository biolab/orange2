
class Plot3D : QGLWidget {

%TypeHeaderCode
#include "plot3d.h"
%End

public:
    Plot3D(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot3D();

    void set_symbol_geometry(int symbol,
                             int type,
                             const QList<QVector3D>& geometry);

    void set_data(quint64 array_address,
                  int num_examples,
                  int example_size);
    void update_data(int x_index, int y_index, int z_index,
                     int color_index, int symbol_index, int size_index, int label_index,
                     const QList<QColor>& colors, int num_symbols_used,
                     bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols);
    void draw_data(GLuint shader_id,
                   float alpha_value);
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
};
