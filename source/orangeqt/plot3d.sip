
class Plot3D : QGLWidget {

%TypeHeaderCode
#include "plot3d.h"
%End

public:
    Plot3D(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot3D();

    void set_symbol_geometry(int symbol, int type, const QList<QVector3D>& geometry);

    void set_data(quint64 array_address, int num_examples, int example_size);
    void update_data(int x_index, int y_index, int z_index,
                int color_index, int symbol_index, int size_index, int label_index,
                const QList<QColor>& colors, int num_symbols_used,
                bool x_discrete, bool y_discrete, bool z_discrete, bool use_2d_symbols);
    void draw_data();
};
