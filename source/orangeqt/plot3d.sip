
class Plot3D : QGLWidget {

%TypeHeaderCode
#include "plot3d.h"
%End

public:
    Plot3D(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot3D();
};
