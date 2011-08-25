#ifndef PLOT_3D_H
#define PLOT_3D_H

#include <QtOpenGL/QGLWidget>

class Plot3D : public QGLWidget {
  Q_OBJECT

public:
    explicit Plot3D(QWidget* parent = 0);
    virtual ~Plot3D();
};

#endif
