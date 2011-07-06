#ifndef OW_GRAPH_H
#define OW_GRAPH_H

#include <QtGui/QGraphicsView>

class OWGraph : public QGraphicsView
{
Q_OBJECT
public:
    OWGraph(QWidget* parent = 0);
    virtual ~OWGraph();
};

#endif // OW_GRAPH_H
