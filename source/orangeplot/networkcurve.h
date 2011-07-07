#ifndef NETWORKCURVE_H
#define NETWORKCURVE_H

#include "curve.h"

struct NodeItem
{
    int index;
    bool marked;
    bool show;
    bool highlight;
    QString label;
    QString tooltip;
    int uuid;
    
    QPixmap* image;
    QPen pen;
    QColor nocolor;
    QColor color;
    int size;
    int style;
};

struct EdgeItem
{
    NodeItem* u;
    NodeItem* v;
    int links_index;
    bool arrowu;
    bool arrowv;
    double weight;
    QString label;
    QPen pen;
};

typedef QPair<double, double> Coord;

class NetworkCurve : public Curve
{
public:
    typedef QMap<int, Coord> Coordinates;
    typedef QList<EdgeItem> Edges;
    typedef QMap<int, NodeItem> Vertices;

    NetworkCurve(const Coordinates& coordinates, const Edges& edges, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    NetworkCurve(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~NetworkCurve();
    
    virtual void updateProperties();
    
    Coordinates coors;
    Vertices vertices;
    Edges edges;
    
private:
    QMap<int, QGraphicsPathItem*> m_vertex_items;
    QList<QGraphicsLineItem*> m_edge_items;
};

#endif // NETWORKCURVE_H
