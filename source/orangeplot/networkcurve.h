#ifndef NETWORKCURVE_H
#define NETWORKCURVE_H

#include "curve.h"

typedef QPair<double, double> Coord;

class NetworkCurve : public Curve
{
public:
    typedef QMap<int, Coord> Coordinates;
    typedef QList<QPair<int, int> > Edges;    

    NetworkCurve(const Coordinates& coordinates, const Edges& edges, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~NetworkCurve();
    
    virtual void updateProperties();
    
    Coordinates coordinates;
    Edges edges;
    
private:
    QMap<int, QGraphicsPathItem*> m_vertex_items;
    QList<QGraphicsLineItem*> m_edge_items;
};

#endif // NETWORKCURVE_H
