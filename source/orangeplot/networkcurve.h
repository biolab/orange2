#ifndef NETWORKCURVE_H
#define NETWORKCURVE_H

#include "curve.h"

struct NodeItem
{
    double x;
    double y;
    
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

class NetworkCurve : public Curve
{
public:
    typedef QList<EdgeItem> Edges;
    typedef QMap<int, NodeItem> Nodes;

    NetworkCurve(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~NetworkCurve();
    
    virtual void updateProperties();
    
    virtual Nodes get_nodes() const = 0;
    virtual Edges get_edges() const = 0;
    
    virtual QRectF dataRect() const;
    
private:
    QMap<int, QGraphicsPathItem*> m_vertex_items;
    QList<QGraphicsLineItem*> m_edge_items;
};

#endif // NETWORKCURVE_H
