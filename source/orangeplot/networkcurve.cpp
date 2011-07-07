#include "networkcurve.h"

#include <QtCore/QMap>
#include <QtCore/QList>

NetworkCurve::NetworkCurve(const Coordinates& coordinates, const Edges& edges, QGraphicsItem* parent, QGraphicsScene* scene): 
Curve(parent, scene),
    coors(coordinates),
edges(edges)
{

}

NetworkCurve::NetworkCurve(QGraphicsItem* parent, QGraphicsScene* scene): Curve(parent, scene)
{

}

NetworkCurve::~NetworkCurve()
{

}

void NetworkCurve::updateProperties()
{
    const Data d = data();
    const QTransform t = graphTransform();
    int m, n;
    
    if (m_vertex_items.keys() != coors.keys())
    {
        qDeleteAll(m_vertex_items);
        m_vertex_items.clear();
        Coordinates::ConstIterator cit = coors.constBegin();
        Coordinates::ConstIterator cend = coors.constEnd();
        for (; cit != cend; ++cit)
        {
            m_vertex_items.insert(cit.key(), new QGraphicsPathItem(this));
        }
    }
    
    QPair<double, double> p;
    QGraphicsPathItem* item;
    Coordinates::ConstIterator cit = coors.constBegin();
    Coordinates::ConstIterator cend = coors.constEnd();
    for (; cit != cend; ++cit)
    {
        p = cit.value();
        item = m_vertex_items[cit.key()];
        item->setPos( t.map(QPointF(p.first, p.second)) );
        item->setBrush(brush());
        NodeItem v = vertices[cit.key()];
        item->setPen(v.pen);
        item->setToolTip(v.tooltip);
        item->setPath(pathForSymbol(symbol(), v.size));
    }
    
    Q_ASSERT(m_vertex_items.size() == coors.size());
    
    n = edges.size();
    m = m_edge_items.size();
    
    for (int i = n; i < m; ++i)
    {
        delete m_edge_items.takeLast();
    }
    
    for (int i = m; i < n; ++i)
    {
        m_edge_items << new QGraphicsLineItem(this);
    }
    
    Q_ASSERT(m_edge_items.size() == edges.size());
    
    QPair<int, int> points;
    EdgeItem edge;
    QLineF line;
    QGraphicsLineItem* line_item;
    n = edges.size();
    for (int i = 0; i < n; ++i)
    {
        edge = edges[i];
        p = coors[edge.u->index];
        line.setP1(QPointF( p.first, p.second ));
        p = coors[edge.v->index];
        line.setP2(QPointF( p.first, p.second ));
        line_item = m_edge_items[i];
        line_item->setLine( t.map(line) );
        line_item->setPen(edges[i].pen);
    }
}
