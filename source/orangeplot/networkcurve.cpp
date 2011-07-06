#include "networkcurve.h"

#include <QtCore/QMap>
#include <QtCore/QList>

NetworkCurve::NetworkCurve(const Coordinates& coordinates, const Edges& edges, QGraphicsItem* parent, QGraphicsScene* scene): 
Curve(parent, scene),
coordinates(coordinates),
edges(edges)
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
    
    qDeleteAll(m_vertex_items);
    m_vertex_items.clear();
    
    QPair<double, double> p;
    QGraphicsPathItem* item;
    Coordinates::ConstIterator cit = coordinates.constBegin();
    Coordinates::ConstIterator cend = coordinates.constEnd();
    QPainterPath path = pathForSymbol(symbol(), pointSize());
    for (; cit != cend; ++cit)
    {
        p = cit.value();
        item = new QGraphicsPathItem(path, this);
        item->setPos( t.map(QPointF(p.first, p.second)) );
        item->setBrush(brush());
        m_vertex_items.insert(cit.key(), item);
    }
    
    Q_ASSERT(m_vertex_items.size() == coordinates.size());
    
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
    QLineF line;
    QGraphicsLineItem* line_item;
    n = edges.size();
    for (int i = 0; i < n; ++i)
    {
        points = edges[i];
        p = coordinates[points.first];
        line.setP1(QPointF( p.first, p.second ));
        p = coordinates[points.second];
        line.setP2(QPointF( p.first, p.second ));
        line_item = m_edge_items[i];
        line_item->setLine( t.map(line) );
        line_item->setPen(pen());
    }
}

