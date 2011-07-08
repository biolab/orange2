#include "networkcurve.h"

#include <QtCore/QMap>
#include <QtCore/QList>

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
    
    const Nodes nodes = get_nodes();
    
    if (m_vertex_items.keys() != nodes.keys())
    {
        qDeleteAll(m_vertex_items);
        m_vertex_items.clear();
        Nodes::ConstIterator it = nodes.constBegin();
        Nodes::ConstIterator end = nodes.constEnd();
        for (; it != end; ++it)
        {
            m_vertex_items.insert(it.key(), new QGraphicsPathItem(this));
        }
    }
    
    NodeItem node;
    QGraphicsPathItem* item;
    Nodes::ConstIterator nit = nodes.constBegin();
    Nodes::ConstIterator nend = nodes.constEnd();
    for (; nit != nend; ++nit)
    {
        node = nit.value();
        item = m_vertex_items[nit.key()];
        item->setPos( t.map(QPointF(node.x, node.y)) );
        item->setBrush(brush());
        item->setPen(node.pen);
        item->setToolTip(node.tooltip);
        item->setPath(pathForSymbol(node.style, node.size));
    }
    
    Q_ASSERT(m_vertex_items.size() == nodes.size());
    
    const Edges edges = get_edges();
    
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
    
    QLineF line;
    QGraphicsLineItem* line_item;
    n = edges.size();
    for (int i = 0; i < n; ++i)
    {
        EdgeItem edge = edges[i];
        node = nodes[edge.u->index];
        line.setP1(QPointF(node.x, node.y));
        node = nodes[edge.v->index];
        line.setP2(QPointF(node.x, node.y));
        line_item = m_edge_items[i];
        line_item->setLine( t.map(line) );
        line_item->setPen(edges[i].pen);
    }
}

QRectF NetworkCurve::dataRect() const
{
    QRectF r;
    bool first = true;
    foreach (const NodeItem& node, get_nodes())
    {
        if (first)
        {
            r = QRectF(node.x, node.y, 0, 0);
            first = false;
        }
        else
        {
            r.setTop( qMin(r.top(), node.y) );
            r.setBottom( qMax(r.bottom(), node.y) );
            r.setLeft( qMin(r.left(), node.x) );
            r.setRight( qMax(r.right(), node.y) );
        }
    }
    qDebug() << "NetworkCurve::dataRect()" << r;
    return r;
}

