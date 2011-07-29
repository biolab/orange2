#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

UnconnectedLinesCurve::UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent, QGraphicsScene* scene): Curve(x_data, y_data, parent, scene)
{

}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::update_properties()
{
    const Data d = data();
    const int n = d.size()/2;
    const int m = m_items.size();
    if (m > n)
    {
        for (int i = n; i < m; ++i)
        {
            delete m_items.takeLast();
        }
    }
    else if (m < n)
    {
        for (int i = m; i < n; ++i)
        {
            m_items << new QGraphicsLineItem(this);
        }
    }
    Q_ASSERT(m_items.size() == n);
    QLineF line;
    QPen p = pen();
    p.setCosmetic(true);
    for (int i = 0; i < n; ++i)
    {
        line.setLine( d[2*i].x, d[2*i].y, d[2*i+1].x, d[2*i+1].y );
        m_items[i]->setLine(graph_transform().map(line));
        m_items[i]->setPen(p);
    }
}
