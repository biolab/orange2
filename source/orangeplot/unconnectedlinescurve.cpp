#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

UnconnectedLinesCurve::UnconnectedLinesCurve(const QList< double >& xData, const QList< double >& yData, QGraphicsItem* parent, QGraphicsScene* scene): Curve(xData, yData, parent, scene)
{

}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::updateProperties()
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
    for (int i = 0; i < n; ++i)
    {
        line.setLine( d[2*i].x, d[2*i].y, d[2*i+1].x, d[2*i+1].y );
        m_items[i]->setLine(graphTransform().map(line));
        m_items[i]->setPen(pen());
    }
}
