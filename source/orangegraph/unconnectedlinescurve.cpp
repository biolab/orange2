#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

UnconnectedLinesCurve::UnconnectedLinesCurve(QList< double > xData, QList< double > yData, QGraphicsItem* parent, QGraphicsScene* scene): Curve(xData, yData, parent, scene)
{

}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::updateProperties()
{
    qDebug() << "updating an ULC with axes" << axes();
    Data d = data();
    int n = d.size()/2;
    int m = m_items.size();
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
    Q_ASSERT(m_items.size() == data().size()/2);
    qDebug() << m_items.size();
    for (int i = 0; i < n; ++i)
    {
        QLineF line;
        line.setP1(QPointF(d[2*i].x, d[2*i].y));
        line.setP2(QPointF(d[2*i+1].x, d[2*i+1].y));
        m_items[i]->setLine(graphTransform().map(line));
        m_items[i]->setPen(m_pen);
    }
}

void UnconnectedLinesCurve::setPen(QPen pen)
{
    m_pen = pen;
    updateProperties();
}

QPen UnconnectedLinesCurve::pen() const
{
    return m_pen;
}
