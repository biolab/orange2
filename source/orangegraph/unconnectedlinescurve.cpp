#include "unconnectedlinescurve.h"
#include <QtGui/QPen>

UnconnectedLinesCurve::UnconnectedLinesCurve(QList< double > xData, QList< double > yData, QGraphicsItem* parent, QGraphicsScene* scene): Curve(xData, yData, parent, scene)
{

}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::update()
{
    Data d = data();
    int n = data().size()/2;
    int m = m_items.size();
    for (int i = n; i < m; ++i)
    {
        delete m_items.takeLast();
    }
    for (int i = m; i < n; ++i)
    {
        m_items << new QGraphicsLineItem(this);
    }
    for (int i = 0; i < n; ++i)
    {
        QLineF line;
        line.setP1(QPointF(d[2*i].x, d[2*i].y));
        line.setP1(QPointF(d[2*i+1].x, d[2*i+1].y));
        m_items[i]->setLine(line);
        m_items[i]->setPen(m_pen);
    }
}

void UnconnectedLinesCurve::setPen(QPen pen)
{
    m_pen = pen;
    update();
}

QPen UnconnectedLinesCurve::pen() const
{
    return m_pen;
}
