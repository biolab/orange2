#ifndef UNCONNECTEDLINESCURVE_H
#define UNCONNECTEDLINESCURVE_H

#include "curve.h"
#include <QtGui/QPen>

class UnconnectedLinesCurve : public Curve
{

public:
    UnconnectedLinesCurve(QList< double > xData, QList< double > yData, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void updateProperties();
    
    void setPen(QPen pen);
    QPen pen() const;
    
private:
    QList<QGraphicsLineItem*> m_items;
    QPen m_pen;
};

#endif // UNCONNECTEDLINESCURVE_H
