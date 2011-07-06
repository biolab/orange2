#ifndef UNCONNECTEDLINESCURVE_H
#define UNCONNECTEDLINESCURVE_H

#include "curve.h"
#include <QtGui/QPen>

class UnconnectedLinesCurve : public Curve
{

public:
    UnconnectedLinesCurve(const QList< double >& xData, const QList< double >& yData, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void updateProperties();
    
private:
    QList<QGraphicsLineItem*> m_items;
};

#endif // UNCONNECTEDLINESCURVE_H
