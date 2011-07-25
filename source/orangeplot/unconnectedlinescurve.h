#ifndef UNCONNECTEDLINESCURVE_H
#define UNCONNECTEDLINESCURVE_H

#include "curve.h"
#include <QtGui/QPen>

class UnconnectedLinesCurve : public Curve
{

public:
    UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update_properties();
    
private:
    QList<QGraphicsLineItem*> m_items;
};

#endif // UNCONNECTEDLINESCURVE_H
