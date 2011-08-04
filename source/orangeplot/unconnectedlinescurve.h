#ifndef UNCONNECTEDLINESCURVE_H
#define UNCONNECTEDLINESCURVE_H

#include "curve.h"

class UnconnectedLinesCurve : public Curve
{

public:
    UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update_properties();
    
private:
     QGraphicsPathItem* m_path_item;
};

#endif // UNCONNECTEDLINESCURVE_H
