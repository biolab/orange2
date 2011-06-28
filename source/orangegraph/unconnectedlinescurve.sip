class UnconnectedLinesCurve : Curve {

%TypeHeaderCode
#include "unconnectedlinescurve.h"
%End

public:
    UnconnectedLinesCurve(QList< double > xData, QList< double > yData, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update();
    
    void setPen(QPen pen);
    QPen pen() const;
};