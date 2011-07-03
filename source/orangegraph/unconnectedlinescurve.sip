class UnconnectedLinesCurve : Curve {

%TypeHeaderCode
#include "unconnectedlinescurve.h"
%End

public:
    UnconnectedLinesCurve(QList< double > xData, QList< double > yData, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void updateProperties();
    
    void setPen(QPen pen);
    QPen pen() const;
};