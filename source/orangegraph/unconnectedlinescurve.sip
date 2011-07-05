class UnconnectedLinesCurve : Curve {

%TypeHeaderCode
#include "unconnectedlinescurve.h"
%End

public:
    UnconnectedLinesCurve(const QList< double >& xData, const QList< double >& yData, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void updateProperties();
};