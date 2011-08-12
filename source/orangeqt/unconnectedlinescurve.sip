class UnconnectedLinesCurve : Curve {

%TypeHeaderCode
#include "unconnectedlinescurve.h"
%End

public:
    UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~UnconnectedLinesCurve();
    
    virtual void update_properties();
};