class OWGraph : QGraphicsView {

%TypeHeaderCode
#include "owgraph.h"
%End

public:
    OWGraph(QWidget* parent /TransferThis/ = 0);
    virtual ~OWGraph();
};