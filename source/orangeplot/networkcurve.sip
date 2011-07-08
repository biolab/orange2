%ModuleCode
#include "networkcurve.h"
%End

struct NodeItem
{
    double x;
    double y;

    int index;
    bool marked;
    bool show;
    bool highlight;
    QString label;
    QString tooltip;
    int uuid;
    
    QPixmap* image;
    QPen pen;
    QColor nocolor;
    QColor color;
    int size;
    int style;
};

struct EdgeItem
{
    NodeItem* u;
    NodeItem* v;
    int links_index;
    bool arrowu;
    bool arrowv;
    double weight;
    QString label;
    QPen pen;
};

class NetworkCurve : Curve
{

%TypeHeaderCode
#include "networkcurve.h"
%End

public:
    typedef QMap<int, NodeItem> Nodes;
    typedef QList<EdgeItem> Edges;
    
    NetworkCurve(QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~NetworkCurve();
    
    virtual void updateProperties();
    
    virtual NetworkCurve::Edges get_edges() const = 0;
    virtual NetworkCurve::Nodes get_nodes() const = 0;

    virtual QRectF dataRect() const;
};