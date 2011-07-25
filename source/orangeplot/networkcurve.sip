%ModuleCode
#include "networkcurve.h"
%End

class NodeItem : Point
{
public:
    NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~NodeItem();
    
    void set_coordinates(double x, double y);
    void set_x(double x);
    void set_y(double y);
    double x() const;
    double y() const;
    
    void set_graph_transform(const QTransform& transform);
    QTransform graph_transform() const;
    
    void set_index(int index);
    int index() const;
    
    void set_label(const QString& label);
    QString label() const;
    void set_tooltip(const QString& tooltip);
    
    void set_uuid(int uuid);
    int uuid() const;
    
    void add_connected_edge(EdgeItem* edge);
    void remove_connected_edge(EdgeItem* edge);
};

class EdgeItem : QGraphicsLineItem
{
public:
    enum Arrow
    {
        ArrowU = 0x01,
        ArrowV = 0x02
    };
    typedef QFlags<EdgeItem::Arrow> Arrows;
    
    EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~EdgeItem();
    
    void set_u(NodeItem* item);
    NodeItem* u();
    void set_v(NodeItem* item);
    NodeItem* v();
    
    void set_label(const QString& label);
    QString label() const;
    void set_tooltip(const QString& tooltip);
    
    void set_links_index(int index);
    int links_index() const;
    
    void set_weight(double weight);
    double weight() const;
    
    void set_arrows(EdgeItem::Arrows arrows);
    void set_arrow(EdgeItem::Arrow arrow, bool enable);
};

class NetworkCurve : Curve
{
public:
    typedef QList<EdgeItem*> Edges /NoTypeName/;
    typedef QMap<int, NodeItem*> Nodes /NoTypeName/;

    NetworkCurve(QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~NetworkCurve();
    
    virtual void update_properties();
    virtual QRectF data_rect() const;
    
    int fr(int steps, bool weighted, double temperature, double cooling);
    int random();
    
    void set_nodes(QMap<int, NodeItem*> nodes);
    void set_edges(Edges edges);
};

