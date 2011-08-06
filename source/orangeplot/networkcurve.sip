%ModuleCode
#include "networkcurve.h"
%End

class NodeItem : Point
{
public:
    NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent /TransferThis/ = 0);
    virtual ~NodeItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
    void set_coordinates(double x, double y);
    
    void set_x(double x);
    double x() const;
    
    void set_y(double y);
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
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    
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
	enum CircularLayoutType
	{
		circular_original = 0x01,
		circular_random = 0x02,
		circular_crossing = 0x03
	};
	
    typedef QList<EdgeItem*> Edges;
    typedef QMap<int, NodeItem*> Nodes;

    NetworkCurve(QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~NetworkCurve();
    
    virtual void update_properties();
    virtual QRectF data_rect() const;
    
    int random();
    int circular(NetworkCurve::CircularLayoutType type);
    int fr(int steps, bool weighted, bool smooth_cooling);

    Nodes nodes() const;
    void set_nodes(const Nodes& nodes);
    void add_nodes(const Nodes& nodes);
    void remove_node(int index);
    void remove_nodes(const QList< int >& nodes);
    
    Edges edges() const;
    void set_edges(const Edges& edges);
    void add_edges(const Edges& edges);
    
    QList<QPair<int, int> > edge_indices();

    void set_node_colors(const QMap<int, QColor*>& colors);    
    void set_node_sizes(const QMap<int, double>& sizes = QMap<int, double>(), double min_size=0, double max_size=0);
    void set_node_labels(const QMap<int, QString>& labels);
    void set_node_tooltips(const QMap<int, QString>& tooltips);
    void set_node_marks(const QMap<int, bool>& marks);
    void clear_node_marks();
    
    void set_edge_color(const QList<QColor*>& colors);
    
    void set_min_node_size(double size);
    double min_node_size() const;

    void set_max_node_size(double size);
    double max_node_size() const;
    
    void set_use_animations(bool use_animations);
    bool use_animations() const;
    
    void set_labels_on_marked_only(bool labels_on_marked_only);
    bool labels_on_marked_only();
    
    void stop_optimization();
};



