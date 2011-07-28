#ifndef NETWORKCURVE_H
#define NETWORKCURVE_H

#include "curve.h"
#include "point.h"
#include "plot.h"

class EdgeItem;

class NodeItem : public Point
{
public:
    NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent = 0);
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
    
    /**
     * @brief Connect an edge to this node
     * 
     * A connected edge is automatically updated whenever this node is moved
     *
     * @param edge the edge to be connected
     **/
    void add_connected_edge(EdgeItem* edge);
    void remove_connected_edge(EdgeItem* edge);
    QList<EdgeItem*> connected_edges();
    
    double m_size_value;

private:
    double m_x;
    double m_y;
    
    int m_index;
    QString m_label;
    int m_uuid;
    
    QList<EdgeItem*> m_connected_edges;
    QTransform m_graph_transform;
};

struct EdgeItem : public QGraphicsLineItem
{
public:
    enum Arrow
    {
        ArrowU = 0x01,
        ArrowV = 0x02
    };
    Q_DECLARE_FLAGS(Arrows, Arrow)
    
    EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
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
    
    void set_arrows(Arrows arrows);
    void set_arrow(Arrow arrow, bool enable);
    Arrows arrows();
    
private:
    Arrows m_arrows;
    NodeItem* m_u;
    NodeItem* m_v;
    int m_links_index;
    double m_weight;
    double m_size;
    QString m_label;
};

class NodeUpdater
{
public:
    NodeUpdater(const QTransform& t, double scale) : m_t(t), m_scale(scale) {}
    void operator()(NodeItem* item) 
    { 
        item->set_graph_transform(m_t); 
        item->setScale(m_scale);
    }
private:
    QTransform m_t;
    double m_scale;
};

class EdgeUpdater
{
public:
    EdgeUpdater(const QTransform& t, double scale) : m_t(t), m_scale(scale) {}
    void operator()(EdgeItem* item)
    {
        if (item->u() && item->v())
        {
            item->setLine(QLineF(item->u()->x(), item->u()->y(), item->v()->x(), item->v()->y()) * m_t);
            QPen p = item->pen();
            p.setWidthF(p.widthF() * m_scale);
            item->setPen(p);
        }
    }
private:
    QTransform m_t;
    double m_scale;
};

class NetworkCurve : public Curve
{
public:
    typedef QList<EdgeItem*> Edges;
    typedef QMap<int, NodeItem*> Nodes;

    NetworkCurve(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~NetworkCurve();

    virtual void update_properties();
    virtual QRectF data_rect() const;
    virtual void register_points();
    
    int fr(int steps, bool weighted);
    int random();
    
    void set_nodes(Nodes nodes);
    void remove_nodes(const QList<int> nodes);
    void set_edges(Edges edges);
    QList<QPair<int, int> > edge_indices();

    void set_node_colors(const QMap<int, QColor*> colors);
    void set_node_sizes(QMap<int, double> sizes=QMap<int, double>(), double min_size=0, double max_size=0);
    void set_edge_color(const QList<QColor*> colors);
    void set_node_labels(const QMap<int, QString> labels);
    void set_node_tooltips(const QMap<int, QString> tooltips);

    void set_min_node_size(double size);
    double min_node_size() const;

    void set_max_node_size(double size);
    double max_node_size() const;

    void set_use_animations(bool use_animations);
    bool use_animations() const;

    void stop_optimization();

private:
    Nodes m_nodes;
    Edges m_edges;

    double m_min_node_size;
    double m_max_node_size;
    bool m_use_animations;
    bool m_stop_optimization;
};

#endif // NETWORKCURVE_H
