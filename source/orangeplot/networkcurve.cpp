#include "networkcurve.h"

#include <QtCore/QMap>
#include <QtCore/QList>

#include <QtCore/qmath.h>
#include <limits>
#include <QStyleOptionGraphicsItem>
#include <QCoreApplication>

/************/
/* NodeItem */
/************/

NodeItem::NodeItem(int index, int symbol, QColor color, int size, QGraphicsItem* parent): Point(symbol, color, size, parent)
{
    set_index(index);
    set_coordinates(((qreal)(qrand() % 1000)) * 1000, ((qreal)(qrand() % 1000)) * 1000);
    setZValue(0.5);
    m_size_value = 1;
    set_marked(false);
    set_selected(false);
    set_label("");
}

NodeItem::~NodeItem()
{

}
/*
void NodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    //painter->setClipRect(option->exposedRect);

    if (is_selected()) {
    	painter->setPen(QPen(Qt::yellow, 3));
    	painter->setBrush(color());
    	QRectF rect(-(size() + 4) / 2, -(size() + 4) / 2, size() + 4, size() + 4);
    	painter->drawEllipse(rect);
    } else if (is_marked()) {
    	painter->setPen(color());
    	painter->setBrush(color());
    	QRectF rect(-size() / 2, -size() / 2, size(), size());
    	painter->drawEllipse(rect);
    } else {
    	painter->setPen(color());
    	painter->setBrush(Qt::white);
    	QRectF rect(-size() / 2, -size() / 2, size(), size());
    	painter->drawEllipse(rect);
    }
}
*/
void NodeItem::set_coordinates(double x, double y)
{
    m_x = x;
    m_y = y;
    setPos(QPointF(m_x, m_y) * m_graph_transform);
}

void NodeItem::set_index(int index)
{
    m_index = index;
}

int NodeItem::index() const
{
    return m_index;
}

void NodeItem::set_graph_transform(const QTransform& transform)
{
    m_graph_transform = transform;
    set_coordinates(m_x, m_y);
}

QTransform NodeItem::graph_transform() const
{
    return m_graph_transform;
}

void NodeItem::set_x(double x)
{
    set_coordinates(x, m_y);
}

double NodeItem::x() const
{
    return m_x;
}

void NodeItem::set_y(double y)
{
    set_coordinates(m_x, y);
}

double NodeItem::y() const
{
    return m_y;
}

void NodeItem::set_label(const QString& label)
{
    m_label = label;
}

QString NodeItem::label() const
{
    return m_label;
}

void NodeItem::set_tooltip(const QString& tooltip)
{
    setToolTip(tooltip);
}

void NodeItem::set_uuid(int uuid)
{
    m_uuid = uuid;
}

int NodeItem::uuid() const
{
    return m_uuid;
}

void NodeItem::add_connected_edge(EdgeItem* edge)
{
    m_connected_edges << edge;
}

void NodeItem::remove_connected_edge(EdgeItem* edge)
{
    m_connected_edges.removeAll(edge);
}

/************/
/* EdgeItem */
/************/

EdgeItem::EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent, QGraphicsScene* scene): QGraphicsLineItem(parent, scene)
{
    set_u(u);
    set_v(v);
    m_size = 1;
    QPen p = pen();
	p.setWidthF(m_size);
	setPen(p);
	setZValue(0);
}

EdgeItem::~EdgeItem()
{

}

void EdgeItem::set_u(NodeItem* item)
{
    m_u = item;
}

NodeItem* EdgeItem::u()
{
    return m_u;
}

void EdgeItem::set_v(NodeItem* item)
{
    m_v = item;
}

NodeItem* EdgeItem::v()
{
    return m_v;
}

void EdgeItem::set_tooltip(const QString& tooltip)
{
    setToolTip(tooltip);
}

void EdgeItem::set_label(const QString& label)
{
    m_label = label;
}

QString EdgeItem::label() const
{
    return m_label;
}

void EdgeItem::set_links_index(int index)
{
    m_links_index = index;
}

int EdgeItem::links_index() const
{
    return m_links_index;
}

void EdgeItem::set_arrow(EdgeItem::Arrow arrow, bool enable)
{
    if (enable)
    {
        set_arrows(arrows() | arrow);
    }
    else
    {
        set_arrows(arrows() & ~arrow);
    }
}

EdgeItem::Arrows EdgeItem::arrows()
{
    return m_arrows;
}

void EdgeItem::set_arrows(EdgeItem::Arrows arrows)
{
    m_arrows = arrows;
    // TODO: Update the QGraphicsItem element, add arrows
}

void EdgeItem::set_weight(double weight)
{
    m_weight = weight;
}

double EdgeItem::weight() const
{
    return m_weight;
}

/****************/
/* NetworkCurve */
/****************/

NetworkCurve::NetworkCurve(QGraphicsItem* parent, QGraphicsScene* scene): Curve(parent, scene)
{
	 m_min_node_size = 5;
	 m_max_node_size = 5;
}

NetworkCurve::~NetworkCurve()
{

}

void NetworkCurve::update_properties()
{
    const QTransform t = graph_transform();
    const double scale = 1.0/zoom_factor();
    update_items(m_nodes, NodeUpdater(t, scale), UpdatePosition);
    update_items(m_edges, EdgeUpdater(t, scale), UpdatePen);
}

QRectF NetworkCurve::data_rect() const
{
    QRectF r;
    bool first = true;
    foreach (const NodeItem* node, m_nodes)
    {
        if (first)
        {
            r = QRectF(node->x(), node->y(), 0, 0);
            first = false;
        }
        else
        {
            r.setTop( qMin(r.top(), node->y()) );
            r.setBottom( qMax(r.bottom(), node->y()) );
            r.setLeft( qMin(r.left(), node->x()) );
            r.setRight( qMax(r.right(), node->x()) );
        }
    }
    qDebug() << "NetworkCurve::dataRect()" << r;
    return r;
}

int NetworkCurve::random()
{
	Nodes::ConstIterator uit = m_nodes.constBegin();
	Nodes::ConstIterator uend = m_nodes.constEnd();

	for (; uit != uend; ++uit)
	{
		uit.value()->set_coordinates(((qreal)(qrand() % 1000)) * 1000, ((qreal)(qrand() % 1000)) * 1000);
	}

	return 0;
}

int NetworkCurve::fr(int steps, bool weighted)
{
	int i, j;
	int count = 0;
	NodeItem *u, *v;
	EdgeItem *edge;
	m_stop_optimization = false;

	double rect[4] = {std::numeric_limits<double>::max(),
			  std::numeric_limits<double>::max(),
			  std::numeric_limits<double>::min(),
			  std::numeric_limits<double>::min()};

	QMap<int, DataPoint> disp;
	foreach (const NodeItem* node, m_nodes)
	{
		DataPoint point;
		point.x = 0;
		point.y = 0;
		disp[node->index()] = point;

		double x = node->x();
		double y = node->y();
		if (rect[0] > x) rect[0] = x;
		if (rect[1] > y) rect[1] = y;
		if (rect[2] < x) rect[2] = x;
		if (rect[3] < y) rect[3] = y;
	}
	QRectF data_r(rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]);
	double area = data_r.width() * data_r.height();
	int updateCheckpoint = steps / 50;
	if (updateCheckpoint == 0 || updateCheckpoint % 2 != 0)
	{
		updateCheckpoint += 1;
	}
	qDebug() << "updateCheckpoint " << updateCheckpoint;
	double k2 = area / m_nodes.size();
	double k = sqrt(k2);
	double kk = 2 * k;
	double kk2 = kk * kk;

	double temperature, cooling, cooling_switch, cooling_1, cooling_2;
	temperature = sqrt(area) / 5;
	cooling = exp(log(k / 10 / temperature) / steps);
	if (steps > 20)
	{
		cooling_switch = sqrt(area) / 100;
		cooling_1 = (temperature - cooling_switch) / 20;
		cooling_2 = (cooling_switch - sqrt(area) / 2000 ) / (steps - 20);
	}
	else
	{
		cooling_switch = sqrt(area) / 1000;
		cooling_1 = (temperature - cooling_switch) / steps;
		cooling_2 = 0;
	}

	// iterations
	for (i = 0; i < steps; i++)
	{
		foreach (const NodeItem* node, m_nodes)
		{
			disp[node->index()].x = 0;
			disp[node->index()].y = 0;
		}

		// calculate repulsive force
		Nodes::ConstIterator uit = m_nodes.constBegin();
		Nodes::ConstIterator uend = m_nodes.constEnd();
		for (; uit != uend; ++uit)
		{
			u = uit.value();
			Nodes::ConstIterator vit(uit);
			++vit;
			for (; vit != uend; ++vit)
			{
				v = vit.value();

				double difx = u->x() - v->x();
				double dify = u->y() - v->y();

				double dif2 = difx * difx + dify * dify;

				if (dif2 < kk2)
				{
					if (dif2 == 0)
						dif2 = 1;

					double dX = difx * k2 / dif2;
					double dY = dify * k2 / dif2;

					disp[u->index()].x += dX;
					disp[u->index()].y += dY;

					disp[v->index()].x -= dX;
					disp[v->index()].y -= dY;
				}
			}
		}

		// calculate attractive forces
		for (j = 0; j < m_edges.size(); ++j)
		{
			edge = m_edges[j];
			double difx = edge->u()->x() - edge->v()->x();
			double dify = edge->u()->y() - edge->v()->y();

			double dif = sqrt(difx * difx + dify * dify);

			double dX = difx * dif / k;
			double dY = dify * dif / k;

			if (weighted) {
				dX *= edge->weight();
				dY *= edge->weight();
			}

			disp[edge->u()->index()].x -= dX;
			disp[edge->u()->index()].y -= dY;

			disp[edge->v()->index()].x += dX;
			disp[edge->v()->index()].y += dY;
		}

		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		Nodes::Iterator nit = m_nodes.begin();
		for (; nit != m_nodes.end(); ++nit)
		{
			u = nit.value();
			double dif = sqrt(disp[u->index()].x * disp[u->index()].x + disp[u->index()].y * disp[u->index()].y);

			if (dif == 0)
				dif = 1;

			u->set_coordinates(u->x() + (disp[u->index()].x * qMin(fabs(disp[u->index()].x), temperature) / dif),
			                   u->y() + (disp[u->index()].y * qMin(fabs(disp[u->index()].y), temperature) / dif));
		}

		if (++count % updateCheckpoint == 0)
		{
			update_properties();
			QCoreApplication::processEvents();
		}

		if (m_stop_optimization)
		{
			return 0;
		}

		//temperature = temperature * cooling;
		if (floor(temperature) > cooling_switch)
		{
			temperature -= cooling_1;
		}
		else
		{
			temperature -= cooling_2;
		}
	}

	return 0;
}

void NetworkCurve::set_edges(NetworkCurve::Edges edges)
{
    qDeleteAll(m_edges);
    m_edges = edges;
}

QList<QPair<int, int> > NetworkCurve::edge_indices()
{
	int i;
	EdgeItem *e;
	QList<QPair<int, int> > edge_indices;

	for (i = 0; i < m_edges.size(); ++i)
	{
		e = m_edges[i];
		edge_indices.append(QPair<int, int>(e->u()->index(), e->v()->index()));
	}

	return edge_indices;
}

void NetworkCurve::set_nodes(NetworkCurve::Nodes nodes)
{
    qDeleteAll(m_nodes);
    plot()->remove_all_points(this);
    
    m_nodes = nodes;
    foreach (NodeItem* node, nodes)
    {
        DataPoint p;
        p.x = node->x();
        p.y = node->y();
        plot()->add_point(p, node, this);
    }
}

void NetworkCurve::set_node_color(const QMap<int, QColor*> colors)
{
	QMap<int, QColor*>::ConstIterator it;
	for (it = colors.constBegin(); it != colors.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_color(*it.value());
	}
}

void NetworkCurve::set_edge_color(const QList<QColor*> colors)
{
	int i;
	for (i = 0; i < colors.size(); ++i)
	{
		QPen p = m_edges[i]->pen();
		p.setColor(*colors[i]);
		m_edges[i]->setPen(p);
	}
}

void NetworkCurve::set_node_size(QMap<int, double> sizes, double min_size, double max_size)
{
	// TODO inverted
	NodeItem* node;
	Nodes::ConstIterator nit;

	double min_size_value = std::numeric_limits<double>::max();
	double max_size_value = std::numeric_limits<double>::min();

	QMap<int, double>::Iterator it;
	for (it = sizes.begin(); it != sizes.end(); ++it)
	{
		m_nodes[it.key()]->m_size_value = it.value();

		if (it.value() < min_size_value)
		{
			min_size_value = it.value();
		}

		if (it.value() > max_size_value)
		{
			max_size_value = it.value();
		}
	}

	// find min and max size value in nodes dict
	bool min_changed = true;
	bool max_changed = true;
	for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
	{
		node = nit.value();

		if (node->m_size_value < min_size_value)
		{
			min_size_value = node->m_size_value;
			min_changed = false;
		}

		if (node->m_size_value > max_size_value)
		{
			max_size_value = node->m_size_value;
			max_changed = false;
		}
	}

	double size_span = max_size_value - min_size_value;


	if (min_size > 0 || max_size > 0 || min_changed || max_changed)
	{
		if (min_size > 0)
		{
			m_min_node_size = min_size;
		}

		if (max_size > 0)
		{
			m_max_node_size = max_size;
		}

		double node_size_span = m_max_node_size - m_min_node_size;
		// recalibrate all
		if (size_span > 0)
		{
			for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
			{
				node = nit.value();
				node->set_size((node->m_size_value - min_size_value) / size_span * node_size_span + m_min_node_size);
			}
		}
		else
		{
			for (nit = m_nodes.constBegin(); nit != m_nodes.constEnd(); ++nit)
			{
				node = nit.value();
				node->set_size(m_min_node_size);
			}
		}
	}
	else if (sizes.size() > 0)
	{
		double node_size_span = m_max_node_size - m_min_node_size;
		// recalibrate given
		if (size_span > 0)
		{
			for (it = sizes.begin(); it != sizes.end(); ++it)
			{
				node = m_nodes[it.key()];
				node->set_size((node->m_size_value - min_size_value) / size_span * node_size_span + m_min_node_size);
			}
		}
		else
		{
			for (it = sizes.begin(); it != sizes.end(); ++it)
			{
				node = m_nodes[it.key()];
				node->set_size(m_min_node_size);
			}
		}
	}
}

void NetworkCurve::set_min_node_size(double size)
{
	set_node_size(QMap<int, double>(), size, 0);
}

double NetworkCurve::min_node_size() const
{
	return m_min_node_size;
}

void NetworkCurve::set_max_node_size(double size)
{
	set_node_size(QMap<int, double>(), 0, size);
}

double NetworkCurve::max_node_size() const
{
	return m_max_node_size;
}

void NetworkCurve::register_points()
{
    Plot* p = plot();
    if (p)
    {
        p->remove_all_points(this);
        const Data d = data();
        const int n = d.size();
        for (int i = 0; i < n; ++i)
        {
            p->add_point(d[i], m_nodes[i], this);
        }
    }
}

void NetworkCurve::set_use_animations(bool use_animations)
{
    m_use_animations = use_animations;
}

bool NetworkCurve::use_animations() const
{
    return m_use_animations;
}
 
void NetworkCurve::stop_optimization()
{
    m_stop_optimization = true;
}
