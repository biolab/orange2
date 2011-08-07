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
    setAcceptHoverEvents(true);
    set_transparent(false);
}

NodeItem::~NodeItem()
{
}

void NodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	Point::paint(painter, option, widget);

	NetworkCurve *curve = (NetworkCurve*)parentItem();
	bool on_marked_only = curve->labels_on_marked_only();

	if (m_label.compare("") != 0 && (!on_marked_only || is_marked()))
	{
		QFontMetrics metrics = painter->fontMetrics();
		int th = metrics.height();
		int tw = metrics.width(m_label);
		QRect r(-tw/2, 0, tw, th);
		//painter->fillRect(r, QBrush(Qt::white));
		painter->drawText(r, Qt::AlignHCenter, m_label);
	}
}

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
    if (!m_connected_edges.contains(edge))
    {
        m_connected_edges << edge;
    }
}

void NodeItem::remove_connected_edge(EdgeItem* edge)
{
    m_connected_edges.removeAll(edge);
}

QList<EdgeItem*> NodeItem::connected_edges()
{
	return m_connected_edges;
}

QList<NodeItem*> NodeItem::neighbors()
{
	QList<NodeItem*> neighbors;

	EdgeItem *e;
	QList<EdgeItem*> edges = connected_edges();
	int size = edges.size();
	foreach(e, edges)
	{
		if (e->u()->index() == index())
		{
			neighbors.append(e->v());
		}
		else
		{
			neighbors.append(e->u());
		}
	}

	return neighbors;
}

/************/
/* EdgeItem */
/************/

EdgeItem::EdgeItem(NodeItem* u, NodeItem* v, QGraphicsItem* parent, QGraphicsScene* scene): QGraphicsLineItem(parent, scene),
m_u(0), m_v(0)
{
    set_u(u);
    set_v(v);
    m_size = 1;
    QPen p = pen();
	p.setWidthF(m_size);
    p.setCosmetic(true);
	setPen(p);
	setZValue(0);
}

EdgeItem::~EdgeItem()
{
    if (m_u)
    {
        m_u->remove_connected_edge(this);
    }
    if (m_v)
    {
        m_v->remove_connected_edge(this);
    }
}


void EdgeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	painter->setRenderHint(QPainter::Antialiasing, false);
	QGraphicsLineItem::paint(painter, option, widget);
}


void EdgeItem::set_u(NodeItem* item)
{
    if (m_u)
    {
        m_u->remove_connected_edge(this);
    }
    if (item)
    {
        item->add_connected_edge(this);
    }
    m_u = item;
}

NodeItem* EdgeItem::u()
{
    return m_u;
}

void EdgeItem::set_v(NodeItem* item)
{
    if (m_v)
    {
        m_v->remove_connected_edge(this);
    }
    if (item)
    {
        item->add_connected_edge(this);
    }
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
    cancelAllUpdates();
    qDeleteAll(m_edges);
    m_edges.clear();
    qDeleteAll(m_nodes);
    m_nodes.clear();
}

void NetworkCurve::update_properties()
{
    const QTransform t = graph_transform();
    update_items(m_nodes, NodeUpdater(t, point_transform()), UpdatePosition);
    update_items(m_edges, EdgeUpdater(t), UpdatePen);
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

#define PI 3.14159265

int NetworkCurve::circular(CircularLayoutType type)
{
	// type
	// 0 - original
	// 1 - random
	// 2 - crossing reduction

	if (type == NetworkCurve::circular_crossing)
	{
		qDebug() << "crossing_reduction";
		return circular_crossing_reduction();
	}

	if (type == NetworkCurve::circular_original)
		qDebug() << "original";

	if (type == NetworkCurve::circular_random)
			qDebug() << "random";

	QRectF rect = data_rect();
	int xCenter = rect.width() / 2;
	int yCenter = rect.height() / 2;
	int r = (rect.width() < rect.height()) ? rect.width() * 0.38 : rect.height() * 0.38;

	int i;
	double fi = PI;
	double step = 2 * PI / m_nodes.size();

	srand(time(NULL));
	std::vector<int> vertices;
	Nodes::ConstIterator it;
	for (it = m_nodes.constBegin(); it != m_nodes.constEnd(); ++it)
	{
		vertices.push_back(it.key());
	}

	for (i = 0; i < m_nodes.size(); ++i)
	{
		if (type == NetworkCurve::circular_original)
		{
			m_nodes[vertices[i]]->set_x(r * cos(fi) + xCenter);
			m_nodes[vertices[i]]->set_y(r * sin(fi) + yCenter);
		}
		else if (type == NetworkCurve::circular_random)
		{
			int ndx = rand() % vertices.size();
			m_nodes[vertices[ndx]]->set_x(r * cos(fi) + xCenter);
			m_nodes[vertices[ndx]]->set_y(r * sin(fi) + yCenter);
			vertices.erase(vertices.begin() + ndx);
		}
		fi = fi - step;
	}
	return 0;
}


int NetworkCurve::circular_crossing_reduction()
{
	QMap<int, QueueVertex*> qvertices;
	std::vector<QueueVertex*> vertices;
	std::vector<QueueVertex*> original;

	Nodes::ConstIterator it;
	for (it = m_nodes.constBegin(); it != m_nodes.constEnd(); ++it)
	{
		QueueVertex *vertex = new QueueVertex();
		vertex->ndx = it.key();
		qvertices[it.key()] = vertex;

		std::vector<int> neighbours;
		vertex->unplacedNeighbours = neighbours.size();
		vertex->neighbours = neighbours;
		vertices.push_back(vertex);
	}
	int i;
	EdgeItem *edge;
	for (i = 0; i < m_edges.size(); i++)
	{
		edge = m_edges[i];
		int u = edge->u()->index();
		int v = edge->v()->index();
		qvertices[u]->neighbours.push_back(v);
		qvertices[u]->unplacedNeighbours += 1;
		qvertices[v]->neighbours.push_back(u);
		qvertices[v]->unplacedNeighbours += 1;
	}
	original.assign(vertices.begin(), vertices.end());

	std::deque<int> positions;
	while (vertices.size() > 0)
	{
		std::sort(vertices.begin(), vertices.end(), QueueVertex());
		QueueVertex *vertex = vertices.back();


		// update neighbours
		for (i = 0; i < vertex->neighbours.size(); ++i)
		{
			int ndx = vertex->neighbours[i];

			original[ndx]->placedNeighbours++;
			original[ndx]->unplacedNeighbours--;
		}

		// count left & right crossings
		if (vertex->placedNeighbours > 0)
		{
			int left = 0;
			std::vector<int> lCrossings;
			std::vector<int> rCrossings;
			for (i = 0; i < positions.size(); ++i)
			{
				int ndx = positions[i];

				if (vertex->hasNeighbour(ndx))
				{
					lCrossings.push_back(left);
					left += original[ndx]->unplacedNeighbours;
					rCrossings.push_back(left);
				}
				else
					left += original[ndx]->unplacedNeighbours;
			}

			int leftCrossings = 0;
			int rightCrossings = 0;

			for (i = 0; i < lCrossings.size(); --i)
				leftCrossings += lCrossings[i];

			rCrossings.push_back(left);
			for (i = rCrossings.size() - 1; i > 0 ; --i)
				rightCrossings += rCrossings[i] - rCrossings[i - 1];
			//cout << "left: " << leftCrossings << " right: " <<rightCrossings << endl;
			if (leftCrossings < rightCrossings)
				positions.push_front(vertex->ndx);
			else
				positions.push_back(vertex->ndx);

		}
		else
			positions.push_back(vertex->ndx);

		vertices.pop_back();
	}

	// Circular sifting
	for (i = 0; i < positions.size(); ++i)
		original[positions[i]]->position = i;

	int step;
	for (step = 0; step < 5; ++step)
	{
		for (i = 0; i < m_nodes.size(); ++i)
		{
			bool stop = false;
			int switchNdx = -1;
			QueueVertex *u = original[positions[i]];
			int vNdx = (i + 1) % m_nodes.size();

			while (!stop)
			{
				QueueVertex *v = original[positions[vNdx]];

				int midCrossings = u->neighbours.size() * v->neighbours.size() / 2;
				int crossings = 0;
				int j,k;
				for (j = 0; j < u->neighbours.size(); ++j)
					for (k = 0; k < v->neighbours.size(); ++k)
						if ((original[u->neighbours[j]]->position == v->position) || (original[v->neighbours[k]]->position == u->position))
							midCrossings = (u->neighbours.size() - 1) * (v->neighbours.size() - 1) / 2;
						else if ((original[u->neighbours[j]]->position + m_nodes.size() - u->position) % m_nodes.size() < (original[v->neighbours[k]]->position + m_nodes.size() - u->position) % m_nodes.size())
							++crossings;

				//cout << "v: " <<  v->ndx << " crossings: " << crossings << " u.n.size: " << u->neighbours.size() << " v.n.size: " << v->neighbours.size() << " mid: " << midCrossings << endl;
				if (crossings > midCrossings)
					switchNdx = vNdx;
				else
					stop = true;

				vNdx = (vNdx + 1) % m_nodes.size();
			}
			int j;
			if (switchNdx > -1)
			{
				//cout << "u: " << u->ndx << " switch: " << original[switchNdx]->ndx << endl << endl;
				positions.erase(positions.begin() + i);
				positions.insert(positions.begin() + switchNdx, u->ndx);

				for (j = i; j <= switchNdx; ++j)
					original[positions[j]]->position = j;
			}
			//else
			//	cout << "u: " << u->ndx << " switch: " << switchNdx << endl;
		}
	}

	QRectF rect = data_rect();
	int xCenter = rect.width() / 2;
	int yCenter = rect.height() / 2;
	int r = (rect.width() < rect.height()) ? rect.width() * 0.38 : rect.height() * 0.38;
	double fi = PI;
	double fiStep = 2 * PI / m_nodes.size();

	for (i = 0; i < m_nodes.size(); ++i)
	{
		m_nodes[positions[i]]->set_x(r * cos(fi) + xCenter);
		m_nodes[positions[i]]->set_y(r * sin(fi) + yCenter);
		fi = fi - fiStep;
	}

        qDeleteAll(original);

	original.clear();
	vertices.clear();
	qvertices.clear();
	return 0;
}

int NetworkCurve::fr(int steps, bool weighted, bool smooth_cooling)
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
	foreach (const NodeItem*   node, m_nodes)
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
	double jitter = sqrt(area) / 2000;
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

	if (smooth_cooling)
	{
		if (steps < 20)
		{
			steps = 20;
		}
		temperature = cooling_switch;
		cooling_1 = 0;
		cooling_2 = (cooling_switch - sqrt(area) / 2000 ) / steps;
	}

	// iterations
	for (i = 0; i < steps; ++i)
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

				// if nodes are close, apply repulsive force
				if (dif2 < kk2)
				{
					// if nodes overlap
					if (dif2 == 0)
					{
						dif2 = 1 / k;
						u->set_x(u->x() + jitter);
						u->set_y(u->y() + jitter);
						v->set_x(v->x() - jitter);
						v->set_y(v->y() - jitter);
					}

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
            Plot* p = plot();
            if (p)
            {
                p->set_dirty();
                p->replot();
            }
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

void NetworkCurve::set_edges(const NetworkCurve::Edges& edges)
{
    cancelAllUpdates();
    qDeleteAll(m_edges);
    m_edges = edges;
}

NetworkCurve::Edges NetworkCurve::edges() const
{
    return m_edges;
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

void NetworkCurve::set_nodes(const NetworkCurve::Nodes& nodes)
{
    cancelAllUpdates();
    qDeleteAll(m_edges);
    m_edges.clear();
    qDeleteAll(m_nodes);
    m_nodes = nodes;
    Q_ASSERT(m_nodes.uniqueKeys() == m_nodes.keys());
    register_points();
}

NetworkCurve::Nodes NetworkCurve::nodes() const
{
    return m_nodes;
}

void NetworkCurve::remove_nodes(const QList<int>& nodes)
{
    cancelAllUpdates();
    foreach (int i, nodes)
    {
        remove_node(i);
    }
    
}

void NetworkCurve::remove_node(int index)
{
    cancelAllUpdates();
    if (!m_nodes.contains(index))
    {
        qWarning() << "Trying to remove node" << index << "which is not in the network";
        return;
    }
    NodeItem* node = m_nodes.take(index);
    Q_ASSERT(node->index() == index);
    Plot* p = plot();
    if (p)
    {
        DataPoint d;
        d.x = node->x();
        d.y = node->y();
        p->remove_point(d, this);
    }
    
    foreach (EdgeItem* edge, node->connected_edges())
    {
        m_edges.removeOne(edge);
        delete edge;
    }
    Q_ASSERT(node->connected_edges().isEmpty());
    delete node;
}

void NetworkCurve::add_edges(const NetworkCurve::Edges& edges)
{
    m_edges.append(edges);
}

void NetworkCurve::add_nodes(const NetworkCurve::Nodes& nodes)
{
    Nodes::ConstIterator it = nodes.constBegin();
    Nodes::ConstIterator end = nodes.constEnd();
    QList<int> indices;
	for (it; it != end; ++it)
	{
		indices.append(it.key());

		if (m_nodes.contains(it.key()))
		{
			remove_node(it.key());
		}
	}

	m_nodes.unite(nodes);
    Q_ASSERT(m_nodes.uniqueKeys() == m_nodes.keys());
	register_points();
}

void NetworkCurve::set_node_colors(const QMap<int, QColor*>& colors)
{
	QMap<int, QColor*>::ConstIterator it;
	for (it = colors.constBegin(); it != colors.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_color(*it.value());
	}
}

void NetworkCurve::set_node_sizes(const QMap<int, double>& sizes, double min_size, double max_size)
{
    cancelAllUpdates();
	// TODO inverted
	NodeItem* node;
	Nodes::ConstIterator nit;

	double min_size_value = std::numeric_limits<double>::max();
	double max_size_value = std::numeric_limits<double>::min();

	QMap<int, double>::ConstIterator it;
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

void NetworkCurve::set_node_labels(const QMap<int, QString>& labels)
{
    cancelAllUpdates();
	QMap<int, QString>::ConstIterator it;
	for (it = labels.constBegin(); it != labels.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_label(it.value());
	}
}

void NetworkCurve::set_node_tooltips(const QMap<int, QString>& tooltips)
{
    cancelAllUpdates();
	QMap<int, QString>::ConstIterator it;
	for (it = tooltips.constBegin(); it != tooltips.constEnd(); ++it)
	{
		m_nodes[it.key()]->setToolTip(it.value());
	}
}

void NetworkCurve::set_node_marks(const QMap<int, bool>& marks)
{
	cancelAllUpdates();
	QMap<int, bool>::ConstIterator it;
	for (it = marks.constBegin(); it != marks.constEnd(); ++it)
	{
		m_nodes[it.key()]->set_marked(it.value());
	}
}

void NetworkCurve::clear_node_marks()
{
	cancelAllUpdates();
	Nodes::Iterator it;
	for (it = m_nodes.begin(); it != m_nodes.end(); ++it)
	{
		it.value()->set_marked(false);
	}
}

void NetworkCurve::set_edge_color(const QList<QColor*>& colors)
{
    cancelAllUpdates();
	int i;
	for (i = 0; i < colors.size(); ++i)
	{
		QPen p = m_edges[i]->pen();
		p.setColor(*colors[i]);
		m_edges[i]->setPen(p);
	}
}

void NetworkCurve::set_min_node_size(double size)
{
	set_node_sizes(QMap<int, double>(), size, 0);
}

double NetworkCurve::min_node_size() const
{
	return m_min_node_size;
}

void NetworkCurve::set_max_node_size(double size)
{
	set_node_sizes(QMap<int, double>(), 0, size);
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
        foreach (NodeItem* node, m_nodes)
        {
            DataPoint d;
            d.x = node->x();
            d.y = node->y();
            p->add_point(d, node, this);
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

void NetworkCurve::set_labels_on_marked_only(bool labels_on_marked_only)
{
	m_labels_on_marked_only = labels_on_marked_only;
}

bool NetworkCurve::labels_on_marked_only()
{
	return m_labels_on_marked_only;
}
 
void NetworkCurve::stop_optimization()
{
    m_stop_optimization = true;
}
