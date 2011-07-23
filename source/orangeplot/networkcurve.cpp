#include "networkcurve.h"

#include <QtCore/QMap>
#include <QtCore/QList>

NetworkCurve::NetworkCurve(QGraphicsItem* parent, QGraphicsScene* scene): Curve(parent, scene)
{

}

NetworkCurve::~NetworkCurve()
{

}

void NetworkCurve::updateProperties()
{
    const Data d = data();
    const QTransform t = graphTransform();
    int m, n;
    
    const Nodes nodes = get_nodes();
    
    if (m_vertex_items.keys() != nodes.keys())
    {
        qDeleteAll(m_vertex_items);
        m_vertex_items.clear();
        Nodes::ConstIterator it = nodes.constBegin();
        Nodes::ConstIterator end = nodes.constEnd();
        for (; it != end; ++it)
        {
            m_vertex_items.insert(it.key(), new QGraphicsPathItem(this));
        }
    }
    
    NodeItem node;
    QGraphicsPathItem* item;
    Nodes::ConstIterator nit = nodes.constBegin();
    Nodes::ConstIterator nend = nodes.constEnd();
    for (; nit != nend; ++nit)
    {
        node = nit.value();
        item = m_vertex_items[nit.key()];
        item->setPos( t.map(QPointF(node.x, node.y)) );
        item->setBrush(brush());
        item->setPen(node.pen);
        item->setToolTip(node.tooltip);
        item->setPath(pathForSymbol(node.style, node.size));
    }
    
    Q_ASSERT(m_vertex_items.size() == nodes.size());
    
    const Edges edges = get_edges();
    
    n = edges.size();
    m = m_edge_items.size();
    
    for (int i = n; i < m; ++i)
    {
        delete m_edge_items.takeLast();
    }
    
    for (int i = m; i < n; ++i)
    {
        m_edge_items << new QGraphicsLineItem(this);
    }
    
    Q_ASSERT(m_edge_items.size() == edges.size());
    
    QLineF line;
    QGraphicsLineItem* line_item;
    n = edges.size();
    for (int i = 0; i < n; ++i)
    {
        EdgeItem edge = edges[i];
        node = nodes[edge.u->index];
        line.setP1(QPointF(node.x, node.y));
        node = nodes[edge.v->index];
        line.setP2(QPointF(node.x, node.y));
        line_item = m_edge_items[i];
        line_item->setLine( t.map(line) );
        line_item->setPen(edges[i].pen);
    }
}

QRectF NetworkCurve::dataRect() const
{
    QRectF r;
    bool first = true;
    foreach (const NodeItem& node, get_nodes())
    {
        if (first)
        {
            r = QRectF(node.x, node.y, 0, 0);
            first = false;
        }
        else
        {
            r.setTop( qMin(r.top(), node.y) );
            r.setBottom( qMax(r.bottom(), node.y) );
            r.setLeft( qMin(r.left(), node.x) );
            r.setRight( qMax(r.right(), node.y) );
        }
    }
    qDebug() << "NetworkCurve::dataRect()" << r;
    return r;
}

int NetworkCurve::fr(int steps, bool weighted, double temperature, double cooling)
{
	int i, j;
	NodeItem u, v;
	EdgeItem edge;
	Nodes nodes = get_nodes();
	Edges edges = get_edges();
	QRectF data_rect = dataRect();

	QMap<int, DataPoint> disp;
	foreach (const NodeItem& node, get_nodes())
	{
		DataPoint point;
		point.x = 0;
		point.y = 0;
		disp[node.index] = point;
	}

	qreal area = data_rect.width() * data_rect.height();

	qreal k2 = area / nodes.size();
	qreal k = sqrt(k2);
	qreal kk = 2 * k;
	qreal kk2 = kk * kk;

	// iterations
	for (i = 0; i < steps; i++)
	{
		DataPoint tmp_point;
		QMap<int, DataPoint>::ConstIterator qit = disp.constBegin();
		QMap<int, DataPoint>::ConstIterator qend = disp.constEnd();
		for (; qit != qend; ++qit)
		{
			tmp_point = qit.value();
			tmp_point.x = 0;
			tmp_point.y = 0;
		}

		// calculate repulsive force
		Nodes::ConstIterator uit = nodes.constBegin();
		Nodes::ConstIterator uend = nodes.constEnd();
		for (; uit != uend; ++uit)
		{
			u = uit.value();
			Nodes::ConstIterator vit(uit);
			++vit;
			for (; vit != uend; ++vit)
			{
				v = vit.value();
				qreal difx = u.x - v.x;
				qreal dify = u.y - v.y;

				qreal dif2 = difx * difx + dify * dify;

				if (dif2 < kk2)
				{
					if (dif2 == 0)
						dif2 = 1;

					qreal dX = difx * k2 / dif2;
					qreal dY = dify * k2 / dif2;

					disp[u.index].x = disp[u.index].x + dX;
					disp[u.index].y = disp[u.index].y + dY;

					disp[v.index].x = disp[v.index].x - dX;
					disp[v.index].y = disp[v.index].y - dY;
				}
			}
		}

		// calculate attractive forces
		if (weighted)
		{
			for (j = 0; j < edges.size(); ++j)
			{
				edge = edges[j];
				qreal difx = edge.u->x - edge.v->x;
				qreal dify = edge.u->y - edge.v->y;

				qreal dif = sqrt(difx * difx + dify * dify);

				qreal dX = difx * dif / k * edge.weight;
				qreal dY = dify * dif / k * edge.weight;

				disp[edge.u->index].x = disp[edge.u->index].x + dX;
				disp[edge.u->index].y = disp[edge.u->index].y + dY;

				disp[edge.v->index].x = disp[edge.v->index].x - dX;
				disp[edge.v->index].y = disp[edge.v->index].y - dY;
			}
		}
		else
		{
			for (j = 0; j < edges.size(); ++j)
			{
				edge = edges[j];
				qreal difx = edge.u->x - edge.v->x;
				qreal dify = edge.u->y - edge.v->y;

				qreal dif = sqrt(difx * difx + dify * dify);

				qreal dX = difx * dif / k;
				qreal dY = dify * dif / k;

				disp[edge.u->index].x = disp[edge.u->index].x + dX;
				disp[edge.u->index].y = disp[edge.u->index].y + dY;

				disp[edge.v->index].x = disp[edge.v->index].x - dX;
				disp[edge.v->index].y = disp[edge.v->index].y - dY;
			}
		}
		// limit the maximum displacement to the temperature t
		// and then prevent from being displaced outside frame
		Nodes::Iterator nit = nodes.begin();
		for (; nit != nodes.end(); ++nit)
		{
			u = nit.value();
			qreal dif = sqrt(disp[u.index].x * disp[u.index].x + disp[u.index].y * disp[u.index].y);

			if (dif == 0)
				dif = 1;

			qDebug() << i << " old " << u.x << " " << u.y;
			u.x = u.x + (disp[u.index].x * std::min(fabs(disp[u.index].x), temperature) / dif);
			u.y = u.y + (disp[u.index].y * std::min(fabs(disp[u.index].y), temperature) / dif);
			qDebug() << i << " new " << u.x << " " << u.y;
		}

		temperature = temperature * cooling;
	}

	return 0;
}
