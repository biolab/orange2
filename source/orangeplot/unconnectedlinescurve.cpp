#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

UnconnectedLinesCurve::UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent, QGraphicsScene* scene): Curve(x_data, y_data, parent, scene)
{
    m_path_item = new QGraphicsPathItem(this);
}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::update_properties()
{
    if (needs_update() & UpdatePosition)
    {
        const Data d = data();
        const int n = d.size();
        QPainterPath path;
        for (int i = 0; i < n; ++i)
        {
            path.moveTo(d[i].x, d[i].y);
            ++i;
            path.lineTo(d[i].x, d[i].y);
        }
        m_path_item->setPath(graph_transform().map(path));
    }
    if (needs_update() & UpdatePen)
    {   
        QPen p = pen();
        p.setCosmetic(true);
        m_path_item->setPen(p);
    }
    set_updated(Curve::UpdateAll);
}
