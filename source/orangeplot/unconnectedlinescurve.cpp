/*
    This file is part of the plot module for Orange
    Copyright (C) 2011  Miha Čančula <miha@noughmad.eu>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "unconnectedlinescurve.h"
#include <QtGui/QPen>
#include <QtCore/QDebug>

UnconnectedLinesCurve::UnconnectedLinesCurve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent, QGraphicsScene* scene): Curve(parent, scene)
{
    m_path_item = new QGraphicsPathItem(this);
    set_data(x_data, y_data);
}

UnconnectedLinesCurve::~UnconnectedLinesCurve()
{

}

void UnconnectedLinesCurve::update_properties()
{
    cancelAllUpdates();
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
