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


#include "multicurve.h"
#include "plot.h"

MultiCurve::MultiCurve(const QList< double >& x_data, const QList< double >& y_data): Curve()
{
    set_continuous(false);
    set_data(x_data, y_data);
}

MultiCurve::~MultiCurve()
{

}

void MultiCurve::set_point_colors(const QList< QColor >& colors)
{
    updateNumberOfItems();
    const QList<Point*> p = points();
    int n = p.size();
    if (colors.size() == n)
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_color(colors[i]);
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_color(Qt::black);
        }
    }
}

void MultiCurve::set_point_labels(const QStringList& labels)
{
    updateNumberOfItems();
    const QList<Point*> p = points();
    int n = p.size();
    if (labels.size() == n)
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_label(labels[i]);
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_label(QString());
        }
    }
}

void MultiCurve::set_point_sizes(const QList<int>& sizes)
{
    updateNumberOfItems();
    const QList<Point*> p = points();
    int n = p.size();
    if (sizes.size() == n)
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_size(sizes[i]);
        }
    }
    else if (!sizes.isEmpty())
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_size(sizes.first());
        }
    }
}

void MultiCurve::set_point_symbols(const QList< int >& symbols)
{
    updateNumberOfItems();
    const QList<Point*> p = points();
    int n = p.size();
    if (symbols.size() == n)
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_symbol(symbols[i]);
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_symbol(Point::Ellipse);
        }
    }
}

void MultiCurve::update_properties()
{
    updateNumberOfItems();
    
    const Data d = data();
    const int n = d.size();
    const QTransform t = graph_transform();
    const QList<Point*> p = points();
    
    for (int i = 0; i < n; ++i)
    {
        p[i]->set_coordinates(d[i]);
    }
    
    register_points();
    update_point_positions();
    update_items(points(), ZoomUpdater(point_transform()), UpdateZoom);
}

void MultiCurve::shuffle_points()
{
    updateNumberOfItems();
    foreach (Point* p, points())
    {
        p->setZValue(qrand() * 1.0 / RAND_MAX);
    }
}

void MultiCurve::set_alpha_value(int alpha)
{
    update_items(points(), PointAlphaUpdater(alpha), UpdateBrush);
}

void MultiCurve::set_points_marked(const QList< bool >& marked)
{
    updateNumberOfItems();
    QList<Point*> p = points();
    const int n = p.size();
    if (marked.size() == n)
    {
        for (int i = 0; i < n; ++i)
        {
            p[i]->set_marked(marked[i]);
        }
    }
    else 
    {
        bool m = marked.isEmpty() ? false : marked.first();
        foreach (Point* point, p)
        {
            point->set_marked(m);
        }
    }
}

