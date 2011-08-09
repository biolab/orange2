/*
    <one line to give the program's name and a brief idea of what it does.>
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


#ifndef MULTICURVE_H
#define MULTICURVE_H

#include "curve.h"


class MultiCurve : public Curve
{
public:
    MultiCurve(const QList< double >& x_data, const QList< double >& y_data);
    virtual ~MultiCurve();
    
    void set_point_colors(const QList<QColor>& colors);
    void set_point_labels(const QStringList& colors);
    void set_point_symbols(const QList< int >& symbols);
    void set_point_sizes(const QList<int>& colors);
    
    void shuffle_points();

    virtual void update_properties();
};

#endif // MULTICURVE_H
