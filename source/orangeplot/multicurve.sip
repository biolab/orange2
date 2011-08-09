
class MultiCurve : Curve
{
%TypeHeaderCode
#include "multicurve.h"
%End
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
