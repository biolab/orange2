class PlotItem : QGraphicsItem {

%TypeHeaderCode
#include "plotitem.h"
%End

public:
    PlotItem(QList<double> xData, QList<double> yData, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF dataRect() const;
    
    void attach(Graph* graph /TransferThis/);
    void detach();
    
    static QRectF boundingRectFromData(QList<double> xData, QList<double> yData);
    
    bool isAutoScale() const;
    void setAutoScale(bool autoScale);
    
    QPair<int, int> axes() const;
    void setAxes(int x_axis, int y_axis);  

private:
    PlotItem(const PlotItem &);
};