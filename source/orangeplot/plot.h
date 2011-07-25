#ifndef PLOT_H
#define PLOT_H

#include <QtGui/QGraphicsView>

class PlotItem;

class Plot : public QGraphicsView
{
    Q_OBJECT
public:
    Plot(QWidget* parent = 0);
    virtual ~Plot();
    
    virtual void replot() = 0;
    
    void add_item(PlotItem* item);
    void remove_item(PlotItem* item);
    
    QRectF data_rect_for_axes(int x_axis, int y_axis);
    QPair< double, double > bounds_for_axis(int axis);
    
    QList<PlotItem*> plot_items();
    
    void set_graph_rect(const QRectF rect);
    
    QGraphicsRectItem* graph_item;
    
    void set_dirty();
    
protected:
    void set_clean();;
    bool is_dirty();
    
private:
    QList<PlotItem*> m_items;
    bool m_dirty;
    QGraphicsRectItem* clipItem;
};

#endif // PLOT_H
