#ifndef CURVE_H
#define CURVE_H

#include "plotitem.h"
#include "point.h"

#include <QtGui/QPen>
#include <QtGui/QBrush>

#include <QtCore/QtConcurrentMap>
#include "pointscollection.h"

struct DataPoint
{
  double x;
  double y;
};

struct ScaleUpdater
{
    ScaleUpdater(double scale) {m_scale = scale;}
    void operator()(QGraphicsItem* item) {item->setScale(m_scale);}
    
private:
    double m_scale;
};

struct PointUpdater
{
    PointUpdater(int symbol, QColor color, int size, Point::DisplayMode mode, double scale)
    {
        m_symbol = symbol;
        m_color = color;
        m_size = size;
        m_mode = mode;
        m_scale = scale;
    }
    
    void operator()(Point* point)
    {
        point->set_symbol(m_symbol);
        point->set_color(m_color);
        point->set_size(m_size);
        point->set_display_mode(m_mode);
        point->setScale(m_scale);
    }
    
    private:
     int m_symbol;
     QColor m_color;
     int m_size;
     Point::DisplayMode m_mode;
     double m_scale;
};

struct Updater
{
    Updater(double scale, const QPen& pen, const QBrush& brush, const QPainterPath& path)
    {
        m_scale = scale;
        m_pen = pen;
        m_brush = brush;
        m_path = path;
    }
    
    void operator()(QGraphicsPathItem* item)
    {
        item->setBrush(m_brush);
        item->setPen(m_pen);
        item->setScale(m_scale);
        item->setPath(m_path);
    }
    
    double m_scale;
    QPen m_pen;
    QBrush m_brush;
    QPainterPath m_path;
};
  
typedef QList< DataPoint > Data;

class Curve : public PlotItem, public PointsCollection
{
  
public:
  enum Style {
    NoCurve = Qt::NoPen,
    Lines = Qt::SolidLine,
    Sticks,
    Steps,
    Dots = Qt::DotLine,
    UserCurve = 100
  };
  
  /**
   * @brief Default constructor
   * 
   * Constructs a Curve from a series of data points
   *
   * @param x_data A list of x coordinates of data points
   * @param y_data A list of y coordinates of data points
   * @param parent parent item
   * @param scene if this is not 0, the Curve is automatically added to it
   **/
  Curve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
  Curve(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
  /**
   * Default destructor
   *
   **/
  virtual ~Curve();
    
  /**
   * @brief Update the curve
   * 
   * Moves all the points to their current locations, and changes their color, shape and size. 
   * Subclasses should reimplement this method to update their specific properties. 
   * 
   **/
   virtual void update_properties();
  
  Point* point_item(double x, double y, int size = 0, QGraphicsItem* parent = 0);
  
  QColor color() const;
  void set_color(const QColor& color);
  
  QPen pen() const;
  void set_pen(QPen pen);
  
  QBrush brush() const;
  void set_brush(QBrush brush);
  
  int point_size() const;
  void set_point_size(int size);
  
  int symbol() const;
  void set_symbol(int symbol);
  
  bool is_continuous() const;
  void set_continuous(bool continuous);

  Data data() const;
  void set_data(const QList<double> x_data, const QList<double> y_data);
  
  virtual QTransform graph_transform() const;
  virtual void set_graph_transform(const QTransform& transform);
  
  QRectF graphArea() const;
  void setGraphArea(const QRectF& area);
  
  int style() const;
  void set_style(int style);
  
  bool auto_update() const;
  void set_auto_update(bool auto_update);
  
  double zoom_factor();
  void set_zoom_factor(double factor);
  
  double max_x_value() const;
  double min_x_value() const;
  double max_y_value() const;
  double min_y_value() const;

  enum UpdateFlag
  {
    UpdateNumberOfItems = 0x01,
    UpdatePosition = 0x02,
    UpdateSymbol = 0x04,
    UpdateSize = 0x08,
    UpdatePen = 0x10,
    UpdateBrush = 0x20,
    UpdateContinuous = 0x40,
    UpdateZoom = 0x80,
    UpdateAll = 0xFF
  };
  
  Q_DECLARE_FLAGS(UpdateFlags, UpdateFlag)
  
  void set_dirty(UpdateFlags flags = UpdateAll);
  
  template <class Sequence, class Updater>
  void update_items(Sequence& sequence, Updater updater, Curve::UpdateFlag flag);
  
private:
  void checkForUpdate();
  void updateNumberOfItems();
  void changeContinuous();
  void cancelAllUpdates();
  
  QColor m_color;
  int m_pointSize;
  int m_symbol;
  int m_style;
  bool m_continuous;
  Data m_data;
  QTransform m_graphTransform;
  QList<Point*> m_pointItems;
  UpdateFlags m_needsUpdate;
  bool m_autoUpdate;
    QGraphicsPathItem* m_lineItem;
    QPainterPath m_line;
    
  QPen m_pen;
  QBrush m_brush;
  double m_zoom_factor;
  QMap<UpdateFlag, QFuture<void> > m_currentUpdate;
};

template <class Sequence, class Updater>
void Curve::update_items(Sequence& sequence, Updater updater, Curve::UpdateFlag flag)
{
    if (m_currentUpdate.contains(flag) && m_currentUpdate[flag].isRunning())
    {
        m_currentUpdate[flag].cancel();
        m_currentUpdate[flag].waitForFinished();
    }
    m_currentUpdate[flag] = QtConcurrent::map(sequence, updater);
}


#endif // CURVE_H
