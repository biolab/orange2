#include "curve.h"

#include <QtCore/QDebug>
#include <QtGui/QBrush>
#include <QtGui/QPen>

#include <QtCore/qmath.h>
#include <QtCore/QtConcurrentRun>
#include "point.h"
#include "plot.h"

Curve::Curve(const QList< double >& x_data, const QList< double >& y_data, QGraphicsItem* parent, QGraphicsScene* scene): PlotItem(parent, scene)
{
    // Don't make any calls to update_properties() until the constructor is finished
    // Otherwise, the program hangs if this is called from a subclass constructor
    m_autoUpdate = false; 
    
    m_style = NoCurve;
    m_continuous = false;
    m_lineItem = 0;
    m_needsUpdate = UpdateAll;
    set_data(x_data, y_data);
    
    m_autoUpdate = true;
}

Curve::Curve(QGraphicsItem* parent, QGraphicsScene* scene): PlotItem(parent, scene)
{
    m_autoUpdate = true;
    m_style = NoCurve;
    m_lineItem = 0;
    m_needsUpdate = 0;
}


Curve::~Curve()
{
    cancelAllUpdates();
}

void Curve::updateNumberOfItems()
{
  cancelAllUpdates();
  if (m_continuous || (m_data.size() == m_pointItems.size()))
  {
    m_needsUpdate &= ~UpdateNumberOfItems;
    return;
  }
  int n = m_data.size();
  if (m_pointItems.size() > n)
  {
    qDeleteAll(m_pointItems.constBegin() + n, m_pointItems.constEnd());
    m_pointItems.erase(m_pointItems.begin() + n, m_pointItems.end());
  }
  int m = n - m_pointItems.size();
  for (int i = 0; i < m; ++i)
  {
    m_pointItems << new Point(m_symbol, m_color, m_pointSize, this);
  }
  register_points();
  Q_ASSERT(m_pointItems.size() == m_data.size());
}

void Curve::update_properties()
{
  set_continuous(m_style != Curve::NoCurve);
  
  if (m_needsUpdate & UpdateContinuous)
  {
      changeContinuous();
  }
  
  if (m_continuous)
  {
    QPen p = m_pen;
    p.setWidthF(m_pen.widthF()/m_zoom_transform.determinant());
    m_lineItem->setPen(p);
    m_line = QPainterPath();
    if (!m_data.isEmpty())
    {
      m_line.moveTo(QPointF(m_data[0].x, m_data[0].y) * m_graphTransform);
      int n = m_data.size();
      QPointF p;
      for (int i = 1; i < n; ++i)
      {
        p = QPointF(m_data[i].x, m_data[i].y);
        m_line.lineTo(m_graphTransform.map(p));
      }
    }
    m_lineItem->setPath(m_line);
    return;
  } 
  
  int n = m_data.size();
  if (m_pointItems.size() != n)
  {
    updateNumberOfItems();
  }
  
  Q_ASSERT(m_pointItems.size() == n);
  
  
  // Move, resize, reshape and/or recolor the items
  if (m_needsUpdate & UpdatePosition)
  {
    cancelAllUpdates();
    QPointF p;
    for (int i = 0; i < n; ++i)
    {
      m_pointItems[i]->set_coordinates(m_data[i]);
    }
    update_items(m_pointItems, PointPosUpdater(m_graphTransform), UpdatePosition);
  } 
  
  if (m_needsUpdate & (UpdateZoom | UpdateBrush | UpdatePen | UpdateSize | UpdateSymbol) )
  {
    update_items(m_pointItems, PointUpdater(m_symbol, m_color, m_pointSize, Point::DisplayPath, point_transform()), UpdateSymbol);
  }
  m_needsUpdate = 0;
}

Point* Curve::point_item(double x, double y, int size, QGraphicsItem* parent)
{
  if (size == 0)
  {
    size = point_size();
  }
  if (parent == 0)
  {
    parent = this;
  }
  Point* item = new Point(m_symbol, m_color, m_pointSize, parent);
  item->setPos(x,y);
  return item;
}

Data Curve::data() const
{
  return m_data;
}

void Curve::set_data(const QList< double > x_data, const QList< double > y_data)
{
  Q_ASSERT(x_data.size() == y_data.size());
  int n = qMin(x_data.size(), y_data.size());
  if (n != m_data.size())
  {
    m_needsUpdate |= UpdateNumberOfItems;
  }
  m_data.clear();
  m_data.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    DataPoint p;
    p.x = x_data[i];
    p.y = y_data[i];
    m_data.append(p);
  }
  set_data_rect(rect_from_data(x_data, y_data));
  m_needsUpdate |= UpdatePosition;
  checkForUpdate();
}

QTransform Curve::graph_transform() const
{
  return m_graphTransform;
}

void Curve::set_graph_transform(const QTransform& transform)
{
  if (transform == m_graphTransform)
  {
    return;
  }
  m_needsUpdate |= UpdatePosition;
  m_graphTransform = transform;
  checkForUpdate();
}

bool Curve::is_continuous() const
{
  return m_continuous;
}

void Curve::set_continuous(bool continuous)
{
  if (continuous == m_continuous)
  {
    return;
  }
  m_continuous = continuous;
  m_needsUpdate |= UpdateContinuous;
  checkForUpdate();
}

QColor Curve::color() const
{
  return m_color;
}

void Curve::set_color(const QColor& color)
{
    m_color = color;
    set_pen(color);
    set_brush(color);
}

QPen Curve::pen() const
{
    return m_pen;
}

void Curve::set_pen(QPen pen)
{
    m_pen = pen;
    m_needsUpdate |= UpdatePen;
    checkForUpdate();
}

QBrush Curve::brush() const
{
    return m_brush;
}

void Curve::set_brush(QBrush brush)
{
    m_brush = brush;
    m_needsUpdate |= UpdateBrush;
    checkForUpdate();
}

int Curve::point_size() const
{
  return m_pointSize;
}

void Curve::set_point_size(int size)
{
  if (size == m_pointSize)
  {
    return;
  }
  
  m_pointSize = size;
  m_needsUpdate |= UpdateSize;
  checkForUpdate();
}

int Curve::symbol() const
{
  return m_symbol;
}

void Curve::set_symbol(int symbol)
{
  if (symbol == m_symbol)
  {
    return;
  }
  m_symbol = symbol;
  m_needsUpdate |= UpdateSymbol;
  checkForUpdate();
}

int Curve::style() const
{
    return m_style;
}

void Curve::set_style(int style)
{
    m_style = style;
    m_needsUpdate |= UpdateSymbol;
    checkForUpdate();
}



bool Curve::auto_update() const
{
  return m_autoUpdate;
}

void Curve::set_auto_update(bool auto_update)
{
  m_autoUpdate = auto_update;
  checkForUpdate();
}

void Curve::checkForUpdate()
{
  if ( m_autoUpdate && m_needsUpdate )
  {
    update_properties();
  }
}

void Curve::changeContinuous()
{
  cancelAllUpdates();
  if (m_continuous)
  {
    qDeleteAll(m_pointItems);
    m_pointItems.clear();
    
    if (!m_lineItem)
    {
      m_lineItem = new QGraphicsPathItem(this);
    }
  } else {
    m_line = QPainterPath();
    delete m_lineItem;
    m_lineItem = 0;
  }
  register_points();
}

void Curve::set_dirty(Curve::UpdateFlags flags)
{
    m_needsUpdate |= flags;
    checkForUpdate();
}

void Curve::set_zoom_transform(const QTransform& transform)
{
    m_zoom_transform = transform;
    m_needsUpdate |= UpdateZoom;
    checkForUpdate();
}

QTransform Curve::zoom_transform()
{
    return m_zoom_transform;
}

void Curve::cancelAllUpdates()
{
    QMap<UpdateFlag, QFuture< void > >::iterator it = m_currentUpdate.begin();
    QMap<UpdateFlag, QFuture< void > >::iterator end = m_currentUpdate.end();
    for (it; it != end; ++it)
    {
        if (it.value().isRunning())
        {
            it.value().cancel();
        }
    }
    for (it = m_currentUpdate.begin(); it != end; ++it)
    {
        if (it.value().isRunning())
        {
            it.value().waitForFinished();
        }
    }
    m_currentUpdate.clear();
}

void Curve::register_points()
{
    Plot* p = plot();
    if (p)
    {
        p->remove_all_points(this);
        p->add_points(m_data, m_pointItems, this);
    }
}

QTransform Curve::point_transform()
{
    const QTransform i = m_zoom_transform.inverted();
    return QTransform(i.m11(), 0, 0, 0, i.m22(), 0, 0, 0, 1.0);
}

Curve::UpdateFlags Curve::needs_update()
{
    return m_needsUpdate;
}

void Curve::set_updated(Curve::UpdateFlags flags)
{
    m_needsUpdate &= ~flags;
}



