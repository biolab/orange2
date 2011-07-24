#include "curve.h"

#include <QtCore/QDebug>
#include <QtGui/QBrush>
#include <QtGui/QPen>

#include <QtCore/qmath.h>
#include <QtCore/QtConcurrentRun>
#include "point.h"

Curve::Curve(const QList< double >& xData, const QList< double >& yData, QGraphicsItem* parent, QGraphicsScene* scene): PlotItem(parent, scene)
{
    m_style = NoCurve;
    m_continuous = false;
    m_lineItem = 0;
    m_needsUpdate = UpdateAll;
    setData(xData, yData);
    checkForUpdate();
}

Curve::Curve(QGraphicsItem* parent, QGraphicsScene* scene): PlotItem(parent, scene)
{
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
  if (m_continuous)
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
  
  Q_ASSERT(m_pointItems.size() == m_data.size());
}

void Curve::updateProperties()
{
  setContinuous(m_style != Curve::NoCurve);
  
  if (m_needsUpdate & UpdateContinuous)
  {
      changeContinuous();
  }
  
  if (m_continuous)
  {
    QPen p = m_pen;
    p.setWidthF(m_pen.widthF()/m_zoom_factor);
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
      p = QPointF(m_data[i].x, m_data[i].y);
      m_pointItems[i]->setPos(m_graphTransform.map(p));
    }
  } 
  
  if (m_needsUpdate & (UpdateZoom | UpdateBrush | UpdatePen | UpdateSize | UpdateSymbol) )
  {
    updateItems(m_pointItems, PointUpdater(m_symbol, m_color, m_pointSize, Point::DisplayPath, 1.0/m_zoom_factor), UpdateSymbol);
  }
  m_needsUpdate = 0;
}

void Curve::updateAll()
{
  m_needsUpdate = UpdateAll;
  updateProperties();
}

Point* Curve::pointItem(qreal x, qreal y, int size, QGraphicsItem* parent)
{
  if (size == 0)
  {
    size = pointSize();
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

void Curve::setData(const QList< qreal > xData, const QList< qreal > yData)
{
  Q_ASSERT(xData.size() == yData.size());
  int n = qMin(xData.size(), yData.size());
  if (n != m_data.size())
  {
    m_needsUpdate |= UpdateNumberOfItems;
  }
  m_data.clear();
  for (int i = 0; i < n; ++i)
  {
    DataPoint p;
    p.x = xData[i];
    p.y = yData[i];
    m_data.append(p);
  }
  setDataRect(boundingRectFromData(xData, yData));
  m_needsUpdate |= UpdatePosition;
  updateBounds();
  checkForUpdate();
}

QTransform Curve::graphTransform() const
{
  return m_graphTransform;
}

void Curve::setGraphTransform(const QTransform& transform)
{
  if (transform == m_graphTransform)
  {
    return;
  }
  m_needsUpdate |= UpdatePosition;
  m_graphTransform = transform;
  checkForUpdate();
}

bool Curve::isContinuous() const
{
  return m_continuous;
}

void Curve::setContinuous(bool continuous)
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

void Curve::setColor(const QColor& color)
{
    m_color = color;
    setPen(color);
    setBrush(color);
}

QPen Curve::pen() const
{
    return m_pen;
}

void Curve::setPen(QPen pen)
{
    m_pen = pen;
    m_needsUpdate |= UpdatePen;
    checkForUpdate();
}

QBrush Curve::brush() const
{
    return m_brush;
}

void Curve::setBrush(QBrush brush)
{
    m_brush = brush;
    m_needsUpdate |= UpdateBrush;
    checkForUpdate();
}

int Curve::pointSize() const
{
  return m_pointSize;
}

void Curve::setPointSize(int size)
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

void Curve::setSymbol(int symbol)
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

void Curve::setStyle(int style)
{
    m_style = style;
    m_needsUpdate |= UpdateSymbol;
    checkForUpdate();
}



bool Curve::autoUpdate() const
{
  return m_autoUpdate;
}

void Curve::setAutoUpdate(bool autoUpdate)
{
  m_autoUpdate = autoUpdate;
  checkForUpdate();
}

QRectF Curve::graphArea() const
{
  return m_graphArea;
}

void Curve::setGraphArea(const QRectF& area)
{
  m_graphArea = area;
  m_needsUpdate |= UpdatePosition;
  checkForUpdate();
}

void Curve::checkForUpdate()
{
  if ( m_autoUpdate && m_needsUpdate )
  {
    updateProperties();
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
}

void Curve::updateBounds()
{
    int n = m_data.size();
    if (!n)
    {
        m_xBounds.min = 0;
        m_xBounds.max = 0;
        m_yBounds.min = 0;
        m_yBounds.max = 0;
        return;
    }

    m_xBounds.min = m_xBounds.max = m_data[0].x;
    m_yBounds.min = m_yBounds.max = m_data[0].y;
    for (int i = 0; i < n; ++i)
    {
        m_xBounds.min = qMin(m_xBounds.min, m_data[i].x);
        m_xBounds.max = qMax(m_xBounds.max, m_data[i].x);
        m_yBounds.min = qMin(m_yBounds.min, m_data[i].y);
        m_yBounds.max = qMax(m_yBounds.max, m_data[i].y);
    }
}

qreal Curve::max_x_value() const
{
    return m_xBounds.max;
}

qreal Curve::min_x_value() const
{
    return m_yBounds.min;
}


qreal Curve::max_y_value() const
{
    return m_yBounds.max;
}

qreal Curve::min_y_value() const
{
    return m_yBounds.min;
}

void Curve::setDirty(Curve::UpdateFlags flags)
{
    m_needsUpdate |= flags;
    checkForUpdate();
}

double Curve::zoom_factor()
{
    return m_zoom_factor;
}

void Curve::set_zoom_factor(double factor)
{
    m_zoom_factor = factor;
    m_needsUpdate |= UpdateZoom;
    checkForUpdate();
}

void Curve::cancelAllUpdates()
{
    foreach (QFuture<void> f, m_currentUpdate)
    {
        if (f.isRunning())
        {
            f.cancel();
            f.waitForFinished();
        }
    }
    m_currentUpdate.clear();
}
