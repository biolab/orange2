#include "curve.h"

#include <QtCore/QDebug>
#include <QtGui/QBrush>
#include <QtGui/QPen>

#define UPDATE_ITEMS(cond, n, function) \
if (m_needsUpdate & (cond)) \
  for (int i = 0; i < n; ++i) \
    m_pointItems[i]->function;
  

#define CHECK_UPDATE \
if (m_autoUpdate) update();
  
Curve::Curve(QGraphicsItem* parent): QGraphicsObject(parent)
{
  
}
  
Curve::Curve(const Data& data, QGraphicsItem* parent) : QGraphicsObject(parent), m_data(data)
{
  qDebug() << "Constructing Curve from C++";
}

Curve::~Curve()
{

}

void Curve::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{

}

QRectF Curve::boundingRect() const
{

}


void Curve::updateNumberOfItems()
{
  int n = m_data.size();
  if (m_pointItems.size() > n)
  {
    qDeleteAll(m_pointItems.begin() + n, m_pointItems.end());
  }
  int m = n - m_pointItems.size();
  for (int i = 0; i < m; ++i)
  {
    QGraphicsPathItem * p = new QGraphicsPathItem(this);
    p->setPen(Qt::NoPen);
    m_pointItems.append(p);
  }
}

void Curve::update()
{
  int n = m_data.size();  

  if (m_needsUpdate & UpdateNumberOfItems)
  {
    updateNumberOfItems();
  }
  
  if (m_needsUpdate & (UpdateSize | UpdateSymbol))
  {
    m_path = pathForSymbol(m_symbol, m_pointSize);
    for (int i = 0; i < n; ++i)
    {
      m_pointItems[i]->setPath(m_path);
    }
  }
  
  // Move, resize, reshape and/or recolor the items
  if (m_needsUpdate & UpdatePosition)
  {
    for (int i = 0; i < n; ++i)
    {
      QPointF p = m_graphTransform.map(QPointF(m_data[i].x, m_data[i].y));
      if (m_graphArea.contains(p))
      {
	m_pointItems[i]->show();
	m_pointItems[i]->setPos(p);
      }
      else
      {
	m_pointItems[i]->hide();
      }
    }
  }
  QBrush brush(m_color);
  UPDATE_ITEMS(UpdateColor, n, setBrush(brush))
  
  m_needsUpdate = 0;
}

void Curve::updateAll()
{
  if (m_needsUpdate & UpdateNumberOfItems)
  {
    updateNumberOfItems();
  }
  
  int n = m_data.size();
  QBrush brush(m_color);
  m_path = pathForSymbol(m_symbol, m_pointSize);
  for (int i = 0; i < n; ++i)
  {
    QGraphicsPathItem* item = m_pointItems[i];
    DataPoint& point = m_data[i];
    item->setPath(m_path);
    QPointF p = m_graphTransform.map(QPointF(point.x, point.y));
    item->setPos(p);
    item->setBrush(brush);
    item->setVisible(m_graphArea.contains(p));
  }
  m_needsUpdate = 0;
}


QGraphicsItem* Curve::pointItem(qreal x, qreal y, int size, QGraphicsItem* parent)
{
  if (size == 0)
  {
    size = pointSize();
  }
  if (parent == 0)
  {
    parent = this;
  }
  QGraphicsPathItem* item = new QGraphicsPathItem(pathForSymbol(symbol(),size), parent);
  item->setPos(x,y);
  item->setPen(Qt::NoPen);
  item->setBrush(m_color);
  return item;
}

QPainterPath Curve::pathForSymbol(int symbol, int size)
{
  QPainterPath path;
  qreal d = 0.5 * size;
  switch (symbol)
  {
    case Ellipse:
      path.addEllipse(-d,-d,d,d);
      break;
      
    case Rect:
      path.addRect(-d,-d,d,d);
      break;
      
    default:
      qWarning() << "Unsupported symbol" << symbol;
  }
  return path;
}

Data Curve::data() const
{
  return m_data;
}

void Curve::setData(const Data& data)
{
  if (data.size() != m_data.size())
  {
    m_needsUpdate |= UpdateNumberOfItems;
  }
  m_data = data;
  m_needsUpdate |= UpdatePosition;
  CHECK_UPDATE
}

void Curve::setData(const QList< qreal > xData, const QList< qreal > yData)
{
  qDebug() << xData.size() << yData.size();
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
  m_needsUpdate |= UpdatePosition;
  CHECK_UPDATE
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
  CHECK_UPDATE
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
  CHECK_UPDATE
}

QColor Curve::color() const
{
  return m_color;
}

void Curve::setColor(const QColor& color)
{
  qDebug() << "Setting color to" << color;
  if (color == m_color)
  {
    return;
  }
  m_color = color;
  m_needsUpdate |= UpdateColor;
  CHECK_UPDATE
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
  CHECK_UPDATE
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
  CHECK_UPDATE
}

bool Curve::autoUpdate() const
{
  return m_autoUpdate;
}

void Curve::setAutoUpdate(bool autoUpdate)
{
  m_autoUpdate = autoUpdate;
  CHECK_UPDATE
}

QRectF Curve::graphArea() const
{
  return m_graphArea;
}

void Curve::setGraphArea(const QRectF& area)
{
  m_graphArea = area;
  m_needsUpdate |= UpdatePosition;
  CHECK_UPDATE
}

#include "curve.moc"
