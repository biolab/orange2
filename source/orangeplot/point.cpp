#include "point.h"

#include <QtGui/QPainter>
#include <QtCore/QDebug>
#include <QtCore/qmath.h>

const int ChangeableColorIndex = 0;

QHash<PointData, QPixmap> Point::pixmap_cache;

uint qHash(const PointData& data)
{
    // uint has 32 bits:
    uint ret = data.size;
    // size only goes up to 20, so 5 bits is enough
    ret |= data.symbol << 5;
    // symbol is less than 16, so 4 bits will do
    ret |= data.state << 9;
    // QRgb takes the full uins, so we just XOR by it
    ret ^= data.color.rgba();
    return ret;
}

bool operator==(const PointData& one, const PointData& other)
{
    return one.symbol == other.symbol && one.size == other.size && one.state == other.state && one.color == other.color;
}


Point::Point(int symbol, QColor color, int size, QGraphicsItem* parent): QGraphicsItem(parent),
 m_symbol(symbol),
 m_color(color),
 m_size(size)
{
    m_display_mode = DisplayPath;
}

Point::Point(QGraphicsItem* parent, QGraphicsScene* scene): QGraphicsItem(parent, scene)
{
    m_symbol = Ellipse;
    m_color = Qt::black;
    m_size = 5;
    m_display_mode = DisplayPath;
}


void Point::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    
    const PointData key(m_size, m_symbol, m_color, m_state);
    // We make the pixmap slighly larger because the point outline has non-zero width
    const int ps = m_size + 4;
    if (!pixmap_cache.contains(key))
    {
        if (m_display_mode == DisplayPath)
        {
            QPixmap pixmap(ps, ps);
            pixmap.fill(Qt::transparent);
            QPainter p;
            p.begin(&pixmap);
            p.setRenderHints(painter->renderHints());
            if (m_state & Selected)
            {
                p.setBrush(m_color);
                p.setPen(Qt::yellow);
            }
            else if (m_state & Marked)
            {
                p.setBrush(m_color);
            }
            else
            {
                p.setPen(m_color);
            }
            const QPainterPath path = path_for_symbol(m_symbol, m_size);
            p.drawPath(path.translated(0.5*ps, 0.5*ps));
            pixmap_cache.insert(key, pixmap);
        } 
        else if (m_display_mode == DisplayPixmap)
        {
            pixmap_cache.insert(key, pixmap_for_symbol(m_symbol, m_color, m_size));
        }
    }
    painter->drawPixmap(QPointF(-0.5*ps, -0.5*ps), pixmap_cache.value(key));
}

QRectF Point::boundingRect() const
{
    return rect_for_size(m_size);
}

Point::~Point()
{

}

QColor Point::color() const
{
    return m_color;
}

void Point::set_color(const QColor& color)
{
    m_color = color;
}

Point::DisplayMode Point::display_mode() const
{
    return m_display_mode;
}

void Point::set_display_mode(Point::DisplayMode mode)
{
    m_display_mode = mode;
}

int Point::size() const
{
    return m_size;
}

void Point::set_size(int size)
{
    m_size = size;
}

int Point::symbol() const
{
    return m_symbol;
}

void Point::set_symbol(int symbol)
{
    m_symbol = symbol;
}

QPainterPath Point::path_for_symbol(int symbol, int size)
{
  QPainterPath path;
  qreal d = 0.5 * size;
  switch (symbol)
  {
    case NoSymbol:
      break;
      
    case Ellipse:
      path.addEllipse(-d,-d,2*d,2*d);
      break;
      
    case Rect:
      path.addRect(-d,-d,2*d,2*d);
      break;
      
    case Diamond:
      path.addRect(-d,-d,2*d,2*d);
      path = QTransform().rotate(45).map(path);
      break;
      
    case Triangle:
    case UTriangle:
      path = trianglePath(d, 0);
      break;
      
    case DTriangle:
      path = trianglePath(d, 180);
      break;
      
    case LTriangle:
      path = trianglePath(d, -90);
      break;
    
    case RTriangle:
      path = trianglePath(d, 90);
      break;

    case Cross:
      path = crossPath(d, 0);
      break;
    
    case XCross:
      path = crossPath(d, 45);
      break;
      
    case HLine:
      path.moveTo(-d,0);
      path.lineTo(d,0);
      break;
      
    case VLine:
      path.moveTo(0,-d);
      path.lineTo(0,d);
      break;
      
    case Star1:
      path.addPath(crossPath(d,0));
      path.addPath(crossPath(d,45));
      break;
      
    case Star2:
      path = hexPath(d, true);
      break;
      
    case Hexagon:
      path = hexPath(d, false);
      break;
      
    default:
     // qWarning() << "Unsupported symbol" << symbol;
        break;
  }
  return path;
}


QPainterPath Point::trianglePath(double d, double rot) {
    QPainterPath path;
    path.moveTo(-d, d*sqrt(3)/3);
    path.lineTo(d, d*sqrt(3)/3);
    path.lineTo(0, -2*d*sqrt(3)/3);
    path.closeSubpath();
    return QTransform().rotate(rot).map(path);
}

QPainterPath Point::crossPath(double d, double rot)
{
    QPainterPath path;
    path.lineTo(0,d);
    path.moveTo(0,0);
    path.lineTo(0,-d);
    path.moveTo(0,0); 
    path.lineTo(d,0);
    path.moveTo(0,0);
    path.lineTo(-d,0);
    return QTransform().rotate(rot).map(path);
}

QPainterPath Point::hexPath(double d, bool star) {
    QPainterPath path;
    if (!star)
    {
        path.moveTo(d,0);
    }
    for (int i = 0; i < 6; ++i)
    {
        path.lineTo( d * cos(M_PI/3*i), d*sin(M_PI/3*i) );
        if (star)
        {
            path.lineTo(0,0);
        }
    }
    path.closeSubpath();
    return path;
}

QPixmap Point::pixmap_for_symbol(int symbol, QColor color, int size)
{
    // Indexed8 is the only format with a color table, which means we can replace entire colors
    // and not only indididual pixels
    QPixmap image(QSize(size, size));
    
    // TODO: Create fils with actual images, preferably SVG so they are scalable
    // image.load(filename);
    return image;
}

QRectF Point::rect_for_size(double size)
{
    return QRectF(-size/2, -size/2, size, size);
}
void Point::set_state(Point::State state) {
    m_state = state;
}
Point::State Point::state() const {
    return m_state;
}
void Point::set_state_flag(Point::StateFlag flag, bool on) {
    if (on)
    {
        m_state |= flag;
    }
    else
    {
        m_size &= ~flag;
    }
    update();
}
bool Point::state_flag(Point::StateFlag flag) const {
    return m_state & flag;
}


void Point::set_selected(bool selected)
{
    set_state_flag(Selected, selected);
}

bool Point::is_selected() const
{
    return state_flag(Selected);
}

void Point::set_marked(bool marked)
{
    set_state_flag(Marked, marked);
}

bool Point::is_marked() const
{
    return state_flag(Marked);
}

void Point::clear_cache()
{
    pixmap_cache.clear();
}




