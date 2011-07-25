#ifndef POINT_H
#define POINT_H

#include <QtGui/QGraphicsItem>


class Point : public QGraphicsItem
{

public:
    enum DisplayMode
    {
        DisplayPixmap,
        DisplayPath
    };
    
  /**
   * @brief Point symbol
   * 
   * The symbols list here matches the one from QwtPlotCurve. 
   **/
  enum Symbol {
    NoSymbol = -1,
    Ellipse = 0,
    Rect = 1,
    Diamond = 2,
    Triangle = 3,
    DTriangle = 4,
    UTriangle = 5,
    LTriangle = 6,
    RTriangle = 7,
    Cross = 8,
    XCross = 9,
    HLine = 10,
    VLine = 11,
    Star1 = 12,
    Star2 = 13,
    Hexagon = 14,
    UserStyle = 1000
  };
  
    Point(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    Point(int symbol, QColor color, int size, QGraphicsItem* parent = 0);
    virtual ~Point();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    void set_symbol(int symbol);
    int symbol() const;
    
    void set_color(const QColor& color);
    QColor color() const;
    
    void set_size(int size);
    int size() const;
    
    void set_display_mode(DisplayMode mode);
    DisplayMode display_mode() const;
    
    /**
    * Creates a path from a symbol and a size
    *
    * @param symbol the point symbol to use
    * @param size the size of the resulting path
    * @return a path that can be used in a QGraphicsPathItem or in QPainter::drawPath()
    **/
    static QPainterPath path_for_symbol(int symbol, int size);
    
    static QImage image_for_symbol(int symbol, QColor color, int size);
    static QRectF rect_for_size(double size);
    
private:
    static QPainterPath trianglePath(double d, double rot);
    static QPainterPath crossPath(double d, double rot);
    static QPainterPath hexPath(double d, bool star);

    int m_symbol;
    QColor m_color;
    int m_size;
    DisplayMode m_display_mode;
};

#endif // POINT_H
