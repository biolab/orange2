#ifndef CURVE_H
#define CURVE_H

#include "plotitem.h"

struct DataPoint
{
  qreal x;
  qreal y;
};
  
typedef QList< DataPoint > Data;

class Curve : public PlotItem
{
  
public:
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
  
  /**
   * @brief Default constructor
   * 
   * Constructs a Curve from a series of data points
   *
   * @param xData A list of x coordinates of data points
   * @param yData A list of y coordinates of data points
   * @param parent parent item
   * @param scene if this is not 0, the Curve is automatically added to it
   **/
  Curve(QList< double > xData, QList< double > yData, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);

  /**
   * Default destructor
   *
   **/
  virtual ~Curve();
    
  /**
   * @brief Update the curve
   * 
   * Moves all the points to their current locations, and changes their color, shape and size. 
   * 
   * @note this method is optimized for cases where only one or two of the curve's properties have been changed. 
   * If there were multiple changes since the last update, updateAll() is probably faster. 
   *
   **/
   virtual void updateProperties();
  
  /**
   * @brief Updates all curve's properties
   * This methods updates all the curve's properties at once, without checking what needs updating. 
   * It is therefore faster for updates that change more than one property at once
   * 
   * @sa update()
   * 
   **/
  virtual void updateAll();
  
  /**
   * @brief ...
   *
   * @param x ...
   * @param y ...
   * @param size ...
   * @param parent ... Defaults to 0.
   * @return QGraphicsItem*
   **/
  QGraphicsItem* pointItem(qreal x, qreal y, int size = 0, QGraphicsItem* parent = 0);
  
  QColor color() const;
  void setColor(const QColor& color);
  
  int pointSize() const;
  void setPointSize(int size);
  
  int symbol() const;
  void setSymbol(int symbol);
  
  bool isContinuous() const;
  void setContinuous(bool continuous);
  
  Data data() const;
  void setData(const QList<qreal> xData, const QList<qreal> yData);
  
  QTransform graphTransform() const;
  void setGraphTransform(const QTransform& transform);
  
  QRectF graphArea() const;
  void setGraphArea(const QRectF& area);
  
  bool autoUpdate() const;
  void setAutoUpdate(bool autoUpdate);
  
  qreal max_x_value() const;
  qreal min_x_value() const;
  qreal max_y_value() const;
  qreal min_y_value() const;
  
  /**
   * Creates a path from a symbol and a size
   *
   * @param symbol the point symbol to use
   * @param size the size of the resulting path
   * @return a path that can be used in a QGraphicsPathItem
   **/
  static QPainterPath pathForSymbol(int symbol, int size);
  
  
private:    

  enum UpdateFlag
  {
    UpdateNumberOfItems = 0x01,
    UpdatePosition = 0x02,
    UpdateSymbol = 0x04,
    UpdateSize = 0x08,
    UpdateColor = 0x10,
    UpdateContinuous = 0x20,
    UpdateAll = 0xFF
  };
  
  struct Bounds
  {
      qreal min;
      qreal max;
  };
  
  Q_DECLARE_FLAGS(UpdateFlags, UpdateFlag)
  
  void checkForUpdate();
  void updateNumberOfItems();
  void changeContinuous();
  void updateBounds();
  
  static QPainterPath trianglePath(double d, double rot);
  
  QColor m_color;
  int m_pointSize;
  int m_symbol;
  bool m_continuous;
  Data m_data;
  QTransform m_graphTransform;
  QPainterPath m_path;
  QList<QGraphicsPathItem*> m_pointItems;
  UpdateFlags m_needsUpdate;
  bool m_autoUpdate;
  QRectF m_graphArea;
    QGraphicsPathItem* m_lineItem;
    QPainterPath m_line;
    
    Bounds m_xBounds;
    Bounds m_yBounds;
};

#endif // CURVE_H
