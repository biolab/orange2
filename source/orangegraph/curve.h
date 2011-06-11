#ifndef CURVE_H
#define CURVE_H

#include <QtGui/QGraphicsObject>


  
struct DataPoint
{
  qreal x;
  qreal y;
};
  
typedef QList< DataPoint > Data;

class Curve : public QGraphicsObject
{
  Q_OBJECT
  Q_ENUMS(Symbol)
  /**
   * @brief the curve's color
   * 
   * Contains this curve's color. 
   * If the curve is continuous, the line is drawn with this color. Otherwise, the points are drawn and filled with this color.
   **/
  Q_PROPERTY(QColor color READ color WRITE setColor)
  Q_PROPERTY(int pointSize READ pointSize WRITE setPointSize)
  Q_PROPERTY(int symbol READ symbol WRITE setSymbol)
  Q_PROPERTY(bool continuous READ isContinuous WRITE setContinuous)
  Q_PROPERTY(Data data READ data WRITE setData)
  Q_PROPERTY(QTransform graphTransform READ transform WRITE setTransform)
  /**
   * @brief Update the curve immediately after every change
   * 
   * If this property is set to true, every change to the curve will result in an immediate update. 
   * If it is false, you must explicitely call update() or updateAll() before the changes become visible. 
   * The default is true
   **/
  Q_PROPERTY(bool autoUpdate READ autoUpdate WRITE setAutoUpdate)
  Q_PROPERTY(QRectF graphArea READ graphArea WRITE setGraphArea)
  
public:
  /**
   * @brief Point symbol
   * 
   * The symbols list here matches the one from QwtPlotCurve. 
   **/
  enum Symbol {
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
  
  Curve(QGraphicsItem* parent = 0);
  
  /**
   * @brief Default constructor
   * 
   * Constructs a Curve from a series of data points
   *
   * @param data A list of data points, i.e. pairs of coordinates (x,y)
   * @param parent parent QGraphicsItem, passed to QGraphicsObject's constructor
   **/
  Curve(const Data& data, QGraphicsItem* parent = 0);
    
  /**
   * Default destructor
   *
   **/
  virtual ~Curve();

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  virtual QRectF boundingRect() const;
    
  /**
   * @brief Update the curve
   * 
   * Moves all the points to their current locations, and changes their color, shape and size. 
   * 
   * @note this method is optimized for cases where only one or two of the curve's properties have been changed. 
   * If there were multiple changes since the last update, updateAll() is probably faster. 
   *
   **/
  void update();
  
  /**
   * @brief Updates all curve's properties
   * This methods updates all the curve's properties at once, without checking what needs updating. 
   * It is therefore faster for updates that change more than one property at once
   * 
   * @sa update()
   * 
   **/
  void updateAll();
  
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
  void setData(const Data& data);
  void setData(const QList<qreal> xData, const QList<qreal> yData);
  
  QTransform graphTransform() const;
  void setGraphTransform(const QTransform& transform);
  
  QRectF graphArea() const;
  void setGraphArea(const QRectF& area);
  
  bool autoUpdate() const;
  void setAutoUpdate(bool autoUpdate);
  
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
  Q_DECLARE_FLAGS(UpdateFlags, UpdateFlag)
  static QPainterPath pathForSymbol(int symbol, int size);
  void updateNumberOfItems();
  
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
};

#endif // CURVE_H
