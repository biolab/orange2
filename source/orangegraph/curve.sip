struct DataPoint {

%TypeHeaderCode
#include "curve.h"
%End

    qreal x;
    qreal y;
};

typedef QList<DataPoint> Data;

class Curve : PlotItem {

%TypeHeaderCode
#include "curve.h"
%End

public:

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

  Curve(QList< double > xData, QList< double > yData, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene = 0);
  virtual ~Curve();

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  virtual QRectF boundingRect() const;

  void update();
  void updateAll();
  
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
  void setData(const QList<qreal>& xData, const QList<qreal>& yData);

  QTransform graphTransform() const;
  void setGraphTransform(const QTransform& transform);
  
  bool autoUpdate() const;
  void setAutoUpdate(bool autoUpdate);

  static QPainterPath pathForSymbol(int symbol, int size);
  
  qreal max_x_value() const;
  qreal min_x_value() const;
  qreal max_y_value() const;
  qreal min_y_value() const;
};