struct DataPoint {

%TypeHeaderCode
#include "curve.h"
%End

    qreal x;
    qreal y;
};

typedef QList<DataPoint> Data;

class Curve : QGraphicsObject {

%TypeHeaderCode
#include "curve.h"
%End

%TypeCode
// Create a Data object from a list of tuples
static Data dataFromTupleList(PyObject* l)
{
  Data data;
  int n = PyList_GET_SIZE(l);
  DataPoint p;
  for (SIP_SSIZE_T i = 0; i < PyList_GET_SIZE(l); ++i)
  {
    if (PyArg_ParseTuple(PyList_GET_ITEM(l,i), "dd", &p.x, &p.y))
    {
      data.append(p);
    }
  }
  return data;
}
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

  Curve(QGraphicsItem* parent /TransferThis/ = 0);
  Curve(SIP_PYLIST data, QGraphicsItem* parent /TransferThis/ = 0) [(Data, QGraphicsItem*)];
%MethodCode
  sipCpp = new sipCurve(dataFromTupleList(a0), a1);
%End

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
  void setData(SIP_PYLIST data);
%MethodCode
  sipCpp->setData(dataFromTupleList(a0));
%End

  void setData(const QList<qreal>& xData, const QList<qreal>& yData);

  QTransform graphTransform() const;
  void setGraphTransform(const QTransform& transform);
  
  QRectF graphArea() const;
  void setGraphArea(const QRectF& area);
  
  bool autoUpdate() const;
  void setAutoUpdate(bool autoUpdate);

  static QPainterPath pathForSymbol(int symbol, int size);
  
};