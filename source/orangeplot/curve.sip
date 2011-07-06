// DataPoint is implemented as a Python 2-element tuple.
%MappedType DataPoint /DocType="tuple-of-float-float"/
{
%TypeHeaderCode
#include "curve.h"
%End

%ConvertFromTypeCode
    // Create the tuple.
    return Py_BuildValue((char *)"dd", sipCpp->x, sipCpp->y);
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PyTuple_Size(sipPy) == 2);
    
    DataPoint* qp = new DataPoint();

    PyArg_ParseTuple(sipPy, "dd", &qp->x, &qp->y);
    *sipCppPtr = qp;
    
    return sipGetState(sipTransferObj);
%End
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


  enum Style {
    NoCurve = Qt::NoPen,
    Lines = Qt::SolidLine,
    Sticks,
    Steps,
    Dots = Qt::DotLine,
    UserCurve
  };

  Curve(const QList< double >& xData, const QList< double >& yData, QGraphicsItem* parent /TransferThis/ = 0, QGraphicsScene* scene /TransferThis/ = 0);
  virtual ~Curve();

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
  virtual QRectF boundingRect() const;

  virtual void updateProperties();
  void updateAll();
  
  QGraphicsItem* pointItem(qreal x, qreal y, int size = 0, QGraphicsItem* parent = 0);
  
  QColor color() const;
  void setColor(const QColor& color);
  
  QPen pen() const;
  void setPen(QPen pen);
  
  QBrush brush() const;
  void setBrush(QBrush brush);
  
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

  int style() const;
  void setStyle(int style);
  
  bool autoUpdate() const;
  void setAutoUpdate(bool autoUpdate);

  static QPainterPath pathForSymbol(int symbol, int size);
  
  qreal max_x_value() const;
  qreal min_x_value() const;
  qreal max_y_value() const;
  qreal min_y_value() const;
  
  enum UpdateFlag
  {
    UpdateNumberOfItems = 0x01,
    UpdatePosition = 0x02,
    UpdateSymbol = 0x04,
    UpdateSize = 0x08,
    UpdatePen = 0x10,
    UpdateBrush = 0x20,
    UpdateContinuous = 0x40,
    UpdateAll = 0xFF
  };
  
  typedef QFlags<Curve::UpdateFlag> UpdateFlags;
  
  void setDirty(Curve::UpdateFlags flags = Curve::UpdateAll);
  
};