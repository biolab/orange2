#ifndef SCENEEVENTFILTER_H
#define SCENEEVENTFILTER_H

#include <QtCore/QObject>


class SceneEventFilter : public QObject
{

public:
    explicit SceneEventFilter(QObject* parent = 0);
    virtual ~SceneEventFilter();

    virtual bool eventFilter(QObject* object, QEvent* event );
};

#endif // SCENEEVENTFILTER_H
