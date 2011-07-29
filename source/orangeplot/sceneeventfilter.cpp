#include "sceneeventfilter.h"
#include <QtCore/QEvent>

SceneEventFilter::SceneEventFilter(QObject* parent)
{

}

bool SceneEventFilter::eventFilter(QObject* object, QEvent* event)
{
    switch (event->type())
    {
        case QEvent::GraphicsSceneHelp:
        case QEvent::GraphicsSceneContextMenu:
        case QEvent::GraphicsSceneMouseDoubleClick:
            return true;
            
        default:
            return QObject::eventFilter(object, event);
    }
}

SceneEventFilter::~SceneEventFilter()
{

}

