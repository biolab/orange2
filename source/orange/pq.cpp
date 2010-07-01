/*
 * Binary heap PQ implementation with 'decrease key'.
 *
 * This implementation is based on the implementation by Marko Robnik Šikonja for 
 * Java (I.Kononenko, M.R. Šikonja - Algoritmi in podatkovne strukture I, FRI).
 *
 * @author:  Anze Vavpetic <anze.vavpetic@gmail.com>
 * @summary: Binary heap PQ implementation with 'decrease key'
 */

#include "pq.hpp"

template <class Key>
PQNode<Key>::PQNode(const int &idx, Key key)
  : m_idx(idx),
    m_key(key)
{
}

template <class Key>
PQNode<Key>::~PQNode()
{
}

template <class Key>
bool PQNode<Key>::operator<(const PQNode<Key> &b)
{
  return m_key < b.m_key;
}

template <class Key>
PQHeap<Key>::PQHeap(int size=DEFAULT_SIZE)
  : m_size(size),
    m_noNodes(0),
    m_nodes(size)
{
}

template <class Key>
PQHeap<Key>::~PQHeap()
{
}

template <class Key>
void PQHeap<Key>::makenull()
{
  m_noNodes = 0;
}

template <class Key>
void PQHeap<Key>::insert(PQNode<Key> *n)
{
  // Indices of the new node and its parent
  int newNode = ++m_noNodes;
  int parent = newNode / DEGREE;

  // Check for overflow
  if (m_noNodes > m_size) {
	  cout << "Queue overflow!" << endl;
	  exit(1);
  }
  
  while (parent > 0 && *n < *m_nodes[parent])
  {
    m_nodes[newNode] = m_nodes[parent];
    m_nodes[newNode]->setHeapIndex(newNode);
    newNode = parent;
    parent = parent / DEGREE;
  }
  
  m_nodes[newNode] = n;
  m_nodes[newNode]->setHeapIndex(newNode);
}

template <class Key>
PQNode<Key> *PQHeap<Key>::deleteMin()
{
  int newNode; // New index of the previous last element
  int child;   // Index of the smaller of its children
  
  if (m_noNodes == 0)
    return NULL;
  
  PQNode<Key> *minEl = m_nodes[1];
  newNode = 1;                      // First, set the last element to the root
  
  if (2 * newNode + 1 <= m_noNodes) {                       // Right child exists
    if (*m_nodes[2 * newNode] < *m_nodes[2 * newNode + 1])
      child = 2 * newNode;
    else
      child = 2 * newNode + 1;
  }
  else if (2 * newNode <= m_noNodes)                        // Left child exists
    child = 2 * newNode;
  else                                                      // No children
    child = m_noNodes + 1;
  
  while (child <= m_noNodes && *m_nodes[child] < *m_nodes[m_noNodes]) 
  {
    m_nodes[newNode] = m_nodes[child];
    m_nodes[newNode]->setHeapIndex(newNode);
    newNode = child;
    
    if (2 * newNode + 1 <= m_noNodes)                        // Right child exists
    {
      if (*m_nodes[2 * newNode] < *m_nodes[2 * newNode + 1])
        child = 2 * newNode;
      else
        child = 2 * newNode + 1;
    }
    else if (2 * newNode <= m_noNodes)                       // Left child exists
      child = 2 * newNode;
    else
      child = m_noNodes + 1;
  }
  m_nodes[newNode] = m_nodes[m_noNodes];
  m_nodes[newNode]->setHeapIndex(newNode);
  m_noNodes--;
  
  return minEl;
}

template <class Key>
void PQHeap<Key>::decreaseKey(PQNode<Key> *n, Key k)
{
  if (n->getKey() < k) 
    return;

  n->setKey(k);
  
  int idx = n->getHeapIndex();
  int parent = idx / DEGREE;
  
  while (parent > 0 && *n < *m_nodes[parent]) 
  {
    m_nodes[idx] = m_nodes[parent];
    m_nodes[idx]->setHeapIndex(idx);
    idx = parent;
    parent = parent / DEGREE;
  }
  
  m_nodes[idx] = n;
  m_nodes[idx]->setHeapIndex(idx);
}
