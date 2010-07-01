/*
 * Binary heap PQ implementation with 'decrease key'.
 *
 * This implementation is based on the implementation by Marko Robnik Šikonja for 
 * Java (I.Kononenko, M.R. Šikonja - Algoritmi in podatkovne strukture I, FRI).
 *
 * @author:  Anze Vavpetic <anze.vavpetic@gmail.com>
 * @summary: Binary heap PQ implementation with 'decrease key'
 */

#ifndef __PQ_HPP
#define __PQ_HPP

#include <vector>

#define DEFAULT_SIZE  100
#define DEGREE          2

using namespace std;

/** 
 * Priority queue element.
 */
template <class Key>
class PQNode {
public:
    /**
     * Class constructor.
     *
     * @param idx: heap index
     * @param key: the key used for ordering
     */
    PQNode(const int &idx, Key key);
    
    /**
     * Class destructor.
     */
    ~PQNode();
    
    /**
     * Returns the key for this element.
     *
     * @return key for this element
     */
    Key getKey() const { return m_key; }
    
    /**
     * Sets the key for this element.
     *
     * @param key: new key
     */
    void setKey(Key key) { m_key = key; }
    
    /**
     * Returns the heap index of this element.
     *
     * @return heap index
     */
    int getHeapIndex() const { return m_idx; }
    
    /**
     * Sets the heap index of this element.
     *
     * @param i: new index
     */
    void setHeapIndex(int i) { m_idx = i; }
    
    /**
     * Method defining the less operator for this class.
     */
    bool operator<(const PQNode<Key> &b);
private:
    Key m_key;   // The ordering key
    int m_idx;   // Index into the heap
};

/**
 * Class implementing the binary heap PQ data structure.
 */ 
template <class Key>
class PQHeap {
public:
    /**
     * Class constructor.
     *
     * @param size: size of the priority queue
     */
    PQHeap(int size=DEFAULT_SIZE);
    
    /**
     * Class destructor.
     */
    ~PQHeap();
    

    /**
     * Empty the pq.
     */
    void makenull();

    /**
     * Returns true if the pq is empty.
     */
    bool empty() const 
    {
      return (m_noNodes == 0 ? true : false);
    }
    
    /**
     * Inserts an element into the queue.
     *
     * @param n: node instance to be inserted
     */
    void insert(PQNode<Key> *n);
    
    /** 
     * Removes and returns the minimum element.
     */
    PQNode<Key> *deleteMin();
    
    /**
     * Decreases the key of the given element.
     */
    void decreaseKey(PQNode<Key> *n, Key k);
private:
    vector<PQNode<Key> *> m_nodes;
    int m_noNodes;  // Number of nodes
    int m_size;     // Vector size
};

#endif