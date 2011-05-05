/*
 * Pathfinder algorithm for Orange
 *
 * @author:  Anze Vavpetic <anze.vavpetic@gmail.com>
 * @summary: Implementation of the binary pathfinder algorithm and an improvement for sparse networks
 */

#ifndef __PATHFINDER_HPP
#define __PATHFINDER_HPP

#include "Python.h"
#include <math.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <queue>

#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "graph.hpp"
#include "pq.hpp"

//#define STATISTICS
#define INFINITY     numeric_limits<int>::max()
#define OUTER        0
#define COMPLETED    1
#define ACTIVE       2

using namespace std;

// Some typedefs for a cleaner code
typedef vector< vector<double> > Matrix;
typedef vector< double > Vector;

/**
 * Class implementing the pq element used in sparse pf.
 */
template <class Key>
class SPFNode : public PQNode<Key> {
public:
    /**
     * Class constructor
     * 
     * @param v: vertex index
     * @param d: distance to the source vertex
     */
    SPFNode(const int &v, const double &d, const short int &state);
    
    short int m_state;  // State of the vertex
    int m_v;            // Vertex index
};
typedef SPFNode<double> SparseNode;

/**
 * Class implementing the queue element used in BFS
 */
class BFSNode {
public:
    /**
     * Class constructor
     */
    BFSNode(const int &idx, const double &d, const int &len);

    int m_idx;    // Vertex index
    int m_len;    // Number of links between this node and the source for the current distance
    double m_d;   // Distance to the source node
};

OMWRAPPER(Pathfinder)

class ORANGEOM_API TPathfinder : public TOrange {
public:
    __REGISTER_CLASS

    /**
     * Class constructor.
     */
    TPathfinder();
    
    /**
     * Class desctructor.
     */
    virtual ~TPathfinder();
  
    /**
     * Transforms the given graph g to a Pathfinder network with the given parameters.
     *
     * Parameters to be used when constructing PFNETs:
     *  @param r: Minkowski r-metric, r >= 1. Note: r should be passed as 
     *            numeric_limits<int>::max() in order to be recognized as 'infinity'!
     *  @param q: The maximum number of intermediate links to be considered
     *  @param g: Target graph.
     *  @param method: The method to be used. Can be "sparse" or "dense".
     *                 Note that only the sparse method is exported to Orange.
     */
    void toPFNET(int r, int q, TGraph *g, const string &method);

    /**
     * Sets the progress callback function.
     */
    inline void setProgressCallback(PyObject *cb) { progressCallback = cb; }

    /**
     * The progress callback function reference.
     */
    PyObject *progressCallback; // Progress callback function
private:
    /**
     * Constructs a PFNET out of the given graph g.
     *   
     * @param g: The orange graph to be transformed
     */
    void toPFNET_binary(TGraph *g);

    /**
     * Pathfinder procedure optimized for sparse networks.
     *
     * @param g: The orange graph to be transformed
     */
    void toPFNET_sparse(TGraph *g);

    /**
     * Pathfinder procedure using an adapted BFS for q < |V|-1
     */
    void BFS(TGraph *g);

    /**
     * Calculates a new distance matrix C as C = A op B, where op is defined by:
     * each element C[k,l] is calculated as 
     * 
     *     MIN{ A[k,l], B[k,l], (A[k,m]^r + B[m,l]^r)^(1/r) } 
     *   
     * @param A: First distance matrix
     * @param B: Second distance matrix
     * @param result: The resulting matrix
     */
    void TPathfinder::op(const Matrix &A, const Matrix &B, Matrix &result) const;
    
    /**
     * Calculates all the possible new distances between nodes i and j.
     *   
     * @param a: Vector a, the i-th line of distance matrix A
     * @param b: Vector b, the j-th line of distance matrix B
     * @param result: vector of possible distances from node i to j
     */
    void distances(const Vector &a, const Vector &b, Vector &result) const;
    
    /**
     * Returns the distance based on the Minkowski r-metric formula
     * 
     * @param a: distance 1
     * @param b: distance 2
     * @return combined distance
     */
    double distance(const double &a, const double &b) const;
    
    /**
     * Return the i-th column of a square matrix
     *
     * @param m: a matrix
     * @param i: the needed column
     * @param col: the i-ith column
     */
    void getCol(const Matrix &m, int i, Vector &col) const;

    /**
     * Handles the progress callback update.
     */
    void updateProgress();

    int m_r; // Minkowski r-metric parameter
    int m_q; // Maximum number of intermediate links to be considered
    int m_nodes; // Number of nodes of the current graph
    int m_done;  // Number of nodes complete
};

#endif
