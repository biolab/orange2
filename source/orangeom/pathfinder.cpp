/*
 * Pathfinder algorithm for Orange
 *
 * @author:  Anze Vavpetic <anze.vavpetic@gmail.com>
 * @summary: Implementation of the binary pathfinder algorithm and an improvement for sparse networks
 */

#include "ppp/pathfinder.ppp"
#include "pq.cpp"

template <class Key>
DNode<Key>::DNode(const int &v, const double &d)
  : PQNode<Key>(-1, d),
    m_v(v),
    m_visited(false)
{
}

template <class Key>
SPFNode<Key>::SPFNode(const int &v, const double &d, const short int &state)
  : PQNode<Key>(-1, d),
    m_v(v),
    m_state(state)
{
}

BFSNode::BFSNode(const int &idx, const double &d, const int &len)
  : m_idx(idx),
    m_d(d),
    m_len(len)
{
}

TPathfinder::TPathfinder(int r, int q)
  : m_r(r),
    m_q(q)
{
}

TPathfinder::~TPathfinder() 
{
}

void TPathfinder::toPFNET(TGraph *g)
{
  int n = g->nVertices;
  int i = 1, nq = 0;
  
  // Define two nxn matrices
  Matrix D  (n, Vector(n, INFINITY));
  Matrix Dq (n, Vector(n, INFINITY));
  
  // Init the first distance matrix to the graph's weight matrix
  for (int j=0; j<n; j++) {
    for (int k=0; k<n; k++) 
    {
      double *w = g->getEdge(j,k);
      
      if (w != NULL) 
        D[j][k] = *w;
      
      // This is just precautionary - we must assume the distance of a node to itself is 0
      if (j == k)
        D[j][k] = 0;
    }
  }
  
  if (m_q % 2 == 1) {
    op(Dq, D, Dq);
    nq = 1;
  }
  
  while (i*2 <= m_q) {
    
    // Get D^(i*2) from D^i and D^i
    op(D, D, D);
    if ((m_q - nq) % (i*4) > 0) {
      op(Dq, D, Dq);
      nq = nq + i*2;
    }    
    i = i * 2;
  }
  
  // Comparing elements of D^q and W^2, wherever dij = wij, mark eij as a link in the PFNET
  for (int j=0; j<n; j++) {
    for (int k=0; k<n; k++) 
    {  
      double *w = g->getEdge(j,k);
      
      if (w != NULL && (Dq[j][k] != *w ))
        g->removeEdge(j,k);
    }
  }
}

void TPathfinder::op(const Matrix &A, const Matrix &B, Matrix &result) const
{ 
  int n = A.size();
  Matrix C (n, Vector(n));
  
  for (int j=0; j<n; j++) {
    for (int k=0; k<n; k++) 
    {    
      Vector candidates, col;
      getCol(B, k, col);
      
      distances(A[j], col, candidates);
      candidates.push_back( A[j][k] );
      candidates.push_back( B[j][k] );
      
      // The resulting new 'distance' is the smallest element of the candidates list
      C[j][k] = *min_element(candidates.begin(), candidates.end());
    }
  }
  
  result.swap(C);
}

void TPathfinder::distances(const Vector &a, const Vector &b, Vector &result) const
{
  int n = a.size();
  Vector d;
  
  for (int i=0; i<n; i++) {
    d.push_back( distance(a[i], b[i]) );
  }
  
  result.swap(d);
}

double TPathfinder::distance(const double &a, const double &b) const
{
    // FIXME
    // I think there is some room for optimization here.
    // The value for 'infinity' can be put much lower -
    // calculating the maximum is much cheaper!!
    if (m_r >= 1 && m_r < INFINITY) 
    {
      return pow( pow(a, m_r) + pow(b, m_r), 1.0 / (double)m_r );
    }
    
    // r -> Infinity
    return max(a, b);
}

void TPathfinder::getCol(const Matrix &m, int i, Vector &col) const
{
  int n = m.size();
  Vector result (n);
  
  for (int j=0; j<n; j++) {
    result[j] = m[j][i];
  }
  
  col.swap(result);
}

void TPathfinder::toPFNET_dijkstra(TGraph *g)
{
  int n = g->nVertices;
  if (m_q < n-1) {
    BFS(g);
    return;
  }

  vector<int> neighbours;
  PQHeap<double> pq(n);                 // Priority queue, sorted by distance from the source vertex
  vector<DijkstraNode*> nodes(n);       // Node storage
  Matrix Dq (n, Vector(n, INFINITY));   // Resulting shortest distances
  
  DijkstraNode *v, *t, *z;              // Source node, current node and the current neighbour

  // Some statistics
  int nodesChecked = 0;
  
  // Create node objects
  for (int j=0; j<n; j++) {
     nodes[j] = new DijkstraNode(j, INFINITY);
  }

  for (int i=0; i<n; i++)
  {    
    for (int j=0; j<n; j++) {
      nodes[j]->setKey(INFINITY);
      nodes[j]->m_visited = false;
    }
    
    v = nodes[i];
    v->setKey(0);
    v->m_visited = true;
    
    pq.insert(v);
    
    while (!pq.empty())
    {
      t = static_cast<DijkstraNode*>(pq.deleteMin());
      g->getNeighboursFrom(t->m_v, neighbours);
      
      for (int k=0; k < neighbours.size(); k++) 
      {
        z = nodes[neighbours[k]];
#ifdef STATISTICS
        nodesChecked++;
#endif
        double newDist = distance(t->getKey(), *g->getEdge(t->m_v, z->m_v));
        
        // Visit the neighbour only if it is along the allowed number of links!
        if (!z->m_visited) {
          z->m_visited = true;
          z->setKey(newDist);
          pq.insert(z);
        }
        // If we find a shorter path, we can use it only if the path is along the allowed number of links!
        else if (newDist < z->getKey()) {
          pq.decreaseKey(z, newDist);
        }
      }
    }
    
    // The result of each Dijkstra's calculation is a new line in the target matrix
    for (int k=0; k<n; k++)
    {
      Dq[i][k] = nodes[k]->getKey();
    }
  }

  // Comparing elements of D^q and W^2, wherever dij = wij, mark eij as a link in the PFNET
  for (int j=0; j<n; j++) {
    for (int k=0; k<n; k++) 
    {  
      double *w = g->getEdge(j,k);
      
      if (w != NULL && (Dq[j][k] != *w ))
        g->removeEdge(j,k);
    }
  }

#ifdef STATISTICS
  cout << "Dijkstra: " << nodesChecked << " nodes checked" << endl;
#endif
}

void TPathfinder::toPFNET_sparse(TGraph *g)
{
  int n = g->nVertices;
  if (m_q < n-1) {
    BFS(g);
    return;
  }

  int neighIdx;
  vector<int> T;                        // Source node's neighbours
  vector<int> neighbours;
  vector<int> dirty;                    // Which nodes to reset after an iteration
  vector<SparseNode*> nodes(n);         // Node storage
  PQHeap<double> pq(n);                 // Priority queue, sorted by distance from the source vertex

  SparseNode *v, *u, *t;

  // Some statistics
  int nodesChecked = 0;

  // Prepare initial node values
  for (int i=0; i<n; i++)
  {
    nodes[i] = new SparseNode(i, INFINITY, OUTER);
  }
  
  for (int i=0; i<n; i++)
  {
    // Select a new source node
    v = nodes[i];
    v->setKey(0);
    //v->m_len = 0;
    g->getNeighboursFrom(v->m_v, T);

    // Reset the needed data structures
    dirty.clear();
    pq.makenull();

    pq.insert(v);
    v->m_state = ACTIVE;
    dirty.push_back(v->m_v);

    while (!T.empty())
    {
      u = static_cast<SparseNode*>(pq.deleteMin());
      u->m_state = COMPLETED;
      
      neighIdx = -1;
      // Is u a neighbour of v?
      for (int j=0; j<T.size(); j++) {
        if (T[j] == u->m_v) {
          neighIdx = j;
          break;
        }
      }
      
      if(neighIdx >= 0)
      {
        // The shortest distance to u is calculated, delete it from T
        T.erase(T.begin() + neighIdx);

        // Remove the edge if it isn't the shortest path between the vertices
        if (u->getKey() != *g->getEdge(v->m_v, u->m_v))
          g->removeEdge(v->m_v, u->m_v);
      }

      //if (u->m_len >= m_q) continue;
      
      g->getNeighboursFrom(u->m_v, neighbours);
      for (int j=0; j<neighbours.size(); j++)
      {
        t = nodes[neighbours[j]];
#ifdef STATISTICS
        nodesChecked++;
#endif
        if (t->m_state != COMPLETED) 
        {
          double newDist = distance(u->getKey(), *g->getEdge(u->m_v, t->m_v));
          //int newLen = u->m_len + 1;
          
          if (t->m_state == OUTER)
          {
            t->setKey(newDist);
            //t->m_len = newLen;
            t->m_state = ACTIVE;
            dirty.push_back(t->m_v);
            pq.insert(t);
          }
          else if (t->getKey() > newDist)
          {
            //t->m_len = newLen;
            pq.decreaseKey(t, newDist);
          }
        }
      }
    } // End while
    for (int j=0; j<dirty.size(); j++)
    {
      nodes[dirty[j]]->m_state = OUTER;
    }
  }

#ifdef STATISTICS
  cout << "Sparse PF: " << nodesChecked << " nodes checked" << endl;
#endif
}

void TPathfinder::BFS(TGraph *g)
{
  int n = g->nVertices;
  queue<BFSNode*> q;
  vector<double> dist(n);
  vector<int> neighbours;
  vector<int>::iterator it;
  Matrix Dq (n, Vector(n, INFINITY));   // Resulting shortest distances

  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++) {
      dist[j] = INFINITY;
    }
    
    double dMax = 0;
    g->getNeighboursFrom(i, neighbours);

    for (it=neighbours.begin(); it<neighbours.end(); it++) {
      double w = *g->getEdge(i, *it);
      if (w > dMax) 
        dMax = w;
    }

    q.push(new BFSNode(i,0,0));
    dist[i] = 0;

    while(!q.empty())
    {
      BFSNode *u = q.front();
      q.pop();
      int newLen = u->m_len + 1;

      // This limits the search depth
      if (newLen <= m_q) 
      {
        g->getNeighboursFrom(u->m_idx, neighbours);
        for (it=neighbours.begin(); it<neighbours.end(); it++) 
        {
          int t = *it;
          double dNew = distance(u->m_d, *g->getEdge(u->m_idx, t));
          if (dNew <= dMax && dNew < dist[t]) {
            dist[t] = dNew;
            q.push(new BFSNode(t, dNew, newLen));
          }
        }
      }
    }
    
    // Save all the calculated distances into the target matrix Dq
    for (int j=0; j<n; j++)
      Dq[i][j] = dist[j];
  }

  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) 
    {  
      double *w = g->getEdge(i,j);
      
      if (w != NULL && (Dq[i][j] != *w))
        g->removeEdge(i,j);
    }
  }
}

/*     ----------       Python interface     ---------       */

#include "externs.px"
#include "orange_api.hpp"
WRAPPER(Orange);
PyObject *Pathfinder_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON (Orange, "(r, q)")
{
  PyTRY
    int r=-1, q=-1;
  
    if (PyArg_ParseTuple(args, "ii", &r, &q)) {
      return WrapNewOrange(mlnew TPathfinder(r, q), type);
    }
    
    PyErr_Format(PyExc_TypeError, "Pathfinder.__new__: Wrong argument type");
    return PYNULL;
    
  PyCATCH;
}

PyObject *Pathfinder_toPFNET(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "g -> None")
{
  PyTRY
    CAST_TO(TPathfinder, pf);
    
    PyObject *pygraph;
    TGraph *g;
    
    if (!PyArg_ParseTuple(args, "O:Graph", &pygraph))
      return PYNULL;
    
    g = PyOrange_AsGraph(pygraph).getUnwrappedPtr();
    pf->toPFNET(g);
    
    RETURN_NONE;
  PyCATCH;
}

PyObject *Pathfinder_toPFNET_dijkstra(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "g -> None")
{
  PyTRY
    CAST_TO(TPathfinder, pf);
    
    PyObject *pygraph;
    TGraph *g;
    
    if (!PyArg_ParseTuple(args, "O:Graph", &pygraph))
      return PYNULL;
    
    g = PyOrange_AsGraph(pygraph).getUnwrappedPtr();
    pf->toPFNET_dijkstra(g);
    
    RETURN_NONE;
  PyCATCH;
}

PyObject *Pathfinder_toPFNET_sparse(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "g -> None")
{
  PyTRY
    CAST_TO(TPathfinder, pf);
    
    PyObject *pygraph;
    TGraph *g;
    
    if (!PyArg_ParseTuple(args, "O:Graph", &pygraph))
      return PYNULL;
    
    g = PyOrange_AsGraph(pygraph).getUnwrappedPtr();
    pf->toPFNET_sparse(g);
    
    RETURN_NONE;
  PyCATCH;
}

#include "px/pathfinder.px"
