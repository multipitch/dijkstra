import itertools
import math
from heapq import heapify, heappush, heappop

import networkx as nx

################################################################################
################################################################################
##                                                                            ##
##  My implementation of Dijkstra's Algorithm and the Bidirectional variant   ##
##  thereof.                                                                  ##
##             -- Sean Tully 09/03/2017                                       ##
##                Student No. 15205062                                        ##
##                                                                            ##
##  The starting point for both alogrithms was the file                       ##
##  'dijkstra-Prioroty_queue.py' by James McDermott, given in lectures.       ##
##                                                                            ##
##  Follows nomenclature given in lectures.                                   ##
##  Uses PriorityQueue class by James McDermott, given in lectures.           ##
##  Functions (can) return values similar to the NetworkX implementations     ##
##  to facilitate comparison.                                                 ##
##                                                                            ##
##  Includes test suite and functions to generate data and plots for use in   ##
##  report.                                                                   ##
##                                                                            ##
################################################################################
################################################################################


def dijkstra(G, r, style='nx'):
    """Dijkstra's algorithm, single source implementation.
    
    An implementation of Dijkstra's algorithm that returns the shortest
    paths in a graph from a root node to all other nodes.  A priority
    queue structure is used.
    
    Args:
        G (nx.Graph): A weighted, unidirectional graph.
        r (obj): Identifier for root node.
        style (str, optional): Sets return style - see Returns.
        
    Returns:
        If style == 'nx' (default):
            dict: Shortest path distances, keyed by target node.
            dict: Sortest path routes, as lists of nodes, keyed by
                target node.
        If style == 'mis40550':
            set: Shortest path spanning tree.
            dict: Parent-pointers, keyed by child node.
            dict: Shortest path distances, keyed by target node.
    """

    if not G.has_node(r):
        raise ValueError('Source node {} not present'.format(r))

    for e1, e2, d in G.edges(data=True):
        if d['weight'] < 0:
            raise ValueError('Negative weight on edge {}-{}'.format(e1, e2))
            
    if style not in ('mscba', 'nx'):
       raise ValueError('Invalid Style: Choose \'nx\' (default) or \'mscba\'.')

    P = {r} # permanent set
    S = PriorityQueue() # V-P. This is the crucial data structure.
    D = {} # estimates of SPST lengths
    p = {} # parent-pointers
    A = set() # our result, an SPST
    
    # Build priority queue
    for n in G.nodes():
        if n == r:
            D[n] = 0
        else:
            if G.has_edge(r, n):
                D[n] = G[r][n]['weight']
            else:
                D[n] = math.inf
            p[n] = r
            S.add_task(n, D[n])

    while len(S):
        u, Du = S.pop_task() # selects u from S with min Du
        if u in P: continue  
        P.add(u) # move one item to the permanent set P and to SPST A
        A.add((p[u], u)) # add (predecessor_of_u, u) to SPST

        for v, Dv in S: # relaxation loop
            if v in P: continue
            if G.has_edge(u, v):
                if D[v] > D[u] + G[u][v]['weight']:
                    D[v] = D[u] + G[u][v]['weight']
                    p[v] = u
                    S.add_task(v, D[v]) # add v, or update its prio
    
    if style == 'mis40550': # return as per example code in lectures
        return A, p, D
        
    # emulate NetworkX return style, incl. removal of unreachable paths
    distance = D
    path = p_to_paths(r, p)
    unreachables = [key for key, value in D.items() if value is math.inf]
    for i in unreachables:
        del distance[i]
        del path[i]
    return distance, path
        

def bidirectional_dijkstra(G, r, s, data=False):
    """Dijkstra's algorithm, bidirectional implementation.
    
    An implementation of the bidirectional variant of Dijkstra's
    algorithm that returns the shortest path in a graph from a root
    node to a target node. A priority queue structure is used.
    
    Args:
        G (nx.Graph): A weighted, unidirectional graph.
        r (obj): Identifier for root node.
        s (obj): Identifier for target node.
        data (bool, optional):  Return additional data.
        
    Returns:
        If data == False (default):
            number: Distance from root node to target node.
            list: Shortest path from root node to target node,
                expressed as a list of nodes.
        If data == True:
            number: Distance from root node to target node.
            list: Shortest path from root node to target node,
                expressed as a list of nodes.
            tuple: Internal data (P, S, D, p, link).       
    """

    if r == s:  # No search required in this case
        if data:
            return 0, [r], None
        return 0, [r]
    
    for n, typ in ((r, 'Source'), (s, 'Target')):    
        if not G.has_node(n):
            raise ValueError('{} node {} not present'.format(typ, n))

    for n1, n2, d in G.edges(data=True):
        if d['weight'] < 0:
            raise ValueError('Negative weight on edge {}-{}'.format(n1, n2))
                             
    P = [{r}, {s}] # forward, reverse permanent sets
    S = [PriorityQueue(), PriorityQueue()] # forward, reverse V/P queues
    D = [{}, {}] # estimates of forward, reverse SPST lengths
    p = [{}, {}] # forward, reverse parent-pointers
    
    # Build starting data.
    for direc, root in enumerate((r, s)):
        for n in G.nodes():
            if n == root:
                D[direc][n] = 0
            else:
                if G.has_edge(root, n):
                    D[direc][n] = G[root][n]['weight']
                else:
                    D[direc][n] = math.inf
                p[direc][n] = root
                S[direc].add_task(n, D[direc][n])

    # Search in forward and reverse directions.
    direction = False # forward == True; reverse == False
    u = None
    while u is None or not (u in P[0] and u in P[1]):
        direction = not direction # Set direction for this iteration
        d = int(direction)        # current direction index
        e = int(not direction)    # other direction index
        if len(S[d]):
            u, Du = S[d].pop_task()
            if u in P[d]: continue

            P[d].add(u) # move one item to the permanent set P

            for v, Dv in S[d]: # relaxation loop
                if v in P[d]: continue
                if G.has_edge(u, v):
                    if D[d][v] > D[d][u] + G[u][v]['weight']:
                        D[d][v] = D[d][u] + G[u][v]['weight']
                        p[d][v] = u
                        S[d].add_task(v, D[d][v])

    # Find SP distance and link node.    
    distance = D[0][u] + D[1][u] # this will become our SP distance
    link = u
    for n in G.nodes():
        dist = D[0][n] + D[1][n]
        if dist < distance:
            distance = dist
            link = n

    if link is None or distance == math.inf:
                raise ValueError('No path between {} and {}.'.format(r, s))
                
    # Generate path.
    path = p_to_path(r, link, p[0]) # get the forward half of the path
    path = path + p_to_path(s, link, p[1])[-2::-1] # add reverse half
    
    # Return results.
    if data: # return all data - useful for plotting etc.
        return distance, path, (P, S, D, p, u, link) 
    return distance, path


def p_to_path(r, s, p):
    """Creates a path from r to s given a parent-pointer list, p.
    """
    
    path = [s]
    n = s
    while n is not r:
            n = p[n]
            path.append(n)
    return path[::-1]


def p_to_paths(r, p):
    """Creates paths from r to all nodes given a parent-pointer list, p.
    """
    
    path = {r: [r]}
    for n in list(p.keys()):
        npath = [n]
        m = n
        while m is not r:
            m = p[m]
            npath.append(m)
        path[n] = npath[::-1]
    return path
  
    
################################################################################
################################################################################
##                                                                            ##
##  Code below copied from priority_queue.py given in lectures.               ##
##                                                                            ##
################################################################################
################################################################################


class PriorityQueue:
    """A priority queue implementation based on std. lib. heapq module.
    
    Taken from https://docs.python.org/2/library/heapq.html, but
    encapsulated in a class. Also iterable, printable, and len-able.
    """

    REMOVED = '<removed-task>' # placeholder for a removed task


    def __init__(self, tasks_prios=None):
        self.pq = []
        self.entry_finder = {} # mapping of tasks to entries
        self.counter = itertools.count() # unique sequence count,
                                         # -- tie-breaker when prios equal
        if tasks_prios:
            for task, prio in tasks_prios:
                # would be nice to use heapify here instead
                self.add_task(task, prio) 


    def __iter__(self):
        return ((task, prio) for (prio, count, task)
                in self.pq if task is not self.REMOVED)


    def __len__(self):
        return len(list(self.__iter__()))


    def __str__(self):
        return str(list(self.__iter__()))


    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)


    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED


    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                # NB a change from the original: we return prio as well
                return task, priority 
        raise KeyError('pop from an empty priority queue')


################################################################################
################################################################################
##                                                                            ##
##  Test suite and functions to generate data and plots for use in report.    ##
##                                                                            ##
################################################################################
################################################################################

if __name__ == "__main__":    
    import csv
    import random
    import time
   
    import numpy as np
    import matplotlib.pyplot as plt
       
    # Create simpple graph for use in test suites
    G = nx.Graph()
    E = (('A', 'B', 2),
         ('A', 'C', 6),
         ('A', 'D', 8),
         ('B', 'G', 10),
         ('B', 'C', 8),
         ('C', 'D', 1),
         ('C', 'E', 5),
         ('C', 'G', 9),
         ('C', 'F', 3),
         ('D', 'F', 9),
         ('G', 'E', 4),
         ('E', 'F', 1))
    G.add_weighted_edges_from(E)    

    
    def test1(G):
        """Test unidirectional dijkstra against small, defined graphs.
        """
    
        print('\nTest 1:  ', end='')
        f1 = dijkstra
        f2 = nx.single_source_dijkstra
        G1 = G.copy() # don't modify original
      
        ok = True
        # Test behavoiur for connected graph given above
        for r in G1.nodes():
                ok = ok and f1(G1, r)[0] == f2(G1, r)[0]
                if not ok:
                    break
        # Test behaviour when non-connected node added
        G1.add_node('X')
        for r in G1.nodes():
                ok = ok and f1(G1, r) == f2(G1, r)
                if not ok:
                    break
     
        if ok:
            print('Pass')
        else:
            print('Fail')


    def test2(G):
        """Test bidirectional dijkstra against small, defined graphs.
        """
    
        print('\nTest 2:  ', end='')
        f1 = bidirectional_dijkstra
        f2 = nx.bidirectional_dijkstra
        G1 = G.copy() # don't modify original
              
        ok = True
        # Test behaviour for connected graph given above
        for r in G1.nodes():
            for s in G1.nodes():
                    ok = ok and f1(G1, r, s)[0] == f2(G1, r, s)[0]
                    if not ok:
                        break
        # Test behaviour when non-connected node added
        G1 = G.copy()
        G1.add_node('X')
        try:
            distance, path = f1(G1, 'A', 'X') # This should fail
            print(distance)
            print(path)
            ok = False
        except:
            pass
            
        if ok:
            print('Pass')
        else:
            print('Fail')

     
    def test3(n=30):
        """Test unidirectional Dijkstra against large random graph.
        """
    
        print('\nTest 3:  ', end='')

        f1 = dijkstra
        f2 = nx.single_source_dijkstra

        G = RCUEWG(n)
        ok = True
        for r in G.nodes():
            ok = ok and f1(G, r)[0] == f2(G, r)[0]
            if not ok:
                break
        if ok:
            print('Pass')
        else:
            print('Fail')


    def test4(n=30):
        """Test bidirectional Dijkstra against large random graph.
        """
    
        print('\nTest 4:  ', end='')

        f1 = bidirectional_dijkstra
        f2 = nx.bidirectional_dijkstra

        G = RCUEWG(n)
        ok = True
        for r in G.nodes():
            for s in G.nodes():
                ok = ok and f1(G, r, s)[0] == f2(G, r, s)[0]
                if not ok:
                    break
            if not ok:
                break
        if ok:
            print('Pass')
        else:
            print('Fail')


    def make_table(fname):
        """Create some timing data for a table.
        """
        
        print('\nCreating Data File \'{}.csv\':  '.format(fname), end='')
        table = []
        fname = fname + '.csv'
        for N in [10,20,50,100,200,500,1000]:
            G = RCUEWG(N)
            start1 = time.time()
            distance, path = dijkstra(G, 0)
            end1 = time.time()
            start2 = time.time()
            distance, path = bidirectional_dijkstra(G, 0, N-1)
            end2 = time.time()
            table.append((N, G.size(), end1 - start1, end2 - start2))
        with open(fname,'w') as f:
            o=csv.writer(f)
            row0 = ['nodes', 'edges', 'dijkstra', 'bidirectional_dijkstra']
            o.writerow(row0)
            for row in table:
                o.writerow(row)
        print('Done')


    def read_data(fname):
        """Read node and edge data from 'fname.cnode' and 'fname.cedge'.
        
        These files can be downloaded from:
        https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm
        
        Reference:
        Li, F., Cheng, D., Hadjieleftheriou, M., Kollios, G. and Teng, S.H.,
        2005, August. On trip planning queries in spatial databases.
        In International Symposium on Spatial and Temporal Databases
        (pp. 273-290). Springer Berlin Heidelberg.
        """
        
        nodefile = fname + '.cnode'
        edgefile = fname + '.cedge'
        print('\nReading From Data Files \'{}.csv\' and \'{}.csv\':  '
              .format(nodefile, edgefile), end='')    
        G = nx.Graph()
        G.position = {}
        try:
            with open(nodefile, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for n, x, y in reader:
                    G.add_node(n)
                    G.position[n] = (float(x), float(y))
        except:
            raise IOError('File {} can\'t be found.  Dowload from '
                          'https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm'
                          .format(nodefile))
        try:
            with open(edgefile, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for _, r, s, d in reader:
                    G.add_edge(r, s, weight=float(d))
        except:
            raise IOError('File {} can\'t be found.  Dowload from '
                          'https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm'
                          .format(edgefile))
        print('Done')
        return G


    def make_plot(G, source, target, fname):
        """Create plot, 'fname.pdf' showing forward and reverse search.
        
        Loosely based on the Knuth Miles example in the NetworkX
        documentation, but modified to work with data in the format
        generated by read_data and using the implementation of
        bidirectional_dijkstra used in this file.
        see:
        /usr/share/doc/python-networkx/examples/graph/knuth_miles.py 
        or:
        http://networkx.readthedocs.io/en/stable/examples/graph/knuth_miles.html
        """

        print('\nCreating Plot \'{}.pdf\' (may take several minutes):  '
              .format(fname), end='')        
        distance, path, data = bidirectional_dijkstra(G, source, 
                                                      target, data=True)
        ntype = []
        for node in G.nodes():
            if node == source:
                ntype.append(1) # source
            elif node == target:
                ntype.append(2) # target
            elif node == data[4]:
                if node != data[5]:
                    ntype.append(3) # halt
                else:
                    ntype.append(4) # link
            elif node == data[5]:
                ntype.append(4)  # link
            elif node in path:
                ntype.append(5) # SP
            elif node in data[0][0]:
                ntype.append(6) # fwd
            elif node in data[0][1]:
                ntype.append(7) # rev
            else:
                ntype.append(0) # not searched
        etype = []
        for edge in G.edges():
            if edge[0] in path and edge[1] in path:
                etype.append(5) # SP
            elif edge[0] in data[0][0] and edge[1] in data[0][0]:
                etype.append(6) # fwd
            elif edge[0] in data[0][1] and edge[1] in data[0][1]:
                etype.append(7) # rev
            else:
                etype.append(0) # other
        nstyle = {
        # priority: (label, colour, shape, weight)
            0: ('not searched', 'gray','o', 10), # not searched
            1: ('source', 'red','v', 100), # source
            2: ('target', 'red','^', 100), # target
            3: ('halt', 'cyan','d', 100), # halt
            4: ('link', 'orange','s', 100), # link
            5: ('SP', 'red','o', 60), # SP
            6: ('fwd', 'blue','>', 10), # fwd
            7: ('rev''purple','<', 10) # rev
        }
        estyle = {
        # priority: (label, colour, weight)
            0: ('other', 'gray', 1), # other
            5: ('SP', 'red', 3), # SP
            6: ('fwd', 'blue', 2), # fwd
            7: ('rev', 'purple', 2) # rev
        }

        plt.figure(figsize=(6,6))
        for n in (1, 2, 4, 3): # order is important here - halt on top of link
            nl = [x for i, x in enumerate(G.nodes()) if ntype[i] == n]
            nx.draw_networkx_nodes(G, G.position, node_color=nstyle[n][1], 
                                   label=nstyle[n][0], nodelist=nl, 
                                   with_labels=True, linewidths=1,
                                   node_shape=nstyle[n][2], 
                                   node_size=nstyle[n][3])
        for e in (0, 6, 7, 5): # order is important here - want SP "on top"
            el = [x for i, x in enumerate(G.edges()) if etype[i] == e]
            nx.draw_networkx_edges(G, G.position, edgelist=el,
                                   edge_color=estyle[e][1], 
                                   width=estyle[e][2])                           
        plt.legend(scatterpoints=1)
        plt.axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(fname + '.pdf', format='pdf')
        print('Done')


    def RCUEWG(n): # Taken from lectures
        """RCUEWG stands for random complete undirected
        edge-weighted graph, obviously."""
        
        M = np.random.random((n, n))
        G = nx.Graph()
        for i in range(n):
            # notice we are using the upper triangle only
            # that is we discard M_ii and M_ji where j<i
            for j in range(i+1, n):
                G.add_edge(i, j, weight=M[i, j])
        return G


    # Run tests.
    test1(G)
    test2(G)
    test3(30)
    test4(30)
    
    # Generate outputs for report. 
    #make_table('table1')
    #G = read_data('cal')
    #make_plot(G, '204', '20800', 'fig1')  # Northwest to Southeast
    #make_plot(G, '7515', '14223', 'fig2') # Short path in middle of graph

