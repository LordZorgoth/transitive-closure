import numpy as np
import time

# The easiest way to test transitive_closure is to use the premade
# transitive_closure_function_test(). Arguments can be passed
# (see documentation), but none are needed and it has good
# defaults. Do make sure to ask for the return values when you
# call it, or you could get a very long terminal output...

# To convince yourself that we really are getting the transitive
# closure, you can use the short function is_transitive() and the
# parameter check_accuracy=True in test_transitive_closure_function.
# The parameter calls is_transitive() on the final output of
# transitive_closure(), and also causes verify_edges() to be invoked
# after every step of the primary loop, which double checks that the
# new edges are all things that must be in the transitive closure.


def transitive_closure(E, N, verbose=False, update_period=10000,
                       update_mode='by_time', check_edges=False):
    """
    transitive_closure(E, N, verbose=False, update_period=10000,
                       update_mode='by_time', check_edges=False)

    Compute the transitive closure of a relation in place. The domain
    of the relation is N, while E is the set of ordered pairs
    comprising the relation.

    Parameters
    ----------
    N : set
        the domain of the relation
    E : a set of 2-tuples, whose elements must be in N
        the set of ordered pairs or edges comprising the relation
    verbose : bool
        if True, will print various information
    update_period : int
        only effective if verbose == True; determines how 
        often to post updates
    check_edges : bool
        if True, we double check that no edges are being added that
        should not have been
    update_mode : string
        if 'by_time' and verbose == True, will post updates
          every update_period milliseconds
        if 'by_nodes' and verbose == True, will post updates
          every update_period node iterations
        if 'by_edges' and verbose == True, will post updates
          for every update_period edge candidates processed
        otherwise, does not post updates except at end of
          primary_loop_step()


    Returns
    -------
    None. This function alters the set E in place. Note that if the
    function is interrupted, E will most likely become an empty set.
    """
    # For our algorithm's internal representation of edges, we will use
    # dictionaries with keys that are nodes and values that are sets of
    # nodes so as to be able to rapidly look up values
    # and test membership.
    if verbose:
        print("Initializing...")
    X = {node: set() for node in N}
    Y = {node: set() for node in N}
    # We use this while loop to iterate over E, removing elements as we
    # go to conserve memory. All information in E is contained in
    # each of X and Y. When we run the loop, X will lose information
    # due to an optimization, so ultimately, we'll reconstruct E from Y.
    while E != set():
        edge = E.pop()
        X[edge[1]].add(edge[0])
        Y[edge[0]].add(edge[1])
    # On the first iteration, we will loop over all nodes, but we will
    # only loop over a subset of nodes, N_marked, in future iterations.
    N_marked = N
    if verbose:
        # stats will be a dictionary we use to record information
        # for reporting to the user
        stats = initialize_stats_dict(update_period=update_period,
                                      update_mode=update_mode)
    else:
        stats = None
    # This is the primary loop. It terminates when no new edges
    # were added in the last step.
    while N_marked != set():
        # The function primary_loop_step() finds all edges (x, z)
        # such that there is a y for which X[x] == y and Y[y] == z,
        # and further returns the set of termini of such edges.
        # It adds all of these edges to Y in place in addition to
        # returning them in X_new.

        X_new, N_new_edge_termini = primary_loop_step(
            X, Y, N_marked,
            verbose=verbose, stats=stats, check_edges=check_edges
            )

        # Our most important optimization comes from realizing that we
        # only need to iterate over nodes where a new edge terminates.
        # For a given node x in N, we still must consider all of the
        # edges emanating from x. Furthermore, we can omit old edges
        # that terminate at x from consideration. It can be proven by
        # induction that these optimizations never prevent an edge
        # from being added that would have been added with the full
        # N and X:

        # Let A be our proposed algorithm, in which we only iterate over
        # nodes that are the terminus of a new edge, and for such a node
        # x, we iterate over all edges emanating from x but only over
        # those edges terminating at x that were added in the previous
        # time step; when iterating over edges, the procedure is that
        # for each edge (w,x) terminating at x, we iterate over all edges
        # (x,y) emanating from x, and we add any edges (w,y) that are
        # not already present.
        #     We seek to prove, for n>=1, the statement P(A,n):
        # that at any step n of the primary loop of A, that for any pair
        # (x,y), (y,z) of edges present before step n, the edge (x,z) is
        # present before step n+1. This is trivial for n=1,
        # since we iterate over all nodes and edges on the first step.

        # Assume that n>1 and that P(A,n-1) is true.
        # For step n, there are three cases to consider for an arbitrary
        # pair of edges (x,y) and (y,z):
        #   Case 1: (x,y) was added in step n-1.
        #     In this case, we will add the edge (x,z) when we iterate
        #     over the node y in step n.
        #   Case 2: (x,y) and (y,z) were both present before step n-1.
        #     In this case, P(A,n-1) implies that (x,z) is already
        #     present before step n.
        #   Case 3: (y,z) was added in step n-1, and (x,y) is older.
        #     For this to be the case, there must be a node w
        #     such that (y,w) and (w,z) were present before step n-1.
        #     P(A,n-1) implies that there is an edge (x,w) at step n.
        #     If (x,w) was present before step n-1, the edge (x,z)
        #     is already present at step n by P(A,n-1). If (x,w) was
        #     added in step n-1, (x,z) will be added when we iterate
        #     over the node w in step n.

        # We therefore replace N_marked with N_new_edge_termini
        # and X with X_new (this second assignment is also a
        # memory optimization).

        N_marked = N_new_edge_termini
        X = X_new

    # We still need to reconstruct E from Y
    if verbose:
        print("\nAlgorithm complete after %s steps, reconstructing E..."
              % stats['steps'])
    while Y != {}:
        values = Y.popitem()
        x = values[0]
        yset = values[1]
        while yset != set():
            y = yset.pop()
            E.add((x, y))


def primary_loop_step(X, Y, N_marked,
                      verbose=False, stats=None, check_edges=False):
    """
    primary_loop_step(X, Y, N_marked,
                      verbose=False, stats=None, check_edges=False)

    This function is the computation step of the primary loop. It finds 
    pairs (x, z) such that there exists y in N such that
    (x, y) and (y, z) are both already edges, and returns them 
    in X_new and adds them in place to Y. It also returns the set 
    N_new_edge_termini, consisting of nodes where new edges terminate.

    Parameters
    ----------
    X : dictionary
        X maps a member "node" of N to a set X[node] of members of N 
        such that (X[node], node) are edges. Note that in practice, 
        this X will not in general contain all edges, but only edges 
        that were added in the last step of the primary loop.
    Y : dictionary
        Y maps a member "node" of N to a set Y[node] of members of N 
        such that (node, Y[node]) are edges. Y contains all edges. 
        This function modifies Y in place to add new edges.
    N_marked : subset of N
        The set of nodes for which we seek connections
    verbose : bool
        Determines whether to output general information
    stats : dict or None
        If verbose is not None, this should be a dictionary for use
        in recording information to be reported to the user.
    check_edges : bool
        If check_edges is True, the function will obligingly
        reassure the user that all new edges are in fact edges
        that should have been added.

    Returns
    -------
    X_new : dictionary
        the dictionary of new edges, in the format of X
    N_new_edge_termini : subset of N
        the set of nodes at which new edges terminate
    """
    if verbose:
        stats['new_edges_count'] = 0
        stats['steps'] += 1
        stats['edge_candidates_processed_in_step'] = 0
        print("Algorithm step %s...\n" % stats['steps'])
    if check_edges:
        new_edges_by_node = {}
    else:
        new_edges_by_node = None
    # The real function starts here
    X_new = {}
    Y_new = {}
    N_new_edge_termini = set()
    for node in N_marked:
        # iterate_over_node modifies X_new, Y_new, and new_edges_by_node
        # in place to add new edges (x, y) such that (x, node) and
        # (node, y) are also edges. It also modifies other arguments;
        # see documentation.
        #     primary_loop_step would obviously be faster if we put the
        # code of iterate_over_node inline, but I'm focusing on
        # readability over optimality, especially because there isn't a
        # real purpose to this module.
        iterate_over_node(node, X, Y, X_new, Y_new, N_new_edge_termini,
                          stats=stats, new_edges_by_node=new_edges_by_node)
        if verbose:
            stats['nodes_iterated'] += 1
            consider_posting_update(stats)
    if check_edges:
        verify_edges(Y, new_edges_by_node, N_marked, verbose=verbose)
    # Update Y
    for node in Y_new.keys():
        Y[node] = Y[node].union(Y_new[node])
    if verbose:
        print("\nAdded %s edges out of %s candidates."
              % (stats['new_edges_count'],
                 stats['edge_candidates_processed_in_step']))
    return X_new, N_new_edge_termini


def iterate_over_node(node, X, Y, X_new={}, Y_new={}, N_new_edge_termini=set(),
                      stats=None, new_edges_by_node=None):
    """
    iterate_over_node(node, X, Y, X_new={}, Y_new={}, 
                      N_new_edge_termini=set(),
                      stats=None, new_edges_by_node=None)

    This function finds edges (x, y) such that x in X[node] and 
    y in Y[node], which only occurs if (x, node) and (node, y) are
    both edges. It adds these edges to X_new and Y_new.

    Parameters
    ----------
    node : a member of N 
        the node we are iterating over
    X : dictionary
        X maps a member "node" of N to a set X[node] of members of N 
        such that (X[node], node) are edges. Note that this X will not
        in general contain all edges, but only edges that were added
        in the last step of the primary loop
    Y : dictionary
        Y maps a member "node" of N to a set Y[node] of members of N 
        such that (node, Y[node]) are edges. Y contains all edges.
    X_new : dictionary
        We record new edges in X_new, in the format of X.
    Y_new : dictionary
        We also record new edges in Y_new, in the format of Y.
    N_new_edge_termini : subset of N
        We record nodes that are the terminus of a new edge
        in N_new_edge_termini.
    stats : dictionary or None
        if not None, a dictionary for storing information for reporting
        to the user
    new_edges_by_node : a dictionary, or None
        if not None, the function will add a key equal to node, whose
        value will be a set of all edges added

    Returns
    -------
    Note that there is no need to collect the return values of this
    function unless X_new, Y_new, and N_new_edge_termini were not passed
    when the function was called.

    X_new : as above
    Y_new : as above
    N_new_edge_termini: as above
    """
    if new_edges_by_node is not None:
        new_edges_by_node[node] = set()
    for x in X[node]:
        for y in Y[node]:
            # Below is the procedure for recording information
            # about a single new edge.
            if y not in Y[x] and (x not in Y_new.keys() or y not in Y_new[x]):
                if y not in X_new.keys():
                    X_new[y] = {x}
                else:
                    X_new[y].add(x)
                if x not in Y_new.keys():
                    Y_new[x] = {y}
                else:
                    Y_new[x].add(y)
                N_new_edge_termini.add(y)
                if new_edges_by_node is not None:
                    new_edges_by_node[node].add((x, y))
                if stats is not None:
                    stats['new_edges_count'] += 1
            if stats is not None:
                stats['edge_candidates_processed'] += 1
                stats['edge_candidates_processed_in_step'] += 1
    return X_new, Y_new, N_new_edge_termini

# The functions below are called by transitive_closure() or its
# subfunctions, but are not part of the algorithm; they are for
# presenting data to the user and for verification purposes.

def initialize_stats_dict(update_period=10000, update_mode='by_time'):
    """
    This function initializes a dictionary used when recording the data
    we output when verbose == True.
    """
    stats = {}
    stats['nodes_iterated'] = 0
    stats['edge_candidates_processed'] = 0
    stats['steps'] = 0
    stats['edge_candidates_processed_in_step'] = 0
    stats['update_period'] = update_period
    stats['new_edges_count'] = 0
    stats['update_mode'] = update_mode
    stats['previous_nodes_iterated'] = 0
    stats['last_update_time'] = time.time()
    return stats


def consider_posting_update(stats):
    """Post an update at the appropriate time."""
    if stats['update_mode'] == 'by_nodes':
        if stats['nodes_iterated'] % stats['update_period'] == 0:
            print("%s total primary node iterations..."
                  % stats['nodes_iterated'])
            print(("In last %s node iterations,"
                   + "analyzed %s potential new edges.")
                  % (stats['update_period'],
                     stats['edge_candidates_processed']))
            stats['edge_candidates_processed'] = 0
    if stats['update_mode'] == 'by_edges':
        if stats['edge_candidates_processed'] >= stats['update_period']:
            print("%s total primary node iterations..."
                  % stats['nodes_iterated'])
            print(("In last %s node iterations,"
                   + "analyzed %s potential new edges.")
                  % (stats['nodes_iterated'] - stats['previous_nodes_iterated'],
                     stats['edge_candidates_processed']))
            stats['previous_nodes_iterated'] = stats['nodes_iterated']
            stats['edge_candidates_processed'] = 0
    if stats['update_mode'] == 'by_time':
        time_elapsed = time.time()-stats['last_update_time']
        if time_elapsed > stats['update_period']/1000:
            stats['last_update_time'] = time.time()
            print("%s total primary node iterations..."
                  % stats['nodes_iterated'])
            print(("In last %s node iterations,"
                   + "analyzed %s potential new edges.")
                  % (stats['nodes_iterated'] - stats['previous_nodes_iterated'],
                     stats['edge_candidates_processed']))
            stats['previous_nodes_iterated'] = stats['nodes_iterated']
            stats['edge_candidates_processed'] = 0


def verify_edges(Y, new_edges_by_node, N_marked, verbose=False):
    """
    This function verifies that all edges added in a primary_loop_step
    are clearly required to create a transitive closure.
    """
    edges_are_ok = True
    if verbose:
        print("\nVerifying new edges...")
    for node in N_marked:
        for edge in new_edges_by_node[node]:
            if (edge[1] not in Y[node]) or (node not in Y[edge[0]]):
                edges_are_ok = False
    if edges_are_ok:
        if verbose:
            print("Confirmed!")
    else:
        print("Extraneous edges! There is a bug in the code!")


# The functions below are not used in the function transitive_closure();
# they are designed to test transitive_closure().


def is_transitive(E, N):
    """This function tests whether E is transitive."""
    X = {node: set() for node in N}
    Y = {node: set() for node in N}
    for edge in E:
        X[edge[1]].add(edge[0])
        Y[edge[0]].add(edge[1])
    for node in N:
        for x in X[node]:
            for y in Y[node]:
                if x not in X[y]:
                    return False
    return True


def test_transitive_closure_function(
         E_size=800000, N_range=1000000,verbose=True, update_period=3000,
         update_mode='by_time', check_accuracy=False,print_seed=False, seed=None
         ):
    """
    This function generates a random relation and tests 
    transitive_closure(), returning the transitive closure of the
    relation, the domain, and the original relation.
    """
    is_transitive_list = []
    E = set()
    if seed is None:
        np.random.seed()
        if print_seed:
            the_seed = np.random.randint(0, 2**32-1)
            np.random.seed(the_seed)
            print("The random seed is %s." % the_seed)
    else:
        np.random.seed(seed)
    for j in range(E_size):
        E.add((1 + np.random.randint(N_range, dtype=np.int32),
               1 + np.random.randint(N_range, dtype=np.int32)))
    original_E = E.copy()
    N = set()
    for edge in E:
        N.add(edge[0])
        N.add(edge[1])
    if verbose:
        print("Generated relation with %s edges on %s nodes..."
              % (len(E), len(N)))
    transitive_closure(E, N, verbose=verbose, check_edges=check_accuracy,
                       update_period=update_period,
                       update_mode=update_mode)
    if check_accuracy:
        if is_transitive(E, N):
            if verbose:
                print("It's transitive!")
        else:
            print("All is lost! It's not transitive!")
    return E, N, original_E

if __name__ == "__main__":
    import sys
    test_transitive_closure_function(E_size=10000, N_range=10000,
                                     verbose=True, update_period=3000,
                                     update_mode='by_time', check_accuracy=True,
                                     print_seed=False, seed=None)
    sys.exit(0)
