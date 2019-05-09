This is a silly project that implements an algorithm for finding the transitive closure of a relation, which can also be thought of as a directed graph, hence the use of the terms nodes and edges in the comments and documentation.

The easiest way to test the principal function, transitive_closure(), is to use the premade transitive_closure_function_test(). You can use this function either after importing transitive-closure to an interpreter, or by calling python transitive-closure.py from the command line. Arguments can be passed in the interpreter (see docstring), but none are needed and it has reasonable defaults. Do make sure to ask for the return values when you call it in an interpreter, or you could get a very long terminal output...

To convince yourself that we really are getting the transitive closure, you can use the short function is_transitive() and the parameter check_accuracy=True in test_transitive_closure_function. The parameter calls is_transitive() on the final output of transitive_closure(), and also causes verify_edges() to be invoked after every step of the primary loop, which double checks that the new edges are all things that must be in the transitive closure.

This module technically requires numpy, but it can easily be edited to work in base Python. The only use of numpy is for generating random 32-bit integers (Python's default int class being much larger in memory). Simply replace np.random with random in all cases, and omit the dtype=np.int32 argument.
