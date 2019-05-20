import gc
import pyspark
import os 
import time
import shutil
import generate_digraph
import transitive_closure

# This function is slow at the moment, but it works.
# It implements the "Semi-Naive" algorithm from the paper,
# so is essentially just repeated joins and set differences,
# which is why I represented edges with DataFrames. DataFrames
# do seem to be intrinsically slow, but I don't know if any
# other types are faster or not as of yet. It is not *quite*
# as slow with alrge batches and large datasets.

def transitive_closure_from_dataframe(digraph):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_file = os.path.join(dir_path,'solution.pickle')
    try:
        shutil.rmtree(save_file)
    except:
        pass
    sc = pyspark.context.SparkContext.getOrCreate()
    sc.setCheckpointDir("~/.transitive_closure")
    spark = pyspark.sql.SparkSession(sc)
    # Translation to Spark format is ludicrously slow with PyArrow
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    start_time=time.time()
    orig_digraph = spark.createDataFrame(digraph.copy())
    new_edges = spark.createDataFrame(digraph.copy()).checkpoint()
    new_edges_mem = digraph
    closed_digraph = spark.createDataFrame(digraph).checkpoint()
    while not new_edges_mem.empty:
        new_edges = spark.createDataFrame(new_edges_mem).persist(
            pyspark.StorageLevel(False, True, False, False, 1)).checkpoint()
        new_edges = new_edges.join(orig_digraph,
                                   (new_edges.terminus == orig_digraph.origin))\
                             .select(new_edges.origin, orig_digraph.terminus)\
                             .union(orig_digraph.join(new_edges,
                                 (new_edges.origin == orig_digraph.terminus))\
                                 .select(orig_digraph.origin, new_edges.terminus))\
                             .distinct()\
                             .exceptAll(closed_digraph)
        # I don't see any copy method, and PyArrow is nearly
        # instantaneous,  so I'm going to use pandas to copy the new
        # edges to memory.
        new_edges_mem = new_edges.toPandas().copy()
        closed_digraph = closed_digraph.union(new_edges.persist(
            pyspark.StorageLevel(True, False, False, True, 1)
            ).checkpoint())
        # Putting this in just because of how much trouble I had with the
        # generate_digraph function
        gc.collect()
    # Ideally, we would be able to use the fact that closed_digraph is
    # already stored on the disk, but I'm not quite sure how to do it.
    # Also, for clusters with separate hard drives,
    # we would want to mount it somewhere with HDFS or similar.
    closed_digraph.rdd.saveAsPickleFile(save_file)
    df = closed_digraph.toPandas()
    sc.stop()
    return df

if __name__ == "__main__":
    import sys
    df = generate_digraph.generate_digraph(40,40,40,50)
    df2 = transitive_closure_from_dataframe(df)
    E, N = generate_digraph.convert_dataframe_to_set_of_pairs(df)
    E2, N2 = generate_digraph.convert_dataframe_to_set_of_pairs(df2)
    transitive_closure.transitive_closure(E,N)
    if E == E2:
        print("We agree with the old algorithm.")
    else:
        print("Got the wrong answer...")
    sys.exit(0)
