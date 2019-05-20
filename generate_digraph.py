import gc
import numpy as np
import pandas as pd
import pyspark
import time

# This function is currently ludicrously slow.

def generate_digraph(edge_count=40, batch_size=50,
                     steps_to_python_gc=50, domain_size=50):
    sc = pyspark.context.SparkContext.getOrCreate()
    sc.setCheckpointDir("~/.transitive_closure")
    spark = pyspark.sql.SparkSession(sc)
    # Translation to Spark format is ludicrously slow without PyArrow
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    batches_to_python_gc = steps_to_python_gc // batch_size
    batch_count = int(np.ceil(edge_count/batch_size))
    digraph=[]
    for i in range(batch_count):
        if i%batches_to_python_gc == 0:
            gc.collect()
        if i == batch_count-1 and edge_count % batch_size > 0:
            batch_size = edge_count % batch_size
        new_origins = np.random.randint(0,domain_size,dtype=np.int32,
                                        size=(batch_size,1))
        new_termini = np.random.randint(0,domain_size,dtype=np.int32,
                                        size=(batch_size,1))
        partial_digraph = pd.DataFrame(np.concatenate(
            [new_origins, new_termini], 1), columns=("origin", "terminus")
            )
        partial_digraph = spark.createDataFrame(partial_digraph).distinct()
        if digraph != []:
            digraph[0] = digraph[1].union(digraph[0])\
                                   .orderBy(["origin", "terminus"],
                                            ascending=[True, True])\
                                   .distinct()\
                                   .persist(pyspark.StorageLevel(True, False,
                                                                 False, True,
                                                                 1))\
                                   .checkpoint()
            digraph[1] = partial_digraph
        else:
            digraph = [partial_digraph,partial_digraph]
    if i == batch_count-1:
        gc.collect()
        digraph = digraph[1].union(digraph[0])\
                            .orderBy(["origin", "terminus"],
                                     ascending=[True, True])\
                            .distinct()\
                            .persist(pyspark.StorageLevel(True, False,
                                                          False, True,
                                                          1))\
                            .checkpoint()
    df=digraph.toPandas()
    sc.stop()
    return df

def convert_dataframe_to_set_of_pairs(df):
    sc = pyspark.context.SparkContext.getOrCreate()
    spark = pyspark.sql.SparkSession(sc)
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    try:
        df=df.toPandas()
    except:
        pass
    E=set()
    N=set()
    for i in range(df.shape[0]):
        E.add((df["origin"][i], df["terminus"][i]))
        N.add(df["origin"][i])
        N.add(df["terminus"][i])
    sc.stop()
    return E, N
