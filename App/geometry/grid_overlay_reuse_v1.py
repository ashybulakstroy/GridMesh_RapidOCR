"""Per-overlay cache, never shared between graphs, jobs or threads."""
def polygon_reader(result, nodes, polygon):
    cache={}
    def read(row,col):
        key=(row,col)
        if key not in cache:
            cache[key]=polygon(result,row,col,nodes=nodes)
        # The only caller reads slices and rounds them, never mutates the cache.
        return cache[key]
    return read
