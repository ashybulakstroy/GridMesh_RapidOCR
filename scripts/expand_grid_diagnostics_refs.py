import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('file',default='candidate_graph_00.json',nargs='?');a=p.parse_args();objects=json.loads((a.folder/'objects.json').read_text(encoding='utf-8'))
def expand(x):
    if isinstance(x,dict) and set(x)=={'$ref'}:return expand(objects[x['$ref']])
    if isinstance(x,dict):return {k:expand(v) for k,v in x.items()}
    if isinstance(x,list):return [expand(v) for v in x]
    return x
print(json.dumps(expand(json.loads((a.folder/a.file).read_text(encoding='utf-8'))),ensure_ascii=False,indent=2))
