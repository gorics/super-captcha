from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--genome',required=True)
    ap.add_argument('--result',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    z=np.load(args.genome)
    result=json.loads(Path(args.result).read_text())
    W=z['W'].astype(np.float32)
    nz=np.argwhere(W!=0)
    edges=[[int(i),int(j),round(float(W[i,j]),6)] for i,j in nz]
    live={
      'schema':'evofly-live-v1',
      'generation':int(z['generation'][0]),
      'fitness':float(result['evolution']['champion_fitness']),
      'n':int(W.shape[0]),
      'edges':edges,
      'leak':[round(float(x),6) for x in z['leak']],
      'threshold':[round(float(x),6) for x in z['threshold']],
      'llm_gain':float(z['llm_gain'][0]),
      'sd_gain':float(z['sd_gain'][0]),
      'global_nodes':result['source']['global_nodes'],
      'source':{k:result['source'][k] for k in ('dataset','neurons','connections','synapses','sampled_real_edges','sampled_real_abs_synapses')},
      'metrics':result['evolution']['champion_metrics'],
      'fusion':result['fusion'],
      'disclaimer':'Vision is a live compound-eye/sensory reconstruction. Korean text is a decoder/LLM interpretation of model state, not evidence of subjective thought.'
    }
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(live,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'out':str(out),'generation':live['generation'],'edges':len(edges),'bytes':out.stat().st_size}))

if __name__=='__main__':
    main()
