from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

N_FULL=139255
NNZ_FULL=3732460
SYN_FULL=50666648

@dataclass
class Genome:
    W: np.ndarray
    leak: np.ndarray
    threshold: np.ndarray
    llm_gain: float=0.08
    sd_gain: float=0.05
    generation: int=0
    fitness: float=-1e9
    uid: str='seed'
    def clone(self):
        return Genome(self.W.copy(),self.leak.copy(),self.threshold.copy(),float(self.llm_gain),float(self.sd_gain),int(self.generation),float(self.fitness),str(self.uid))

def load_real_connectome(path:Path):
    raw=np.memmap(path,mode='r',dtype=np.uint8)
    n=N_FULL; pbytes=4*(n+1)
    indptr=np.frombuffer(raw[:pbytes],dtype='<i4').copy()
    nnz=int(indptr[-1])
    if nnz!=NNZ_FULL: raise RuntimeError(f'expected {NNZ_FULL} edges, got {nnz}')
    d0=pbytes; d1=d0+4*nnz
    delta=np.frombuffer(raw[d0:d1],dtype='<i4')
    weight=np.frombuffer(raw[d1:d1+2*nnz],dtype='<i2')
    syn=int(np.abs(weight.astype(np.int64)).sum())
    if syn!=SYN_FULL: raise RuntimeError(f'expected {SYN_FULL} synapses, got {syn}')
    return indptr,delta,weight,syn

def decode_row(delta,a,b):
    if b<=a:return np.empty(0,dtype=np.int32)
    return np.cumsum(delta[a:b],dtype=np.int64).astype(np.int32)

def seed_subgraph(indptr,delta,weight,n_nodes=96):
    deg=np.diff(indptr)
    k=min(96,len(deg))
    top=np.argpartition(deg,-k)[-k:]
    top=top[np.argsort(deg[top])[::-1]]
    nodes=[]; seen=set()
    for s0 in top:
        s=int(s0)
        if s not in seen: nodes.append(s);seen.add(s)
        a,b=int(indptr[s]),int(indptr[s+1])
        dst=decode_row(delta,a,b); ww=weight[a:b]
        if len(dst):
            for q in np.argsort(np.abs(ww))[::-1][:48]:
                d=int(dst[q])
                if d not in seen:
                    nodes.append(d);seen.add(d)
                    if len(nodes)>=n_nodes:break
        if len(nodes)>=n_nodes:break
    for s0 in top:
        if len(nodes)>=n_nodes:break
        s=int(s0)
        if s not in seen:nodes.append(s);seen.add(s)
    nodes=np.array(nodes[:n_nodes],dtype=np.int32)
    local={int(g):i for i,g in enumerate(nodes)}
    W=np.zeros((len(nodes),len(nodes)),dtype=np.float32)
    real_edges=0; real_syn=0
    for i,g0 in enumerate(nodes):
        g=int(g0);a,b=int(indptr[g]),int(indptr[g+1])
        dst=decode_row(delta,a,b);ww=weight[a:b]
        for d,w0 in zip(dst,ww):
            j=local.get(int(d))
            if j is not None:
                w=int(w0);W[j,i]=np.sign(w)*math.log1p(abs(w))/6.0
                real_edges+=1;real_syn+=abs(w)
    synthetic=0
    if np.count_nonzero(W)<len(nodes):
        rng=np.random.default_rng(783)
        for i in range(len(nodes)-1):
            if W[i+1,i]==0:
                W[i+1,i]=rng.normal(0,0.15);synthetic+=1
    g=Genome(W,np.full(len(nodes),.82,dtype=np.float32),np.zeros(len(nodes),dtype=np.float32),uid='v783-seed')
    meta={'global_nodes':nodes.tolist(),'sampled_real_edges':real_edges,'sampled_real_abs_synapses':real_syn,'synthetic_backbone_edges':synthetic}
    return g,meta

def rollout(g:Genome,x):
    s=np.zeros((x.shape[0],g.W.shape[0]),dtype=np.float32)
    for t in range(x.shape[1]):
        inp=np.zeros_like(s);inp[:,0]=x[:,t,0]*1.8-.9;inp[:,1]=x[:,t,1]*1.8-.9
        s=np.tanh(s*g.leak+s@g.W.T+inp-g.threshold)
    return s

def fitness(g:Genome,seed=0):
    rng=np.random.default_rng(seed);m=128
    bits=rng.integers(0,2,size=(m,6,2)).astype(np.float32)
    s=rollout(g,bits)
    yxor=(bits[:,-1,0]!=bits[:,-1,1]).astype(np.float32);ymem=bits[:,0,0]
    px=1/(1+np.exp(-3*s[:,-1]));pm=1/(1+np.exp(-3*s[:,-2]))
    mse=float(np.mean((px-yxor)**2)+np.mean((pm-ymem)**2))
    acc=float(np.mean((px>=.5)==yxor));macc=float(np.mean((pm>=.5)==ymem))
    complexity=float(np.count_nonzero(g.W)/g.W.size)
    # A small nonzero coupling prior prevents LLM/SD organs from being selected away before expensive foundation-model evaluation.
    fusion_prior=.02*(1-abs(g.llm_gain-.10))+.02*(1-abs(g.sd_gain-.08))
    score=2.0-mse+.4*acc+.4*macc-.02*complexity+fusion_prior
    return score,{'xor_accuracy':acc,'memory_accuracy':macc,'mse':mse,'edges':int(np.count_nonzero(g.W)),'fusion_prior':fusion_prior}

def mutate(parent:Genome,rng,uid):
    g=parent.clone();g.uid=uid;g.generation=parent.generation+1;n=g.W.shape[0]
    mask=rng.random(g.W.shape)<.006
    g.W[mask]+=rng.normal(0,.16,size=int(mask.sum())).astype(np.float32)
    ops=[]
    for _ in range(max(3,n//20)):
        op=str(rng.choice(['add','delete','rewire','duplicate'],p=[.34,.24,.27,.15]));ops.append(op)
        if op=='add':
            a,b=rng.integers(0,n,size=2)
            if a!=b:g.W[a,b]=rng.normal(0,.25)
        elif op=='delete':
            nz=np.argwhere(g.W!=0)
            if len(nz):
                a,b=nz[rng.integers(0,len(nz))];g.W[a,b]=0
        elif op=='rewire':
            nz=np.argwhere(g.W!=0)
            if len(nz):
                a,b=nz[rng.integers(0,len(nz))];v=g.W[a,b];g.W[a,b]=0;c=int(rng.integers(0,n))
                if c!=b:g.W[c,b]=v
        else:
            src=int(rng.integers(2,n-2));dst=int(rng.integers(2,n-2))
            g.W[dst,:]=g.W[src,:]+rng.normal(0,.03,size=n);g.W[:,dst]=g.W[:,src]+rng.normal(0,.03,size=n)
            g.leak[dst]=np.clip(g.leak[src]+rng.normal(0,.02),.2,.99);g.threshold[dst]=g.threshold[src]+rng.normal(0,.03)
    g.leak=np.clip(g.leak+rng.normal(0,.008,size=n),.2,.99).astype(np.float32)
    g.threshold=(g.threshold+rng.normal(0,.01,size=n)).astype(np.float32)
    g.llm_gain=float(np.clip(g.llm_gain+rng.normal(0,.012),.005,.35));g.sd_gain=float(np.clip(g.sd_gain+rng.normal(0,.012),.005,.35))
    return g,ops

def evolve(seed_g,generations,population,seed=783):
    rng=np.random.default_rng(seed);elite=seed_g.clone();history=[]
    base_generation=elite.generation
    for step in range(generations):
        gen=base_generation+step+1;pop=[(elite.clone(),['elite'])]
        for i in range(population-1):pop.append(mutate(elite,rng,f'g{gen:05d}-{i:03d}'))
        scored=[]
        for cand,ops in pop:
            sc,met=fitness(cand,seed=step%7);cand.fitness=sc;scored.append((sc,cand,met,ops))
        scored.sort(key=lambda z:z[0],reverse=True);elite=scored[0][1].clone();elite.generation=gen
        row={'generation':gen,'fitness':float(scored[0][0]),**scored[0][2],'uid':elite.uid,'llm_gain':elite.llm_gain,'sd_gain':elite.sd_gain,'winning_mutations':scored[0][3]}
        history.append(row);print('EVOLVE',json.dumps(row),flush=True)
    return elite,history

def bridge_vector(g:Genome,dim=16):
    rng=np.random.default_rng(20260929);bits=rng.integers(0,2,size=(1,8,2)).astype(np.float32);v=rollout(g,bits)[0][-dim:].astype(np.float32);return v/(np.linalg.norm(v)+1e-6)

def project(v,out_dim,seed):
    rng=np.random.default_rng(seed);P=rng.normal(0,1/math.sqrt(len(v)),size=(len(v),out_dim)).astype(np.float32);return v@P

def load_llm(model_id):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval()
    return tok,model

def llm_forward(g,tok,model,outdir):
    import torch
    prompt="An evolved fruit-fly connectome and this language organ are one computational organism. Describe its current role in one short sentence:"
    ids=tok(prompt,return_tensors='pt').input_ids;emb_layer=model.get_input_embeddings();v=bridge_vector(g)
    with torch.no_grad():
        emb=emb_layer(ids);bias=torch.tensor(project(v,emb.shape[-1],991),dtype=emb.dtype).view(1,1,-1)*g.llm_gain
        seq=emb+bias;generated=[];hidden_feedback=0.0
        for _ in range(12):
            out=model(inputs_embeds=seq,output_hidden_states=True,use_cache=False);nxt=torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True);generated.append(int(nxt.item()))
            hidden_feedback=.8*hidden_feedback+.2*float(torch.tanh(out.hidden_states[-1][:,-1,:].mean()).item());seq=torch.cat([seq,emb_layer(nxt)+bias],dim=1)
    text=tok.decode(generated,skip_special_tokens=True).strip();res={'model':model.config._name_or_path,'text':text,'feedback_to_brain':hidden_feedback,'brain_gain':g.llm_gain,'embedding_dim':int(seq.shape[-1])}
    (outdir/'llm_result.json').write_text(json.dumps(res,indent=2));return res

def load_sd(model_id):
    import torch
    from diffusers import StableDiffusionPipeline
    pipe=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False);pipe=pipe.to('cpu');return pipe

def sd_forward(g,pipe,outdir):
    import torch
    prompt='scientific visualization of an evolving drosophila neural organism, synapses and neural topology, laboratory visualization'
    with torch.no_grad():
        toks=pipe.tokenizer(prompt,padding='max_length',max_length=pipe.tokenizer.model_max_length,truncation=True,return_tensors='pt');enc=pipe.text_encoder(toks.input_ids)[0];v=bridge_vector(g)
        bias=torch.tensor(project(v,enc.shape[-1],1773),dtype=enc.dtype).view(1,1,-1)*g.sd_gain;pe=enc+bias
        latent=torch.randn((1,pipe.unet.config.in_channels,16,16),generator=torch.Generator().manual_seed(783));pred=pipe.unet(latent,torch.tensor([500],dtype=torch.long),encoder_hidden_states=pe).sample
        feedback=float(torch.tanh(pred.mean()).item())
        img=pipe(prompt_embeds=pe,guidance_scale=1.0,num_inference_steps=2,height=128,width=128,generator=torch.Generator().manual_seed(783)).images[0]
    p=outdir/'champion.png';img.save(p);res={'model':getattr(pipe,'name_or_path','segmind/tiny-sd'),'image':str(p),'feedback_to_brain':feedback,'brain_gain':g.sd_gain,'prompt_embedding_dim':int(pe.shape[-1])};(outdir/'sd_result.json').write_text(json.dumps(res,indent=2));return res

def foundation_coupling_refinement(g,tok,llm,pipe,rounds=2):
    # Expensive but real: mutate one structural edge + coupling gains, run both foundation organs, select stable bidirectional activation.
    rng=np.random.default_rng(4242);trace=[];elite=g.clone()
    def eval_one(c):
        import torch
        base,_=fitness(c,123);v=bridge_vector(c)
        with torch.no_grad():
            ids=tok('fly organism cognition',return_tensors='pt').input_ids;emb=llm.get_input_embeddings()(ids);bias=torch.tensor(project(v,emb.shape[-1],991),dtype=emb.dtype).view(1,1,-1)*c.llm_gain;o=llm(inputs_embeds=emb+bias,output_hidden_states=True,use_cache=False);lf=float(torch.tanh(o.hidden_states[-1][:,-1,:].mean()).item())
            toks=pipe.tokenizer('neural organism',padding='max_length',max_length=pipe.tokenizer.model_max_length,truncation=True,return_tensors='pt');enc=pipe.text_encoder(toks.input_ids)[0];sbias=torch.tensor(project(v,enc.shape[-1],1773),dtype=enc.dtype).view(1,1,-1)*c.sd_gain;pe=enc+sbias;latent=torch.randn((1,pipe.unet.config.in_channels,16,16),generator=torch.Generator().manual_seed(99));sf=float(torch.tanh(pipe.unet(latent,torch.tensor([500]),encoder_hidden_states=pe).sample.mean()).item())
        # Reward non-saturated two-way activity while preserving cognitive fitness.
        coupling=(1-abs(abs(lf)-.08))+(1-abs(abs(sf)-.08));return base+.03*coupling,lf,sf
    for r in range(rounds):
        cands=[elite.clone()]
        for i in range(2):
            c,_=mutate(elite,rng,f'fusion-r{r}-{i}');cands.append(c)
        scored=[]
        for c in cands:
            sc,lf,sf=eval_one(c);scored.append((sc,c,lf,sf))
        scored.sort(key=lambda z:z[0],reverse=True);elite=scored[0][1].clone();trace.append({'round':r+1,'score':float(scored[0][0]),'llm_feedback':scored[0][2],'sd_feedback':scored[0][3],'uid':elite.uid,'llm_gain':elite.llm_gain,'sd_gain':elite.sd_gain})
        print('FUSION_EVOLVE',json.dumps(trace[-1]),flush=True)
    return elite,trace

def save_genome(g,path):
    np.savez_compressed(path,W=g.W,leak=g.leak,threshold=g.threshold,llm_gain=np.array([g.llm_gain],dtype=np.float32),sd_gain=np.array([g.sd_gain],dtype=np.float32),generation=np.array([g.generation]),fitness=np.array([g.fitness]))

def load_genome(path):
    z=np.load(path);return Genome(z['W'].astype(np.float32),z['leak'].astype(np.float32),z['threshold'].astype(np.float32),float(z['llm_gain'][0]),float(z['sd_gain'][0]),int(z['generation'][0]),float(z['fitness'][0]),'resumed')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--connectome',required=True);ap.add_argument('--out',default='evofly/results');ap.add_argument('--generations',type=int,default=8);ap.add_argument('--population',type=int,default=8);ap.add_argument('--nodes',type=int,default=96);ap.add_argument('--resume',default='');ap.add_argument('--llm',default='HuggingFaceTB/SmolLM2-135M-Instruct');ap.add_argument('--sd',default='segmind/tiny-sd');ap.add_argument('--skip-foundation',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);t0=time.time();indptr,delta,weight,syn=load_real_connectome(Path(args.connectome));real_seed,source=seed_subgraph(indptr,delta,weight,args.nodes);seed_g=load_genome(Path(args.resume)) if args.resume and Path(args.resume).exists() else real_seed
    seed_score,seed_metrics=fitness(seed_g,0);champ,hist=evolve(seed_g,args.generations,args.population);fusion_trace=[];llm_res=sd_res=None
    if not args.skip_foundation:
        tok,llm=load_llm(args.llm);pipe=load_sd(args.sd);champ,fusion_trace=foundation_coupling_refinement(champ,tok,llm,pipe,rounds=2);llm_res=llm_forward(champ,tok,llm,out);sd_res=sd_forward(champ,pipe,out)
    final_score,final_metrics=fitness(champ,0);save_genome(champ,out/'champion.npz')
    result={'source':{'dataset':'FlyWire FAFB v783','neurons':N_FULL,'connections':NNZ_FULL,'synapses':syn,**source},'evolution':{'generations_this_run':args.generations,'population':args.population,'seed_generation':seed_g.generation,'champion_generation':champ.generation,'seed_fitness':seed_score,'champion_fitness':final_score,'fitness_delta':final_score-seed_score,'seed_metrics':seed_metrics,'champion_metrics':final_metrics,'history':hist},'fusion':{'llm':llm_res,'stable_diffusion':sd_res,'fusion_evolution':fusion_trace,'architecture':'One phenotype: evolved recurrent FlyWire-derived brain state directly perturbs LLM token embeddings and Stable Diffusion text embeddings; hidden/UNet activations are evaluated as feedback during coupling selection.'},'runtime_seconds':time.time()-t0}
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT_JSON',json.dumps(result),flush=True)
if __name__=='__main__':main()
