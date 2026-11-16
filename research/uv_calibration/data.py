import io,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def image(row,mod,root):
    p=Path(root)/row[mod+'_path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==row[mod+'_sha256'],'Training image changed: '+str(p)
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size==(int(row['source_width']),int(row['source_height']))
        im=im.convert('RGB');w,h=im.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2;im=im.crop((left,top,left+s,top+s)).resize((1024,1024),Image.LANCZOS)
        return np.ascontiguousarray((np.asarray(im,dtype=np.float32)/255.).transpose(2,0,1))


class Images(Dataset):
    def __init__(self,manifest,indices,target,root):
        d=json.loads(Path(manifest).read_text());assert d['split'] in ('train','val');self.rows=d['rows'];self.indices=indices;self.target=target;self.root=root
    def __len__(self):return len(self.indices)
    def __getitem__(self,pos):
        i=self.indices[pos]
        if i<0:return np.zeros((3,1024,1024),np.float32),np.zeros((3,1024,1024),np.float32),-1,'zero_weight_nonexample'
        r=self.rows[i];return image(r,'rgb',self.root),image(r,self.target,self.root),i,r['sample_id']


def rank_indices(order,world,rank,global_batch=12):
    result=[]
    for start in range(0,len(order),global_batch):
        b=order[start:start+global_batch]
        for k in range((len(b)+world-1)//world):
            j=k*world+rank;result.append(b[j] if j<len(b) else -1)
    return result
