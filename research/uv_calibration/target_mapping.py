"""Only change target image identity; retain original selector and loader semantics."""
def remap(rows,target):
 assert target in ('plh','plv')
 for row in rows:
  if row.get('split') not in ('train','val'):continue
  assert row.get(target+'_path') and row.get(target+'_sha256')
  z=dict(row);z['uv_path']=row[target+'_path'];z['uv_sha256']=row[target+'_sha256'];yield z

def same_training_config(old,new):
 import copy
 a=copy.deepcopy(old);b=copy.deepcopy(new)
 for d in (a,b):
  d.pop('output_dir');d.pop('experiment_id');d.pop('target_modality',None)
  d['protocol'].pop('task')
 assert a==b,'An original UV training condition changed'
 return True
