"""Importable thin target adapter; the original trainer's source is not rewritten."""
import os,sys,importlib.util,json,hashlib
from pathlib import Path
_N=Path('/study/siiba/experiments/pl_controls_original_uv_pipeline_20261031_v1')
_target=os.environ['SIIBA_PL_TARGET'];assert _target in ('plh','plv')
from chain_common import checked
_q=checked(json.loads((_N/'request.json').read_text()));_p=Path(_q['original_uv']['trainer']['path'])
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert _sha(_p)==_q['original_uv']['trainer']['sha256']
for _d in _q['sources'].values():assert _sha(_d['path'])==_d['sha256']
__file__=str(_p)
exec(compile(_p.read_bytes(),str(_p),'exec'),globals())
from target_mapping import remap,same_training_config
_base_select=select_training_rows
def select_training_rows(source_rows,expected_counts=None):
 return _base_select(remap(source_rows,_target),expected_counts)
_base_load=load_frozen
def load_frozen(config_path):
 out=_base_load(config_path);cfg=out[0]
 assert cfg['target_modality']==_target
 same_training_config(read_json(_q['original_uv']['config']['path']),cfg)
 report=read_json(_N/('target_mapping_'+_target+'.json'))
 assert report['passed'] and report['TRAIN']==76642 and report['VAL']==10948
 assert report['protocol_sha256']==digest(out[1]/'protocol.json')
 return out
