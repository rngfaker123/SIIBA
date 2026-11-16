"""Strict execution-only AST amendment of original U-Net++ recipe; loss/model functions exact."""
import ast,hashlib
OLD_BATCH='global16_rank_stride_4_4_4_4_physical1_tail2_1_1_0_0_one_gradient_mean_v1'
NEW_BATCH='global16_rank_stride_6_5_5_physical1_tail2_1_1_0_one_gradient_mean_execution3_v1'
def transform(raw):
 tree=ast.parse(raw);changes=[]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WORLD' for t in n.targets):
   assert isinstance(n.value,ast.Constant) and n.value.value==4;n.value.value=3;changes.append('WORLD4_to3')
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BATCHING' for t in n.targets):
   assert n.value.value==OLD_BATCH;n.value.value=NEW_BATCH;changes.append('batching_execution_metadata')
  if isinstance(n,ast.FunctionDef) and n.name=='check_preflight':
   count=0
   for x in ast.walk(n):
    if isinstance(x,ast.Compare) and isinstance(x.left,ast.Call) and isinstance(x.left.func,ast.Attribute) and x.left.func.attr=='get' and len(x.left.args)==1 and isinstance(x.left.args[0],ast.Constant) and x.left.args[0].value=='rank_full_images':
     assert len(x.comparators)==1 and isinstance(x.comparators[0],ast.Constant) and x.comparators[0].value==4
     x.comparators[0]=ast.parse('len(local_indices(range(16), rank))',mode='eval').body;count+=1
   assert count==1;changes.append('native_rank_count_gate_6_5_5')
  if isinstance(n,ast.FunctionDef) and n.name=='prepare':
   mappings={'sum_local_4_times_image_NLL_div_actual_global_n_then_allreduce_mean4_idle_tail_ranks_zero_grad':'sum_local_3_times_image_NLL_div_actual_global_n_then_allreduce_mean3_idle_tail_rank_zero_grad','original_UNetplusplus_FP32_physical1_same_science_clean_data_four_rank_manual_mean_not_bitwise_three_rank':'original_UNetplusplus_FP32_physical1_same_global16_three_rank_manual_mean_not_bitwise_four_rank','lowest_full_val_mean_UV_L1':'lowest_full_val_mean_declared_PL_target_L1'}
   found=[]
   for x in ast.walk(n):
    if isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value in mappings:found.append(x.value);x.value=mappings[x.value]
   assert set(found)==set(mappings);changes.append('truthful_execution_target_protocol_metadata')
  if isinstance(n,ast.FunctionDef) and n.name=='setup_distributed':
   for x in ast.walk(n):
    if isinstance(x,ast.Constant) and x.value=='Must run via torchrun --nproc_per_node=4 on the authorized node':x.value='Must run via torchrun --nproc_per_node=3 on the authorized node';changes.append('rank_count_error_message')
 assert set(changes)=={'WORLD4_to3','batching_execution_metadata','native_rank_count_gate_6_5_5','truthful_execution_target_protocol_metadata','rank_count_error_message'}
 original=ast.parse(raw);allowed={'prepare','check_preflight','setup_distributed'}
 before={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef)}
 after={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
 assert set(before)==set(after)
 assert all(before[k]==after[k] for k in before if k not in allowed),'A model/loss/optimizer/update/order/checkpoint/validation function changed'
 ast.fix_missing_locations(tree)
 report=dict(changes=changes,all_other_function_AST_exact=True,unchanged_functions=sorted(k for k in before if k not in allowed),compiled_AST_SHA256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),world=3,global_batch=16,microbatch=1,full_rank_counts=[6,5,5],tail_rank_counts=[1,1,0],not_bit_identical_to_old_four_rank_training=True)
 return tree,report
