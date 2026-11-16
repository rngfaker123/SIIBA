import unittest
import numpy as np
from scipy import ndimage
from frozen_f_partition import partition_frozen_f, REGION_NAMES, PartitionQCError


def seeds(f,positions):
    ww={}
    for k,(y,x) in zip(REGION_NAMES,positions):
        a=np.zeros_like(f);a[y,x]=f[y,x];ww[k]=a
    return ww


class FrozenFaceTest(unittest.TestCase):
    def test_zero_outside_and_no_weight_loss(self):
        f=np.zeros((18,22),np.float32)
        f[3:15,3:19]=1;f[8,10]=0;f[0,2:15]=.041;f[0,17]=.01
        w=seeds(f,[(4,4),(8,5),(8,15),(8,11),(13,10)])
        L,a=partition_frozen_f(f,w)
        self.assertEqual(a["satellite_count"],2)
        np.testing.assert_array_equal(sum(np.where(L==i,f,0) for i in range(1,6)),f)
        groups,count=ndimage.label(f>0)
        core=np.zeros_like(f,bool);core[3:15,3:19]=True;core &= f>0
        for gid in np.unique(groups[np.logical_and(f>0,~core)]):
            island=np.argwhere(groups==gid)
            def distance(i):
                dst=np.argwhere((L==i)&core)
                return ((island[:,None]-dst[None,:])**2).sum(axis=2).min()
            nearest=min(range(1,6),key=distance)
            self.assertEqual(np.unique(L[groups==gid]).tolist(),[nearest])
        for i,name in enumerate(REGION_NAMES,1):
            self.assertTrue(np.all(L[w[name]>0]==i))

    def test_equidistant_island_uses_priority(self):
        f=np.zeros((10,13),np.float32);f[1,1]=1;f[1,11]=1;f[1,6]=.25
        f[8,5:8]=.75
        w=seeds(f,[(1,11),(1,1),(8,5),(8,6),(8,7)])
        L,a=partition_frozen_f(f,w)
        self.assertEqual(L[1,6],1)
        self.assertEqual(a["islands"][0]["distance_tied_labels"],[1,2])

    def test_empty_and_cross_component_seeds_report(self):
        f=np.ones((7,11),np.float32);f[:,5]=0
        w=seeds(f,[(0,1),(2,2),(2,8),(3,4),(5,4)])
        w["forehead"][0,8]=1
        with self.assertRaisesRegex(PartitionQCError,"seed_crosses_disconnected_F"):
            partition_frozen_f(f,w)
        w["forehead"][:]=0
        with self.assertRaisesRegex(PartitionQCError,"empty_seed"):
            partition_frozen_f(f,w)

if __name__=="__main__": unittest.main()
