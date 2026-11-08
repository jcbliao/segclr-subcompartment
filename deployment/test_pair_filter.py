import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0,str(Path(__file__).parent))
if not Path('/orcd/home/002/jcbliao/rotation/segclr/quantify_segclr_pairs/scripts').exists():
    pytest.skip('Historical audit caches and companion source are cluster-only',allow_module_level=True)
import pair_filter as f


def brute(graph,projection,p,labels,edges):
    distance=dijkstra(graph,directed=False)
    out=np.zeros((6,2,len(edges)-1))
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            d=distance[projection[i],projection[j]]
            if not np.isfinite(d) or d>150000:continue
            k=f.bins(np.array([np.float32(d)]),edges)[0]
            g=p[i]*p[j];b=p[i]*(1-p[j])+p[j]*(1-p[i])
            for fold in range(6):
                if fold==0 or labels[i,fold-1]==labels[j,fold-1]:out[fold,:,k]+=[g,b]
    return out


def test_filter_exact_enumeration_and_collapsed_endpoints():
    # A-B-C path: filtering B must never sever the eligible A-C path.
    # Vertex 4 is disconnected; zero-distance *distinct* L2 endpoints count.
    a=np.array([0,1,1,2,2,3]);b=np.array([1,0,2,1,3,2])
    graph=coo_matrix((np.full(6,50000.),(a,b)),shape=(5,5)).tocsr()
    edges=np.linspace(0,150000,21)
    projection=np.array([0,0,1,2,2,3,4])
    p=np.array([1,.25,0,.4,.8,0,1.])
    rng=np.random.default_rng(17);labels=rng.integers(0,6,(7,5))
    labels[:,0]=0
    native_p=np.array([.3,0,.9,.5,1.]);native_labels=rng.integers(0,6,(5,5));native_labels[:,0]=0
    actual=f.skeleton_histograms(graph,[(projection,p,labels),(np.arange(5),native_p,native_labels)],edges)
    np.testing.assert_allclose(actual[0],brute(graph,projection,p,labels,edges),atol=1e-12)
    np.testing.assert_allclose(actual[1],brute(graph,np.arange(5),native_p,native_labels,edges),atol=1e-12)
    np.testing.assert_allclose(actual[:,0],actual[:,1],atol=1e-12)
    coords=np.column_stack([np.arange(5)*50000,np.zeros((5,2))])
    l2=f.l2_histograms(graph,coords,native_p,native_labels,edges)
    np.testing.assert_allclose(l2,actual[1],atol=1e-12)


def test_no_anchors():
    graph=coo_matrix((3,3)).tocsr();p=np.zeros(3);labels=np.zeros((3,5),int)
    result=f.skeleton_histograms(graph,[(np.arange(3),p,labels)],np.zeros(21))
    assert not result.any()


def test_compact_uint8_labels_do_not_overflow_vertex_indices():
    p=np.array([.25,.75]);projection=np.array([0,299]);labels=np.full((2,5),5,np.uint8)
    pp,qq,sg,sb=f.pooled_population(300,projection,p,labels)
    assert pp[6,299]==.75 and pp[6,0]==.25
    assert pp[30,299]==.75
