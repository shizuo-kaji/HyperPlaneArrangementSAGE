import pytest
from hyperplane_arrangements.library import braid, type_B, type_D, shi, catalan, linial, ish

def test_presets():
    A = braid(3)
    assert A.num_planes == 3
    assert A.n == 3
    
    B = type_B(3)
    assert B.num_planes == 9
    assert B.n == 3
    
    D = type_D(3)
    assert D.num_planes == 6
    
    S = shi(3)
    assert S.num_planes == 7
    assert S.n == 4
    
    C = catalan(3)
    assert C.num_planes == 10
    
    L = linial(3)
    assert L.num_planes == 4
    
    I = ish(3)
    assert I.num_planes == 7
    assert I.n == 4


from hyperplane_arrangements.library import exceptional_coxeter

def test_exceptional_coxeter():
    F4 = exceptional_coxeter('F4')
    assert F4.num_planes == 24
    assert F4.n == 4
    
    G2 = exceptional_coxeter('G2')
    assert G2.num_planes == 6
    assert G2.n == 3
    
    H3 = exceptional_coxeter('H3')
    assert H3.num_planes == 15
    assert H3.n == 3
    
    H4 = exceptional_coxeter('H4')
    assert H4.num_planes == 60
    assert H4.n == 4

