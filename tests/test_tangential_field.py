import numpy as np
import pytest
from hyperplane_arrangements.tangential_field import ConvexPolygonFlow, Vortex

def test_convex_polygon_flow_init():
    vertices = [(0, 0), (1, 0), (0, 1)]
    flow = ConvexPolygonFlow(vertices, n_vortices=2, seed=42)
    assert len(flow.vortices) == 2
    assert flow.bounds is not None
    assert flow.centroid is not None
    assert len(flow.edges) == 3

def test_convex_polygon_flow_invalid_vertices():
    # Only 2 vertices
    with pytest.raises(ValueError, match="need at least three vertices"):
        ConvexPolygonFlow([(0, 0), (1, 1)])
    
    # Collinear vertices
    with pytest.raises(ValueError, match="polygon vertices must be non-collinear"):
        ConvexPolygonFlow([(0, 0), (1, 1), (2, 2)])

def test_convex_polygon_flow_velocity():
    vertices = [(0, 0), (10, 0), (10, 10), (0, 10)]
    flow = ConvexPolygonFlow(vertices, seed=42)
    vel = flow.velocity((5, 5))
    assert isinstance(vel, np.ndarray)
    assert vel.shape == (2,)

    # test out of bounds point
    with pytest.raises(ValueError, match="point lies outside the polygon"):
        flow.velocity((20, 20))

def test_convex_polygon_flow_sampling():
    vertices = [(0, 0), (1, 0), (0, 1)]
    flow = ConvexPolygonFlow(vertices, seed=42)
    pts = flow.sample_random_points(10)
    assert pts.shape == (10, 2)
    for pt in pts:
        assert flow.path.contains_point(pt)
        
    vel_dict = flow.sample_velocity(5, noise=0.1)
    assert len(vel_dict) == 5
    for k, v in vel_dict.items():
        assert isinstance(k, tuple)
        assert len(k) == 2
        assert isinstance(v, np.ndarray)

def test_vorticity_divergence():
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    flow = ConvexPolygonFlow(vertices, seed=42)
    vor = flow.vorticity((0.5, 0.5))
    div = flow.divergence((0.5, 0.5))
    assert isinstance(vor, float)
    assert isinstance(div, float)
