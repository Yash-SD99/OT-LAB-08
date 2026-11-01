"""Unit and Verification Tests for Track A: Continuous Optimization."""

import unittest
import numpy as np
from tracks.A_continuous.landscapes import (
    StronglyConvexQuadratic,
    SaddlePointLandscape,
    RosenbrockLandscape,
    RastriginLandscape,
    NoisyObjectiveWrapper
)
from tracks.A_continuous.optimizers import (
    GradientDescent,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    NewtonRaphson,
    GaussNewton,
    empirical_fisher,
    diagonal_curvature,
    low_rank_curvature,
    hessian_vector_product_fd
)
from tracks.A_continuous.models import SimpleMLP


def test_strongly_convex_quadratic():
    """Verifies that 1st order optimizers converge on strongly convex quadratic."""
    A = np.array([[4.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    b = np.array([1.0, 2.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)
    
    # Check known analytic optimum x* = A^{-1} b
    x_expected = np.linalg.solve(A, b)
    np.testing.assert_allclose(obj.x_opt, x_expected, rtol=1e-5)
    
    # Test Gradient Descent
    x = np.array([5.0, 5.0], dtype=np.float64)
    opt = GradientDescent(lr=0.1)
    for _ in range(100):
        g = obj.gradient(x)
        x = opt.step(x, g)
    np.testing.assert_allclose(x, obj.x_opt, atol=1e-3)
    
    # Test Adam
    x = np.array([5.0, 5.0], dtype=np.float64)
    opt = Adam(lr=0.2)
    for _ in range(150):
        g = obj.gradient(x)
        x = opt.step(x, g)
    np.testing.assert_allclose(x, obj.x_opt, atol=1e-2)


def test_newton_quadratic_convergence():
    """Newton's method must converge to the exact optimum in 1 step on a pure quadratic."""
    A = np.array([[5.0, 2.0], [2.0, 3.0]], dtype=np.float64)
    b = np.array([4.0, -1.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)
    
    x = np.array([100.0, -50.0], dtype=np.float64)
    opt = NewtonRaphson(lr=1.0, damping=0.0)
    
    g = obj.gradient(x)
    H = obj.hessian(x)
    x_next = opt.step(x, g, H)
    np.testing.assert_allclose(x_next, obj.x_opt, atol=1e-7)


def test_gauss_newton_rosenbrock():
    """Gauss-Newton with damping minimizes non-linear least squares Rosenbrock."""
    obj = RosenbrockLandscape()
    x = np.array([-1.2, 1.0], dtype=np.float64)
    opt = GaussNewton(lr=0.8, damping=1e-3)
    
    for _ in range(50):
        r = obj.residuals(x)
        J = obj.jacobian(x)
        x = opt.step(x, r, J)
        
    np.testing.assert_allclose(x, np.array([1.0, 1.0]), atol=1e-2)


def test_backpropagation_gradient_check():
    """Verifies that analytical backpropagation matches finite differences (< 1e-5 relative error)."""
    np.random.seed(42)
    mlp = SimpleMLP(in_dim=3, hidden_dim=4, out_dim=2, reg_lambda=0.01)
    X = np.random.randn(5, 3)
    y = np.random.randn(5, 2)
    
    rel_error = mlp.numerical_gradient_check(X, y, eps=1e-6)
    assert rel_error < 1e-5, f"Gradient check failed with relative error: {rel_error}"


def test_hessian_vector_product_accuracy():
    """Hessian-vector product via finite differences matches exact H @ v."""
    A = np.array([[3.0, 1.5], [1.5, 6.0]], dtype=np.float64)
    b = np.array([2.0, -3.0], dtype=np.float64)
    obj = StronglyConvexQuadratic(A, b)
    
    x = np.array([1.0, 1.0])
    v = np.array([0.6, -0.8])
    
    # Exact product
    exact_Hv = obj.hessian(x) @ v
    
    # Finite difference HVP
    fd_Hv = hessian_vector_product_fd(obj.gradient, x, v, eps=1e-5)
    
    np.testing.assert_allclose(fd_Hv, exact_Hv, rtol=1e-4, atol=1e-4)


def test_curvature_approximations():
    """Tests Empirical Fisher, Diagonal, and Low-rank decomposition."""
    np.random.seed(42)
    grads = np.random.randn(20, 5) # 20 samples, 5 parameters
    F = empirical_fisher(grads)
    
    assert F.shape == (5, 5)
    # Fisher must be positive semi-definite
    eigvals = np.linalg.eigvalsh(F)
    assert np.all(eigvals >= -1e-10)
    
    diag_F = diagonal_curvature(F)
    assert diag_F.shape == (5,)
    np.testing.assert_allclose(diag_F, np.diag(F))
    
    # Low-rank approximation of rank 2
    F_lowrank = low_rank_curvature(F, rank=2)
    assert F_lowrank.shape == (5, 5)
    # Check that rank is <= 2
    rank = np.linalg.matrix_rank(F_lowrank, tol=1e-7)
    assert rank <= 2


def test_saddle_point_exact_vs_perturbed():
    """Verifies that exact saddle point is an equilibrium for GD, but perturbed escapes."""
    saddle = SaddlePointLandscape()
    
    # Exact saddle start [0, 0]
    x_exact = np.array([0.0, 0.0])
    for _ in range(20):
        x_exact = x_exact - 0.1 * saddle.gradient(x_exact)
    np.testing.assert_allclose(x_exact, np.array([0.0, 0.0]))
    
    # Perturbed start [1e-5, 1e-5]
    x_pert = np.array([1e-5, 1e-5])
    for _ in range(20):
        x_pert = x_pert - 0.1 * saddle.gradient(x_pert)
    # y must diverge negatively along negative curvature
    assert abs(x_pert[1]) > 1e-4
    assert abs(x_pert[1]) > abs(x_pert[0])


def test_rastrigin_multi_start_local_minima():
    """Verifies that Rastrigin contains multiple local minima across random restarts."""
    rastrigin = RastriginLandscape(dim=2)
    rng = np.random.default_rng(123)
    
    final_vals = []
    for _ in range(15):
        x = rng.uniform(-5.12, 5.12, size=2)
        for _ in range(200):
            x = x - 0.005 * rastrigin.gradient(x)
        final_vals.append(round(rastrigin.value(x), 1))
        
    unique_basins = set(final_vals)
    # Must find at least 3 distinct basins
    assert len(unique_basins) >= 3


def test_minibatch_variance_reduction():
    """Verifies that larger batch sizes strictly reduce gradient variance."""
    rng = np.random.default_rng(42)
    N = 100
    X = rng.normal(0, 1, size=(N, 2))
    y = rng.choice([-1.0, 1.0], size=(N, 1))
    mlp = SimpleMLP(in_dim=2, hidden_dim=4, out_dim=1, reg_lambda=1e-4, seed=42)
    mlp.loss(X, y)
    full_g = mlp.backward(y)
    
    # Variance for B=1
    grads_1 = [mlp.backward(y[i:i+1]) for i in range(20) if mlp.loss(X[i:i+1], y[i:i+1]) is not None]
    var_1 = np.mean([np.sum((g - full_g)**2) for g in grads_1])
    
    # Variance for B=20
    grads_20 = []
    for _ in range(20):
        idx = rng.choice(N, size=20, replace=False)
        mlp.loss(X[idx], y[idx])
        grads_20.append(mlp.backward(y[idx]))
    var_20 = np.mean([np.sum((g - full_g)**2) for g in grads_20])
    
    assert var_20 < var_1, f"Expected var(B=20) < var(B=1), got {var_20} vs {var_1}"

