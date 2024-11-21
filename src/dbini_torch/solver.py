import torch


def conjugate_gradient_solver(
    A: torch.sparse.Tensor,
    b: torch.Tensor,
    maxiter: int,
    x0: torch.Tensor | None = None,
    M: torch.sparse.Tensor | None = None,
    rtol: float = 1e-5,
    atol: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """
    Conjugate gradient solver optimized for GPU execution.
    Solves Ax = b with convergence test: norm(r) <= max(rtol*norm(b), atol)

    Args:
        A: Sparse system matrix (n x n)
        b: Right-hand side vector (n)
        maxiter: Maximum number of iterations
        x0: Initial guess vector (optional)
        M: Preconditioner sparse matrix (optional)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        tuple[torch.Tensor, int]: (solution vector, convergence flag)
        convergence flag: 0 if converged, >0 if hit maxiter

    Raises:
        RuntimeError: If A is not positive definite or numerical breakdown
    """
    n = b.shape[0]

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    if M is None:
        indices = torch.arange(2 * n, device=b.device).reshape(2, n)
        M = torch.sparse_coo_tensor(
            indices=indices, values=torch.ones(n, device=b.device), size=(n, n)
        )

    r = torch.empty_like(b)
    z = torch.empty_like(b)
    p = torch.empty_like(b)
    Ap = torch.empty_like(b)

    if x.any():
        r = A @ x
        r.neg_().add_(b)
    else:
        r.copy_(b)

    bnorm = torch.linalg.norm(b)
    if bnorm == 0:
        return x, 0
    tol = max(rtol * bnorm, atol)

    z = M @ r
    p.copy_(z)
    rz = torch.dot(r, z)

    for iteration in range(maxiter):
        if torch.linalg.norm(r) <= tol:
            return x, 0

        Ap = A @ p
        pAp = torch.dot(p, Ap)

        if pAp <= 0:
            raise RuntimeError("Matrix A is not positive definite")

        alpha = rz / pAp
        # Update x and r directly without addmm_
        x += alpha * p
        r += -alpha * Ap

        z = M @ r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz

        # Update p directly
        p = z + beta * p
        rz = rz_new

        if torch.isnan(rz) or torch.isinf(rz):
            raise RuntimeError("Numerical breakdown in conjugate gradient solver")

    return x, maxiter