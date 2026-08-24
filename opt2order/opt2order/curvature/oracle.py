"""Curvature oracles: Hessian-vector and Gauss-Newton-vector products.

Implements the primitives described in Section 4.1.2 of the thesis.
The HVP uses the standard double-backward construction; the GNVP uses
the Pearlmutter-style J^T H_L J v decomposition that preserves
positive semi-definiteness when the loss is convex in the model output.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Parameter flattening utilities
# ---------------------------------------------------------------------------

def flatten(tensors: Iterable[Tensor]) -> Tensor:
    """Flatten an iterable of tensors into a single 1-D tensor."""
    return torch.cat([t.contiguous().view(-1) for t in tensors])


def unflatten_to(flat: Tensor, like: Sequence[Tensor]) -> List[Tensor]:
    """Split a flat 1-D tensor into pieces shaped like ``like``."""
    out, offset = [], 0
    for t in like:
        n = t.numel()
        out.append(flat[offset:offset + n].view_as(t))
        offset += n
    if offset != flat.numel():
        raise ValueError(
            f"flat tensor has {flat.numel()} elements but template needs {offset}")
    return out


def flat_grad(loss: Tensor, params: Sequence[Tensor],
              create_graph: bool = False, retain_graph: bool = False) -> Tensor:
    """Return ∇_params loss as a single flat 1-D tensor."""
    grads = torch.autograd.grad(
        loss, params,
        create_graph=create_graph,
        retain_graph=retain_graph or create_graph,
        allow_unused=True,
    )
    pieces = []
    for g, p in zip(grads, params):
        pieces.append(torch.zeros_like(p) if g is None else g)
    return flatten(pieces)


# ---------------------------------------------------------------------------
# Hessian-vector product
# ---------------------------------------------------------------------------

def hvp(loss_closure: Callable[[], Tensor],
        params: Sequence[Tensor],
        vec: Tensor) -> Tensor:
    """Compute H(theta) @ vec, where H is the Hessian of the scalar loss.

    Parameters
    ----------
    loss_closure
        Zero-arg function returning the scalar loss with a fresh graph each
        call.  Must be callable repeatedly.
    params
        The parameters with respect to which the Hessian is taken.
    vec
        Flat 1-D tensor of the same length as the concatenated parameters.

    Returns
    -------
    Tensor
        H @ vec, as a flat 1-D tensor.
    """
    loss = loss_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = flatten(grads)
    inner = torch.dot(flat, vec)
    hv = torch.autograd.grad(inner, params, retain_graph=False, allow_unused=True)
    pieces = [torch.zeros_like(p) if h is None else h for h, p in zip(hv, params)]
    return flatten(pieces)


# ---------------------------------------------------------------------------
# Gauss-Newton-vector product (G v)
# ---------------------------------------------------------------------------

def gnvp(model_output_closure: Callable[[], Tensor],
         loss_fn: Callable[[Tensor], Tensor],
         params: Sequence[Tensor],
         vec: Tensor) -> Tensor:
    """Compute G(theta) @ vec for the Gauss-Newton matrix G.

    G = J^T H_L J, where J is the Jacobian of the model output w.r.t.
    parameters and H_L is the Hessian of the loss w.r.t. the model
    output.  When the loss is convex in the output (cross-entropy, MSE),
    G is positive semi-definite, which is what we want for CG.

    Computed via three autograd calls:
      1. v_out = J v        (Pearlmutter trick: forward-mode by VJP-of-VJP)
      2. h_out = H_L v_out  (Hessian of scalar loss w.r.t. output)
      3. result = J^T h_out (standard reverse-mode VJP)
    """
    output = model_output_closure()           # shape: (B, ...) with grad
    output_flat = output.reshape(-1)

    # Step 1: J v using the double-backward trick.
    # We compute u = (∂output_flat / ∂params) v by:
    #   - making a dummy r with the shape of output_flat, requiring grad
    #   - computing g = ∂(output_flat · r) / ∂params  (these are ordinary VJPs;
    #     g is a function of r, linear in r)
    #   - then J v = ∂(g · v) / ∂r
    r = torch.zeros_like(output_flat, requires_grad=True)
    grads_r = torch.autograd.grad(
        output_flat, params, grad_outputs=r,
        create_graph=True, retain_graph=True, allow_unused=True,
    )
    grads_r = [torch.zeros_like(p) if gr is None else gr
               for gr, p in zip(grads_r, params)]
    flat_grads_r = flatten(grads_r)
    Jv = torch.autograd.grad(flat_grads_r, r, grad_outputs=vec,
                             retain_graph=True)[0]   # shape: output_flat

    # Step 2: H_L (J v).  Loss is a scalar function of output;
    # H_L is the Hessian of loss w.r.t. output.
    # We use the trick: H_L u = ∂(∂loss/∂output · u) / ∂output.
    output_for_loss = output_flat.detach().clone().requires_grad_(True)
    loss = loss_fn(output_for_loss.view_as(output))
    g_out = torch.autograd.grad(loss, output_for_loss, create_graph=True)[0]
    Hv_out = torch.autograd.grad(
        g_out, output_for_loss, grad_outputs=Jv.view_as(g_out),
        retain_graph=False)[0].reshape(-1)

    # Step 3: J^T (H_L J v) — a standard VJP.
    Jt_Hv = torch.autograd.grad(
        output_flat, params, grad_outputs=Hv_out,
        retain_graph=False, allow_unused=True,
    )
    Jt_Hv = [torch.zeros_like(p) if g is None else g
             for g, p in zip(Jt_Hv, params)]
    return flatten(Jt_Hv)
