import sys

import torch

from dist import LOGGER


def lstq(A, F_u, F_v, lamda=0.01):
    # cols = A.shape[2]
    # assert all(cols == torch.linalg.matrix_rank(A))  # something better?
    try:
        Q, R = torch.linalg.qr(A)
        theta_x = torch.bmm(torch.bmm(torch.linalg.inv(R), Q.transpose(1, 2)), F_u)
        theta_y = torch.bmm(torch.bmm(torch.linalg.inv(R), Q.transpose(1, 2)), F_v)
    except:
        LOGGER.exception("Least Squares failed")
        sys.exit(-1)
    return theta_x, theta_y

def get_quad_flow(masks_softmaxed, flow, grid_x, grid_y):
    rec_flow = 0
    for k in range(masks_softmaxed.size(1)):
        mask = masks_softmaxed[:, k].unsqueeze(1)
        _F = flow * mask
        M = mask.flatten(1)
        bs = _F.shape[0]
        x = grid_x.unsqueeze(0).flatten(1)
        y = grid_y.unsqueeze(0).flatten(1)

        F_u = _F[:, 0].flatten(1).unsqueeze(2)  # B x L x 1
        F_v = _F[:, 1].flatten(1).unsqueeze(2)  # B x L x 1
        A = torch.stack([x * M, y * M, x*x *M, y*y*M, x*y*M, torch.ones_like(y) * M], 2)  # B x L x 2

        theta_x, theta_y = lstq(A, F_u, F_v, lamda=.01)
        rec_flow_m = torch.stack([torch.einsum('bln,bnk->blk', A, theta_x).view(bs, *grid_x.shape),
                                      torch.einsum('bln,bnk->blk', A, theta_y).view(bs, *grid_y.shape)], 1)

        rec_flow += rec_flow_m
    return rec_flow


SUBSAMPLE = 8
SKIP = 0.4
SIZE = 0.3
NITER = 50
METHOD = 'inv_score'

def set_subsample_skip(sub=None, skip=None, size=None, niter=None, method=None):
    global SUBSAMPLE, SKIP, SIZE, NITER, METHOD
    if sub is not None: SUBSAMPLE=sub
    if skip is not None: SKIP=skip
    if size is not None: SIZE=size
    if niter is not None: NITER=niter
    if method is not None: METHOD=method
