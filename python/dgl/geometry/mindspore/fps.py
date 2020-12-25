"""Farthest Point Sampler for MindSpore Geometry package"""
#pylint: disable=no-member, invalid-name
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

from ..capi import farthest_point_sampler


class FarthestPointSampler(nn.Cell):
    """Farthest Point Sampler

    In each batch, the algorithm starts with the sample index specified by ``start_idx``.
    Then for each point, we maintain the minimum to-sample distance.
    Finally, we pick the point with the maximum such distance.
    This process will be repeated for ``sample_points`` - 1 times.

    Parameters
    ----------
    npoints : int
        The number of points to sample in each batch.
    """
    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def construct(self, pos):
        r"""Memory allocation and sampling

        Parameters
        ----------
        pos : tensor
            The positional tensor of shape (B, N, C)

        Returns
        -------
        tensor of shape (B, self.npoints)
            The sampled indices in each batch.
        """
        B, N, C = pos.shape
        dtype = pos.dtype

        pos = pos.asnumpy().reshape(-1, C)
        dist = np.zeros((B * N), dtype=pos.dtype)
        start_idx = np.random.randint(0, N - 1, (B, ), dtype=np.int)
        result = np.zeros((self.npoints * B), dtype=np.int)
        farthest_point_sampler(pos, B, self.npoints, dist, start_idx, result)

        return Tensor(result.reshape(B, self.npoints), dtype=dtype)
