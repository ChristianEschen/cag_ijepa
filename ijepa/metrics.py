import torch
from torch import Tensor
from enum import Enum
DEFAULT_MAX_SAMPLES = 25_600  # number used in the paper
EPSILON = 1e-7  # suitable for float32


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def calc_rankme(embeddings: Tensor, epsilon: float = EPSILON) -> float:
    """
    Calculate the RankMe score (the higher, the better).
    RankMe(Z) = exp (
        - sum_{k=1}^{min(N, K)} p_k * log(p_k)
    ),
    where p_k = sigma_k (Z) / ||sigma_k (Z)||_1 + epsilon
    where sigma_k is the kth singular value of Z.
    where Z is the matrix of embeddings
    RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank
    https://arxiv.org/pdf/2210.02885.pdf
    Args:
        embeddings: the embeddings to calculate the RankMe score for
        epsilon: the epsilon value to use for the calculation. The paper recommends 1e-7 for float32.
    Returns:
        the RankMe score
    """
    # average across second dimension
  #  embeddings = torch.mean(embeddings, dim=1)
    print('embed shape', embeddings.shape)
    print('embed max', torch.max(embeddings))
    print('embed min', torch.min(embeddings))
    embeddings = embeddings / torch.norm(
        embeddings, dim=1, keepdim=True
    )
    # cast embeddings to float32
    embeddings = embeddings.to(torch.float32)

    # compute the singular values of the embeddings
    _u, s, _vh = torch.linalg.svd(
        embeddings, full_matrices=False
    )  # s.shape = (min(N, K),)

    # normalize the singular values to sum to 1 [[Eq. 2]]
    p = (s / torch.sum(s, axis=0)) + epsilon
    # if torch.any(p < 1e-5) or torch.any(p > (1 - 1e-5)):
    #     print("Problematic p values detected!")
    #p = torch.clamp(p, min=epsilon, max=1-epsilon)


    # RankMe score is the exponential of the entropy of the singular values [[Eq. 1]]
    # this is sometimes called the `perplexity` in information theory
    entropy = -torch.sum(p * torch.log(p))
    rankme = torch.exp(entropy)#.item()

    # test if rankme is nan

    # if torch.isinf(rankme):
    #     print('stop')
    return rankme