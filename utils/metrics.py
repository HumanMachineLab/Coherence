# Natural Language Toolkit: Text Segmentation Metrics
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         David Doukhan <david.doukhan@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT


"""
Text Segmentation Metrics

1. Windowdiff

Pevzner, L., and Hearst, M., A Critique and Improvement of
  an Evaluation Metric for Text Segmentation,
Computational Linguistics 28, 19-36


2. Generalized Hamming Distance

Bookstein A., Kulyukin V.A., Raita T.
Generalized Hamming Distance
Information Retrieval 5, 2002, pp 353-375

Baseline implementation in C++
http://digital.cs.usu.edu/~vkulyukin/vkweb/software/ghd/ghd.html

Study describing benefits of Generalized Hamming Distance Versus
WindowDiff for evaluating text segmentation tasks
Begsten, Y.  Quel indice pour mesurer l'efficacite en segmentation de textes ?
TALN 2009


3. Pk text segmentation metric

Beeferman D., Berger A., Lafferty J. (1999)
Statistical Models for Text Segmentation
Machine Learning, 34, 177-210
"""

try:
    import numpy as np
except ImportError:
    pass


def windowdiff(seg1, seg2, k, boundary="1", weighted=False):
    """
    Compute the windowdiff score for a pair of segmentations.  A
    segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

        >>> s1 = "000100000010"
        >>> s2 = "000010000100"
        >>> s3 = "100000010000"
        >>> '%.2f' % windowdiff(s1, s1, 3)
        '0.00'
        >>> '%.2f' % windowdiff(s1, s2, 3)
        '0.30'
        >>> '%.2f' % windowdiff(s2, s3, 3)
        '0.80'

    :param seg1: a segmentation
    :type seg1: str or list
    :param seg2: a segmentation
    :type seg2: str or list
    :param k: window width
    :type k: int
    :param boundary: boundary value
    :type boundary: str or int or bool
    :param weighted: use the weighted variant of windowdiff
    :type weighted: boolean
    :rtype: float
    """

    if len(seg1) != len(seg2):
        raise ValueError("Segmentations have unequal length")
    if k > len(seg1):
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0
    for i in range(len(seg1) - k + 1):
        ndiff = abs(seg1[i : i + k].count(boundary) - seg2[i : i + k].count(boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (len(seg1) - k + 1.0)


# Generalized Hamming Distance


def _init_mat(nrows, ncols, ins_cost, del_cost):
    mat = np.empty((nrows, ncols))
    mat[0, :] = ins_cost * np.arange(ncols)
    mat[:, 0] = del_cost * np.arange(nrows)
    return mat


def _ghd_aux(mat, rowv, colv, ins_cost, del_cost, shift_cost_coeff):
    for i, rowi in enumerate(rowv):
        for j, colj in enumerate(colv):
            shift_cost = shift_cost_coeff * abs(rowi - colj) + mat[i, j]
            if rowi == colj:
                # boundaries are at the same location, no transformation required
                tcost = mat[i, j]
            elif rowi > colj:
                # boundary match through a deletion
                tcost = del_cost + mat[i, j + 1]
            else:
                # boundary match through an insertion
                tcost = ins_cost + mat[i + 1, j]
            mat[i + 1, j + 1] = min(tcost, shift_cost)


def ghd(ref, hyp, ins_cost=2.0, del_cost=2.0, shift_cost_coeff=1.0, boundary="1"):
    """
    Compute the Generalized Hamming Distance for a reference and a hypothetical
    segmentation, corresponding to the cost related to the transformation
    of the hypothetical segmentation into the reference segmentation
    through boundary insertion, deletion and shift operations.

    A segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

    Recommended parameter values are a shift_cost_coeff of 2.
    Associated with a ins_cost, and del_cost equal to the mean segment
    length in the reference segmentation.

        >>> # Same examples as Kulyukin C++ implementation
        >>> ghd('1100100000', '1100010000', 1.0, 1.0, 0.5)
        0.5
        >>> ghd('1100100000', '1100000001', 1.0, 1.0, 0.5)
        2.0
        >>> ghd('011', '110', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('1', '0', 1.0, 1.0, 0.5)
        1.0
        >>> ghd('111', '000', 1.0, 1.0, 0.5)
        3.0
        >>> ghd('000', '111', 1.0, 2.0, 0.5)
        6.0

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the hypothetical segmentation
    :type hyp: str or list
    :param ins_cost: insertion cost
    :type ins_cost: float
    :param del_cost: deletion cost
    :type del_cost: float
    :param shift_cost_coeff: constant used to compute the cost of a shift.
    shift cost = shift_cost_coeff * |i - j| where i and j are
    the positions indicating the shift
    :type shift_cost_coeff: float
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """

    ref_idx = [i for (i, val) in enumerate(ref) if val == boundary]
    hyp_idx = [i for (i, val) in enumerate(hyp) if val == boundary]

    nref_bound = len(ref_idx)
    nhyp_bound = len(hyp_idx)

    if nref_bound == 0 and nhyp_bound == 0:
        return 0.0
    elif nref_bound > 0 and nhyp_bound == 0:
        return nref_bound * ins_cost
    elif nref_bound == 0 and nhyp_bound > 0:
        return nhyp_bound * del_cost

    mat = _init_mat(nhyp_bound + 1, nref_bound + 1, ins_cost, del_cost)
    _ghd_aux(mat, hyp_idx, ref_idx, ins_cost, del_cost, shift_cost_coeff)
    return mat[-1, -1]


# Beeferman's Pk text segmentation evaluation metric


def pk(ref, hyp, k=None, boundary="1"):
    """
    Compute the Pk metric for a pair of segmentations A segmentation
    is any sequence over a vocabulary of two items (e.g. "0", "1"),
    where the specified boundary value is used to mark the edge of a
    segmentation.

    >>> '%.2f' % pk('0100'*100, '1'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0100'*100, 2)
    '0.00'

    :param ref: the reference segmentation
    :type ref: str or list
    :param hyp: the segmentation to evaluate
    :type hyp: str or list
    :param k: window size, if None, set to half of the average reference segment length
    :type boundary: str or int or bool
    :param boundary: boundary value
    :type boundary: str or int or bool
    :rtype: float
    """

    if k is None:
        k = int(round(len(ref) / (ref.count(boundary) * 2.0)))

    err = 0
    for i in range(len(ref) - k + 1):
        r = ref[i : i + k].count(boundary) > 0
        h = hyp[i : i + k].count(boundary) > 0
        if r != h:
            err += 1
    return err / (len(ref) - k + 1.0)


# skip doctests if numpy is not installed
def setup_module(module):
    from nose import SkipTest

    try:
        import numpy
    except ImportError:
        raise SkipTest("numpy is required for nltk.metrics.segmentation")


# returns the number of times a prediction was made correctly exactly, very near, or near the real target
def get_proximity(real, predictions, proximity=2):
    exact_matches = 0
    very_close_matches = 0  # within 1 step
    close_matches = 0  # within 2 steps

    assert len(predictions) == len(
        real
    ), "predictions and real should be the same length."
    for i, (p, r) in enumerate(zip(predictions, real)):
        # take care of all the cases where we're at the
        # beginning of the arrays
        prev_idx = i - 1
        prev_prev_idx = i - 2
        if i == 0 or i == 1:
            prev_idx = 0
            prev_prev_idx = 0

        prev_prediction = predictions[prev_idx]
        prev_prev_prediction = predictions[prev_prev_idx]

        # take care of all the cases where we're at the
        # end of the arrays
        next_idx = i + 1
        next_next_idx = i + 2
        if i == (len(predictions) - 1):
            next_idx = i
            next_next_idx = i
        if i == (len(predictions) - 2):
            next_idx = i + 1
            next_next_idx = i + 1

        next_prediction = predictions[next_idx]
        next_next_prediction = predictions[next_next_idx]

        if r == 1 and p == 1:
            exact_matches += 1
            continue
        if r == 1:
            if prev_prediction == 1 or next_prediction == 1:
                very_close_matches += 1
                continue
            if prev_prev_prediction == 1 or next_next_prediction == 1:
                close_matches += 1
                continue
            continue

    num_positive_predictions = predictions.count(1) or 1 # set to 1 if there are 0 predictions
    num_real_positives = real.count(1) or 1 # set to 1 if there are 0 predictions

    proximity = (
        (1 / 3 * (close_matches / num_real_positives))
        + (1 / 2 * (very_close_matches / num_real_positives))
        + (exact_matches / num_real_positives)
    ) * (
        (1 / 3 * (close_matches / num_positive_predictions))
        + (1 / 2 * (very_close_matches / num_positive_predictions))
        + (exact_matches / num_positive_predictions)
    )

    # proximity *= 100

    return proximity, exact_matches, very_close_matches, close_matches
