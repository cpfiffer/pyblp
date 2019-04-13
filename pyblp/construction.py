"""Data construction."""

from typing import Any, Callable, Dict, Hashable, Iterator, List, Mapping, Optional

import numpy as np

from . import options
from .configurations.formulation import Formulation
from .configurations.integration import Integration
from .utilities.basics import Array, Groups, RecArray, extract_matrix, interact_ids, structure_matrices


def build_id_data(T: int, J: int, F: int) -> RecArray:
    r"""Build a balanced panel of market and firm IDs.

    This function can be used to build ``id_data`` for :class:`Simulation` initialization.

    Parameters
    ----------
    T : `int`
        Number of markets.
    J : `int`
        Number of products in each market.
    F : `int`
        Number of firms. If ``J`` is divisible by ``F``, firms produce ``J / F`` products in each market. Otherwise,
        firms with smaller IDs will produce excess products.

    Returns
    -------
    `recarray`
        IDs that associate products with markets and firms. Each of the ``T * J`` rows corresponds to a product. Fields:

            - **market_ids** : (`object`) - Market IDs that take on values from ``0`` to ``T - 1``.

            - **firm_ids** : (`object`) - Firm IDs that take on values from ``0`` to ``F - 1``.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_id_data.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # validate the counts
    if not isinstance(T, int) or not isinstance(F, int) or T < 1 or F < 1:
        raise ValueError("Both T and F must be positive ints.")
    if not isinstance(J, int) or J < F:
        raise ValueError("J must be an int that is at least F.")

    # build and structure IDs
    return structure_matrices({
        'market_ids': (np.repeat(np.arange(T), J).astype(np.int), np.object),
        'firm_ids': (np.floor(np.tile(np.arange(J), T) * F / J).astype(np.int), np.object)
    })


def build_ownership(product_data: Mapping, kappa_specification: Optional[Callable[[Any, Any], float]] = None) -> Array:
    r"""Build ownership matrices, :math:`O`.

    Ownership matrices are defined by their cooperation matrix counterparts, :math:`\kappa`. For each market :math:`t`,
    :math:`O_{jk} = \kappa_{fg}` where :math:`j \in \mathscr{J}_{ft}`, the set of products produced by firm :math:`f` in
    the market, and similarly, :math:`g \in \mathscr{J}_{gt}`.

    Parameters
    ----------
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

    kappa_specification : `callable, optional`
        A function that specifies each market's cooperation matrix, :math:`\kappa`. The function is of the following
        form::

            kappa(f, g) -> value

        where ``value`` is :math:`O_{jk}` and both ``f`` and ``g`` are firm IDs from the ``firm_ids`` field of
        ``product_data``.

        The default specification, ``lambda: f, g: int(f == g)``, constructs traditional ownership matrices. That is,
        :math:`\kappa = I`, the identify matrix, implies that :math:`O_{jk}` is :math:`1` if the same firm produces
        products :math:`j` and :math:`k`, and is :math:`0` otherwise.

        If ``firm_ids`` happen to be indices for an actual :math:`\kappa` matrix, ``lambda f, g: kappa[f, g]`` will
        build ownership matrices according to the matrix ``kappa``.

    Returns
    -------
    `ndarray`
        Stacked :math:`J_t \times J_t` ownership matrices, :math:`O`, for each market :math:`t`. If a market has fewer
        products than others, extra columns will contain ``numpy.nan``.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_ownership.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # extract and validate IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None:
        raise KeyError("product_data must have a market_ids field.")
    if firm_ids is None:
        raise KeyError("product_data must have a firm_ids field.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # validate or use the default kappa specification
    if kappa_specification is None:
        kappa_specification = lambda f, g: np.where(f == g, 1, 0).astype(options.dtype)
    elif callable(kappa_specification):
        kappa_specification = np.vectorize(kappa_specification, [options.dtype])
    if not callable(kappa_specification):
        raise ValueError("kappa_specification must be None or callable.")

    # determine the overall number of products and the maximum number in a market
    N = market_ids.size
    max_J = np.unique(market_ids, return_counts=True)[1].max()

    # construct the ownership matrices
    ownership = np.full((N, max_J), np.nan, options.dtype)
    for t in np.unique(market_ids):
        ids_t = firm_ids[market_ids.flat == t]
        tiled_ids_t = np.tile(np.c_[ids_t], ids_t.size)
        ownership[market_ids.flat == t, :ids_t.size] = kappa_specification(tiled_ids_t, tiled_ids_t.T)
    return ownership


def build_blp_instruments(formulation: Formulation, product_data: Mapping) -> Array:
    r"""Construct traditional excluded BLP instruments.

    Traditional excluded BLP instruments are

    .. math:: \text{BLP}(X) = [\text{BLP Other}(X), \text{BLP Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`\text{BLP Other}(X)` is a second matrix that
    consists of sums over characteristics of non-rival goods, and :math:`\text{BLP Rival}(X)` is a third matrix that
    consists of sums over rival goods. All three matrices have the same dimensions.

    Let :math:`x_{jt}` be the vector of characteristics in :math:`X` for product :math:`j` in market :math:`t`, which is
    produced by firm :math:`f`. That is, :math:`j \in \mathscr{J}_{ft}`. Its counterpart in :math:`\text{BLP Other}(X)`
    is

    .. math:: \sum_{r \in \mathscr{J}_{ft} \setminus \{j\}} x_{rt}.

    Its counterpart in :math:`\text{BLP Rival}(X)` is

    .. math:: \sum_{r \notin \mathscr{J}_{ft}} x_{rt}.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for :math:`X`, the matrix of product characteristics used to build excluded
        instruments. Variable names should correspond to fields in ``product_data``.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Along with ``market_ids`` and ``firm_ids``, the names of any additional fields can be used as variables in
        ``formulation``.

    Returns
    -------
    `ndarray`
        Traditional excluded BLP instruments, :math:`\text{BLP}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_blp_instruments.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # load IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None or firm_ids is None:
        raise KeyError("product_data must have market_ids and firm_ids fields.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # initialize grouping objects
    market_groups = Groups(market_ids)
    paired_groups = Groups(interact_ids(market_ids, firm_ids))

    # build the instruments
    X = build_matrix(formulation, product_data)
    other = paired_groups.expand(paired_groups.sum(X)) - X
    rival = market_groups.expand(market_groups.sum(X)) - X - other
    return np.ascontiguousarray(np.c_[other, rival])


def build_differentiation_instruments(
        formulation: Formulation, product_data: Mapping, version: str = 'local', interact: bool = False) -> Array:
    r"""Construct excluded differentiation instruments.

    Differentiation instruments in the spirit of :ref:`references:Gandhi and Houde (2017)` are

    .. math:: \text{Diff}(X) = [\text{Diff Other}(X), \text{Diff Rival}(X)],

    in which :math:`X` is a matrix of product characteristics, :math:`\text{Diff Other}(X)` is a second matrix that
    consists of sums over functions of differences between non-rival goods, and :math:`\text{Diff Rival}(X)` is a third
    matrix that consists of sums over rival goods. Without optional interaction terms, all three matrices have the same
    dimensions.

    .. note::

       To construct simpler, firm agnostic instruments that are sums over functions of differences between all different
       goods, specify a constant column of firm IDs and keep only the first half of instrument columns.

    Let :math:`x_{jtk}` be characteristic :math:`k` in :math:`X` for product :math:`j` in market :math:`t`, which is
    produced by firm :math:`f`. That is, :math:`j \in \mathscr{J}_{ft}`. Its counterpart in the "local" version of
    :math:`\text{Diff Other}(X)` is

    .. math:: \sum_{r \in \mathscr{J}_{ft} \setminus \{j\}} 1(|d_{jrtk}| < \text{SD}_k)

    where :math:`d_{jrtk} = x_{rtk} - x_{jtk}` is the difference between products :math:`j` and :math:`r` along
    dimension :math:`k`, :math:`\text{SD}_k` is the standard deviation of these pairwise differences computed across all
    markets, and :math:`1(|d_{jrtk}| < \text{SD}_k)` indicates that products :math:`j` and :math:`r` are close to each
    other in terms of characteristic :math:`k`.

    The intuition behind this "local" version is that demand for products is often most influenced by a small number of
    other goods that are very similar. For the "quadratic" version of :math:`\text{Diff Other}(X)`, which uses a more
    continuous measure of the distance between goods, we instead have

    .. math:: \sum_{r \in \mathscr{J}_{ft} \setminus \{j\}} d_{jrtk}^2.

    Counterparts for the "local" and "quadratic" versions of :math:`\text{Diff Rival}(X)` are

    .. math:: \sum_{r \notin \mathscr{J}_{ft}} 1(|d_{jrtk}| < \text{SD}_k)

    and

    .. math:: \sum_{r \notin \mathscr{J}_{ft}} d_{jrtk}^2.

    With interaction terms, which reflect covariances between different characteristics, the summands for the "local"
    versions are :math:`1(|d_{jrtk}| < \text{SD}_k) \times d_{jrt\ell}` for all characteristics :math:`\ell`, and the
    summands for the "quadratic" versions are :math:`d_{jrtk} \times d_{jrt\ell}` for all :math:`\ell \geq k`.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for :math:`X`, the matrix of product characteristics used to build excluded
        instruments. Variable names should correspond to fields in ``product_data``.
    product_data : `structured array-like`
        Each row corresponds to a product. Markets can have differing numbers of products. The following fields are
        required:

            - **market_ids** : (`object`) - IDs that associate products with markets.

            - **firm_ids** : (`object`) - IDs that associate products with firms.

        Along with ``market_ids`` and ``firm_ids``, the names of any additional fields can be used as variables in
        ``formulation``.

    version : `str, optional`
        The version of differentiation instruments to construct:

            - ``'local'`` (default) - Construct instruments that consider only the characteristics of "close" products
              in each market.

            - ``'quadratic'`` - Construct more continuous instruments that consider all products in each market.

    interact : `bool, optional`
        Whether to include interaction terms between different product characteristics, which can help capture
        covariances between product characteristics.

    Returns
    -------
    `ndarray`
        Excluded differentiation instruments, :math:`\text{Diff}(X)`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_differentiation_instruments.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # load IDs
    market_ids = extract_matrix(product_data, 'market_ids')
    firm_ids = extract_matrix(product_data, 'firm_ids')
    if market_ids is None or firm_ids is None:
        raise KeyError("product_data must have market_ids and firm_ids fields.")
    if market_ids.shape[1] > 1:
        raise ValueError("The market_ids field of product_data must be one-dimensional.")
    if firm_ids.shape[1] > 1:
        raise ValueError("The firm_ids field of product_data must be one-dimensional.")

    # identify markets
    market_indices = {t: np.where(market_ids.flat == t) for t in np.unique(market_ids)}

    # build the matrix and count its dimensions
    X = build_matrix(formulation, product_data)
    N, K = X.shape

    # build distance matrices for each characteristic and market, nullifying distances between the same products
    distances_mapping: Dict[int, Array] = {}
    for k in range(K):
        distances_mapping[k] = {}
        for t, indices in market_indices.items():
            x = X[indices, [k]]
            distances = x - x.T
            np.fill_diagonal(distances, np.nan)
            distances_mapping[k][t] = distances

    # compute standard deviations of pairwise differences across all markets
    sd_mapping: Dict[int, Array] = {}
    if version == 'local':
        for k in range(K):
            sd_mapping[k] = np.nanstd(np.hstack(d.flatten() for d in distances_mapping[k].values()))

    # define a function that generates market-level terms used to create instruments
    def generate_instrument_terms(t: Hashable) -> Iterator[Array]:
        """Generate terms for a market that will be summed to create instruments."""
        for k1 in range(K):
            if version == 'quadratic':
                for k2 in range(k1, K if interact else k1 + 1):
                    yield np.nan_to_num(distances_mapping[k1][t]) * np.nan_to_num(distances_mapping[k2][t])
            elif version == 'local':
                with np.errstate(invalid='ignore'):
                    close = np.nan_to_num(np.abs(distances_mapping[k1][t]) < sd_mapping[k1])
                if not interact:
                    yield close
                else:
                    for k2 in range(K):
                        yield close * np.nan_to_num(distances_mapping[k2][t])
            else:
                raise ValueError("version must be 'local' or 'quadratic'.")

    # create the instruments
    other_blocks: List[List[Array]] = []
    rival_blocks: List[List[Array]] = []
    for t, indices in market_indices.items():
        other_blocks.append([])
        rival_blocks.append([])
        ownership = firm_ids[indices] == firm_ids[indices].T
        for term in generate_instrument_terms(t):
            other_blocks[-1].append((ownership * term).sum(axis=1, keepdims=True))
            rival_blocks[-1].append((~ownership * term).sum(axis=1, keepdims=True))
    return np.c_[np.block(other_blocks), np.block(rival_blocks)]


def build_integration(integration: Integration, dimensions: int) -> RecArray:
    r"""Build nodes and weights for integration over agent choice probabilities.

    This function can be used to build custom ``agent_data`` for :class:`Problem` initialization. Specifically, this
    function affords more flexibility than passing an :class:`Integration` configuration directly to :class:`Problem`.
    For example, if agents have unobserved tastes over only a subset of nonlinear product characteristics (i.e., if
    there are zeros on the diagonal of ``sigma`` in :meth:`Problem.solve`), this function can be used to build agent
    data with fewer columns of integration nodes than the number of unobserved product characteristics, :math:`K_2`.
    This function can also be used to construct nodes that are transformed into demographic variables.

    To build nodes and weights for multiple markets, this function can be called multiple times, once for each market.

    Parameters
    ----------
    integration : `Integration`
        :class:`Integration` configuration for how to build nodes and weights for integration.
    dimensions : `int`
        Number of dimensions over which to integrate, or equivalently, the number of columns of integration nodes.
        When an :class:`Integration` configuration is passed directly to :class:`Problem`, this is the number of
        nonlinear product characteristics, :math:`K_2`.

    Returns
    -------
    `recarray`
        Nodes and weights for integration over agent utilities. Fields:

            - **weights** : (`numeric`) - Integration weights, :math:`w`.

            - **nodes** : (`numeric`) - Unobserved agent characteristics called integration nodes, :math:`\nu`.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_integration.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """

    # validate the arguments
    if not isinstance(integration, Integration):
        raise TypeError("integration must be an Integration instance.")
    if not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer.")

    # build and structure the nodes and weights
    nodes, weights = integration._build(dimensions)
    return structure_matrices({
        'weights': (weights, options.dtype),
        'nodes': (nodes, options.dtype)
    })


def build_matrix(formulation: Formulation, data: Mapping) -> Array:
    r"""Construct a matrix according to a formulation.

    Parameters
    ----------
    formulation : `Formulation`
        :class:`Formulation` configuration for the matrix. Variable names should correspond to fields in ``data``.
    data : `structured array-like`
        Fields can be used as variables in ``formulation``.

    Returns
    -------
    `ndarray`
        The built matrix.

    Examples
    --------
    .. raw:: latex

       \begin{examplenotebook}

    .. toctree::

       /_notebooks/api/build_matrix.ipynb

    .. raw:: latex

       \end{examplenotebook}

    """
    if not isinstance(formulation, Formulation):
        raise TypeError("formulation must be a Formulation instance.")
    if formulation._absorbed_terms:
        raise ValueError("formulation does not support fixed effect absorption.")
    return formulation._build_matrix(data)[0]
