import pandas as pd
import numpy as np
import alphalens as al
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

npattern = '\d'
lpattern = '[a-zA-Z]'
DECIMAL_TO_BPS = 10000

def get_intraday_clean_factor_and_forward_returns(factor,
                                         prices,
                                         frq,
                                         groupby=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None,
                                         max_loss=0.35,
                                         zero_aware=False,
                                         cumulative_returns=True):
    """
    this method mimic alphalens.utils.get_clean_factor_and_forward_returns except it take
    intra day time series instead of daily time series.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        ::
        -----------------------------------------------
                   date         |    asset   |
        -----------------------------------------------
                                |   AAPL     |   0.5
                                -----------------------
                                |   BA       |  -1.1
                                -----------------------
        2014-01-01 10:00:00     |   CMG      |   1.7
                                -----------------------
                                |   DAL      |  -0.1
                                -----------------------
                                |   LULU     |   2.7
                                -----------------------

    prices : pd.DataFrame
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns.
        Pricing data must span the factor analysis time period plus an
        additional buffer window that is greater than the maximum number
        of expected periods in the forward returns calculations.
        It is important to pass the correct pricing data in depending on
        what time of period your signal was generated so to avoid lookahead
        bias, or  delayed calculations.
        'Prices' must contain at least an entry for each timestamp/asset
        combination in 'factor'. This entry should reflect the buy price
        for the assets and usually it is the next available price after the
        factor is computed but it can also be a later price if the factor is
        meant to be traded later (e.g. if the factor is computed at market
        open but traded 1 hour after market open the price information should
        be 1 hour after market open).
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        ::
          --------------------------------------------------------------
                                | AAPL |  BA  |  CMG  |  DAL  |  LULU  |
          --------------------------------------------------------------
            Date                |      |      |       |       |        |
          --------------------------------------------------------------
           2014-01-01 10:00:00  |605.12| 24.58|  11.72| 54.43 |  37.14 |
          --------------------------------------------------------------
           2014-01-01 11:00:00  |604.35| 22.23|  12.21| 52.78 |  33.63 |
          --------------------------------------------------------------
           2014-01-01 12:00:00  |607.94| 21.68|  14.36| 53.94 |  29.37 |
          --------------------------------------------------------------

    groupby : pd.Series - MultiIndex or dict
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    binning_by_group : bool
        If True, compute quantile buckets separately for each group.
        This is useful when the factor values range vary considerably
        across gorups so that it is wise to make the binning group relative.
        You should probably enable this if the factor is intended
        to be analyzed for a group neutral portfolio
    quantiles : int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0, .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-None
    bins : int or sequence[float]
        Number of equal-width (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
        Only one of 'quantiles' or 'bins' can be not-None
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels : dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.
    max_loss : float, optional
        Maximum percentage (0.00 to 1.00) of factor data dropping allowed,
        computed comparing the number of items in the input factor index and
        the number of items in the output DataFrame index.
        Factor data can be partially dropped due to being flawed itself
        (e.g. NaNs), not having provided enough price data to compute
        forward returns for all factor values, or because it is not possible
        to perform binning.
        Set max_loss=0 to avoid Exceptions suppression.
    zero_aware : bool, optional
        If True, compute quantile buckets separately for positive and negative
        signal values. This is useful if your signal is centered and zero is
        the separation between long and short signals, respectively.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    merged_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.levels[0].freq) will be
          set to a trading calendar (pandas DateOffset) inferred from the input
          data (see infer_trading_calendar for more details). This is currently
          used only in cumulative returns computation
        ::
        ----------------------------------------------------------------------------
                            |       | 1D  | 5D  | 10D  |factor|group|factor_quantile
        ----------------------------------------------------------------------------
                date        | asset |     |     |      |      |     |
        ----------------------------------------------------------------------------
                            | AAPL  | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                            --------------------------------------------------------
                            | BA    | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                            --------------------------------------------------------
        2014-01-01 10:00:00 | CMG   | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                            --------------------------------------------------------
                            | DAL   |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                            --------------------------------------------------------
                            | LULU  |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
                            --------------------------------------------------------
    """
    forward_returns = compute_intraday_forward_returns(factor,prices,frq,periods,filter_zscore,cumulative_returns)
    
    factor_data = al.utils.get_clean_factor(factor, forward_returns, groupby=groupby,
                                   groupby_labels=groupby_labels,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)
    return factor_data

def compute_intraday_forward_returns(factor,
                            prices,
                            freq_label,
                            periods=(1, 5, 10),
                            filter_zscore=None,
                            cumulative_returns=True):
    """
    this method mimic alphalens.utils.compute_forward_returns except that it take
    intra day time series.

    Parameters
    ----------
    factor : pd.Series - MultiIndex
        A MultiIndex Series indexed by timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.

        - See full explanation in utils.get_clean_factor_and_forward_returns

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.
    filter_zscore : int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    cumulative_returns : bool, optional
        If True, forward returns columns will contain cumulative returns.
        Setting this to False is useful if you want to analyze how predictive
        a factor is for a single forward day.

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by timestamp (level 0) and asset
        (level 1), containing the forward returns for assets.
        Forward returns column names follow the format accepted by
        pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc).
        'date' index freq property (forward_returns.index.levels[0].freq)
        will be set to a trading calendar (pandas DateOffset) inferred
        from the input data (see infer_trading_calendar for more details).
    """
    factor_dateindex = factor.index.levels[0]
    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = \
            returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())
            forward_returns[mask] = np.nan

        label = str(int(re.sub(lpattern,'',freq_label))*period) + re.sub(npattern,'',freq_label)
        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),
        inplace=True
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]

    #df.index.levels[0].freq = freq
    df.index.set_names(["date", "asset"])

    return df

def bull_bear_prob_by_quantile(factor_data,
                            by_group=False):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
        
    Returns
    -------
    bull_bear_prob : pd.DataFrame
        probability of up by specified factor quantile adjusted by 50.
        A positive reading indicate the probability of up is greater than 50%.
    """

    factor_data = factor_data.copy()

    grouper = ['factor_quantile']

    if by_group:
        grouper.append('group')

    bull_bear_prob = factor_data.groupby(grouper)[
        al.utils.get_forward_returns_columns(factor_data.columns)] \
        .apply(lambda x: np.sum(x>0)/len(x)*100 - 50)

    return bull_bear_prob

def factor_ic_by_asset(factor_data,
                                   by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group['factor']
        _ic = group[al.utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values('asset')]

    if by_group:
        grouper = ['group']

    ic = factor_data.groupby(grouper).apply(src_ic)

    return ic

def plot_q_bb_prob_bar(bull_bear_prob_by_q,
                              by_group=False,
                              ylim_percentiles=None,
                              ax=None):
    """
    Plots bull/bear index period wise for factor quantiles.

    Parameters
    ----------
    bull_bear_prob_by_q : pd.DataFrame
        DataFrame with quantile, (group) and bull/bear period wise probability(-50).
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    bull_bear_prob_by_q = bull_bear_prob_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(bull_bear_prob_by_q.values,
                                 ylim_percentiles[0]))
        ymax = (np.nanpercentile(bull_bear_prob_by_q.values,
                                 ylim_percentiles[1]))
    else:
        ymin = -50
        ymax = 50

    if by_group:
        num_group = len(
            bull_bear_prob_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(v_spaces, 2, sharex=False,
                                 sharey=True, figsize=(18, 6 * v_spaces))
            ax.set_yticks(np.arange(-5,6)*10)
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, bull_bear_prob_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
                .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Prob of Up (%) -50',
                  ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))
            ax.set_yticks(np.arange(-5,6)*10)

        (bull_bear_prob_by_q
            .plot(kind='bar',
                  title="Bull bear index By Factor Quantile", ax=ax))
        ax.set(xlabel='', ylabel='Prob of Up (%) -50',
               ylim=(ymin, ymax))

        return ax

def plot_quantile_rr_skew_violin(return_by_q,
                                 ylim_percentiles=None,
                                 ax=None):
    """
    Plots a violin box plot of risk rewards skews for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = (return_by_q
                    .multiply(DECIMAL_TO_BPS))
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='RR Skew (bps)',
           title="Risk Rewards Skew By Factor Quantile",
           ylim=(ymin, ymax))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax

def plot_quantile_rr_ratio_violin(return_by_q,
                                 ylim_percentiles=None,
                                 ax=None):
    """
    Plots a violin box plot of risk rewards ratio for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]))
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]))
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    unstacked_dr = (return_by_q)
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='RR Ratio',
           title="Risk Rewards Ratio By Factor Quantile",
           ylim=(ymin, ymax))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax