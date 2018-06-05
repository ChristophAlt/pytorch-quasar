from toolz import dicttoolz


def trials_to_dimensions(trials, precision=6, remove_single_values=True):
    """
    Converts hyperparameter configurations into the Plotly dimensions, e.g. used
    to plot parallel coordinates.


    Args:
        trials (list): Hyperparameter configurations for a sequence of trials.
        A trial is represented by a dict of hyperparameters and corresponding values.
        precision (int, optional): The precision for displaying values of type float.
        remove_single_values (bool, optional): If true, remove hyperparameters with a
        single unique value.

    Returns:
        :class:`list` of :class:`dict`: A list of dictionaries describing each
        hyperparameter dimension.
    """
    # TODO: If floats are below 1., switch to exponential notation
    # TODO: Scale axis to prevent ticks from overlapping

    def to_dimension_dict(hparam):
        label, values = hparam

        elem = values[0]
        if isinstance(elem, str):
            unique = list(sorted(set(values)))
            stoi = {v: i for i, v in enumerate(unique)}
            tick_values = list(range(len(unique)))
            tick_text = unique
            values = [stoi[v] for v in values]
        elif isinstance(elem, float):
            fmt = '{{:.{}g}}'
            fmt = fmt.format(precision)
            tick_values = list(sorted(set(values)))
            tick_text = [fmt.format(v) for v in tick_values]
        else:
            tick_values = list(sorted(set(values)))
            tick_text = [str(v) for v in tick_values]

        dct = dict(range=[min(values), max(values)],
                   label=label,
                   values=values,
                   tickvals=tick_values,
                   ticktext=tick_text)

        return dct

    hparams = dicttoolz.merge_with(list, trials)
    dimensions = [to_dimension_dict(hparam) for hparam in hparams.items()]

    if remove_single_values:
        dimensions = list(filter(lambda d: len(d['tickvals']) > 1, dimensions))

    return dimensions
