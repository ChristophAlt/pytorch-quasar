import plotly.graph_objs as go

from toolz import dicttoolz


def hparams_to_parallel_coordinates(hparams, precision=3, remove_single_values=True):
    """
    Converts given hyperparameter configurations into the Plotly parallel coordinates
    format.


    Args:
        hparams (list): Hyperparameter configurations for a sequence of trials.
        A trial is represented by a dict of hyperparameters and corresponding values.
        precision (int, optional): The precision for displaying values of type float.
        remove_single_values (bool, optional): If to remove hyperparameters with a single unique value.
    """

    #TODO: If floats are below 1., switch to exponential notation
    #TODO: If displaying floats as strings, sort them numerically
    
    def format_values(values):
        if not values:
            return values
        
        elem = values[0]
        if isinstance(elem, float):
            fmt = '{{:.{}f}}'.format(precision)
            values = [float(fmt.format(value)) for value in values]
        elif isinstance(elem, str):
            values = list(range(len(set(values))))
        return values
    
    def to_dimension_dict(hparam):
        label, values = hparam
        formated_values = format_values(values)
        dct = dict(range=[min(formated_values), max(formated_values)],
                   label=label,
                   values=formated_values,
                   tickvals=list(sorted(set(formated_values))))
        
        if values and isinstance(values[0], str):
            stoi = {v: i for i, v in enumerate(sorted(set(values)))}
            dct['values'] = [stoi[value] for value in values]
            dct['ticktext'] = [v for v, i in sorted(stoi.items(), key=lambda x: x[1])]
        return dct

    hparams = dicttoolz.merge_with(list, hparams)
    
    dimensions = [to_dimension_dict(hparam) for hparam in hparams.items()]
    
    if remove_single_values:
        dimensions = list(filter(lambda d: len(d['tickvals']) > 1, dimensions))
    
    return go.Parcoords(line=dict(color = 'blue'), dimensions=dimensions)
