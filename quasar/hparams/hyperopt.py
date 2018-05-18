def hparams_from_trials(trials):
    hparams = []
    for trial in trials.trials:
        hparam_config = trial['result']['hparams']
        hparam_config['loss'] = trial['result']['loss']
        hparams.append(hparam_config)
    
    return hparams
