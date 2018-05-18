import numpy as np

from ignite.engine import Events

from ..visualization.visdom import create_plot_window


def train_model(model,
                trainer,
                epochs,
                evaluator_train,
                train_loader,
                evaluator_dev=None,
                dev_loader=None,
                lr_scheduler=None,
                early_stopping=None,
                checkpoint=None,
                visdom=None,
                log_interval=100):

    visdom_windows = {}

    if visdom:
        visdom_windows['train_loss'] = create_plot_window(visdom,
                                                          x_label='iteration',
                                                          y_label='loss',
                                                          title='train_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration = (engine.state.iteration - 1) % len(train_loader) + 1
        if iteration % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] train_loss: {:.2f}"
                  "".format(engine.state.epoch, iteration, len(train_loader),
                            engine.state.output))
            if visdom and 'train_loss' in visdom_windows:
                visdom.line(X=np.array([engine.state.iteration]),
                            Y=np.array([engine.state.output]),
                            update='append', win=visdom_windows['train_loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        prefix = 'train'

        evaluator_train.run(train_loader)
        metrics = evaluator_train.state.metrics

        metrics_str = ' '.join('{}_{}: {:.2f}'.format(prefix, name, value)
                               for name, value in metrics.items())

        print("Train - Epoch: {}  {}".format(engine.state.epoch, metrics_str))

        if visdom:
            for name, value in metrics.items():
                metric_name = '{}_{}'.format(prefix, name)

                if metric_name not in visdom_windows:
                    visdom_windows[metric_name] = create_plot_window(visdom,
                                                                     x_label='epoch',
                                                                     y_label=name,
                                                                     title=metric_name)
                visdom.line(X=np.array([engine.state.epoch]),
                            Y=np.array([value]),
                            update='append', win=visdom_windows[metric_name])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        prefix = 'dev'

        evaluator_dev.run(dev_loader)
        metrics = evaluator_dev.state.metrics

        metrics_str = ' '.join('{}_{}: {:.2f}'.format(prefix, name, value)
                               for name, value in metrics.items())

        print("Dev - Epoch: {}  {}".format(engine.state.epoch, metrics_str))

        if visdom:
            for name, value in metrics.items():
                metric_name = '{}_{}'.format(prefix, name)

                if metric_name not in visdom_windows:
                    visdom_windows[metric_name] = create_plot_window(visdom,
                                                                     x_label='epoch',
                                                                     y_label=name,
                                                                     title=metric_name)
                visdom.line(X=np.array([engine.state.epoch]),
                            Y=np.array([value]),
                            update='append', win=visdom_windows[metric_name])

    if lr_scheduler:
        trainer.add_event_handler(Events.ITERATION_STARTED,
                                  lambda engine: lr_scheduler.step(engine.state.epoch))

    if early_stopping and evaluator_dev:
        evaluator_dev.add_event_handler(Events.COMPLETED, early_stopping)

    if checkpoint and evaluator_dev:
        evaluator_dev.add_event_handler(Events.EPOCH_COMPLETED, checkpoint,
                                        {'model': model})

    trainer.run(train_loader, max_epochs=epochs)
