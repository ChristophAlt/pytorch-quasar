import datetime

from peewee import Model, CharField, TextField, DoubleField, IntegerField, \
    DateTimeField, BooleanField, ForeignKeyField, SqliteDatabase
from playhouse.sqlite_ext import JSONField

from .summary_logger import RunSummaryLogger


db = SqliteDatabase('experiments.db')


class BaseModel(Model):
    class Meta:
        database = db


class Experiment(BaseModel):
    name = CharField(unique=True)
    description = TextField(default='')
    created_date = DateTimeField(default=datetime.datetime.now)


class Trial(BaseModel):
    experiment = ForeignKeyField(Experiment, backref='trials')
    name = CharField()
    description = TextField(default='')
    created_date = DateTimeField(default=datetime.datetime.now)
    finished_date = DateTimeField(null=True)  # default null?
    is_finished = BooleanField(default=False)
    parameters = JSONField(default=None, null=True)
    loss = DoubleField(null=True)
    loss_name = CharField(null=True)


class Metric(BaseModel):
    trial = ForeignKeyField(Trial, backref='metrics')
    name = CharField()
    value = DoubleField()
    step = IntegerField()


class DBRunSummaryLogger(RunSummaryLogger):
    def __init__(self, db, experiment_name):
        super(DBRunSummaryLogger, self).__init__()
        self.db = db
        self.current_trial = None
        with self.db:
            db.create_tables([Experiment, Trial, Metric])
            self.experiment, _ = Experiment.get_or_create(name=experiment_name)

    def on_run_start(self, trainer):
        with self.db:
            self.current_trial = Trial(experiment=self.experiment, name=self.run_name)
            self.current_trial.save()

    def batch_summary(self, engine):
        pass

    def epoch_summary(self, evaluator, trainer, phase):
        metrics = []
        for name, value in evaluator.state.metrics.items():
            metric = '{}_{}'.format(phase, name)
            metrics.append(dict(trial=self.current_trial, name=metric, value=value,
                                step=trainer.state.epoch))

        with self.db:
            Metric.insert_many(metrics).execute()

    def run_summary(self, config, loss):
        with self.db:
            trial = self.current_trial
            trial.parameters = config
            trial.loss = loss
            trial.finished_date = datetime.datetime.now()
            trial.is_finished = True
            trial.save()
