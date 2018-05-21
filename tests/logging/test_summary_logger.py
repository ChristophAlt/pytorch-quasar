import pytest

from quasar.logging import VisdomSummaryLogger
from ignite.engine import Events


@pytest.fixture
def mocked_visdom(mocker):
    visdom_mock = mocker.patch('visdom.Visdom')
    visdom_instance = visdom_mock.return_value
    visdom_instance.check_connection.return_value = True
    return visdom_mock


def test_visdom_summary_logger_attach_trainer(mocked_visdom, mocker):
    logger = VisdomSummaryLogger()

    engine = mocker.MagicMock()

    logger.attach_trainer(engine)

    calls = [
        mocker.call(Events.ITERATION_COMPLETED, mocker.ANY),
        mocker.call(Events.STARTED, mocker.ANY)
    ]

    engine.add_event_handler.assert_has_calls(calls)
    assert engine.add_event_handler.call_count == 2

    mocked_visdom.assert_called_once()


def test_visdom_summary_logger_attach_evaluator(mocked_visdom, mocker):
    logger = VisdomSummaryLogger()

    engine = mocker.MagicMock()

    trainer = 'trainer'
    phase = 'phase'

    logger.attach_evaluator(engine, trainer, phase)

    engine.add_event_handler.assert_called_once_with(
        Events.EPOCH_COMPLETED, mocker.ANY, trainer, phase)

    mocked_visdom.assert_called_once()
