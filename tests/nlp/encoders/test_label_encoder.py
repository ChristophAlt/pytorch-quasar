from quasar.nlp.encoders import LabelEncoder


def test_label_encoder():
    input_ = 'label_b'
    sample = ['label_a', 'label_b']
    encoder = LabelEncoder(sample)
    output = encoder.encode(input_)

    assert encoder.vocab_size == 2
    assert len(output) == 1
    assert encoder.decode(output) == input_


def test_label_encoder_sequence():
    input_ = ['label_b', 'label_c']
    sample = ['label_a', 'label_b', 'label_c']
    encoder = LabelEncoder(sample)
    output = encoder.encode(input_)

    assert encoder.vocab_size == 3
    assert len(output) == 2
    assert encoder.decode(output) == input_
