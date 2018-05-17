from torchnlp.text_encoders.identity_encoder import IdentityEncoder


class LabelEncoder(IdentityEncoder):
    """ Encodes a set of discrete labels.

    Args:
        sample (list of strings): Sample of labels to build dictionary on

    Example:

        >>> encoder = LabelEncoder(['label_a', 'label_b'])
        >>> encoder.encode('label_a')
         0
        [torch.LongTensor of size 1]
        >>> encoder.vocab
        ['label_a', 'label_b']
        >>>
        >>> encoder = LabelEncoder(['token_a', 'token_b', 'token_c'])
        >>> encoder.encode(['token_b', 'token_c'])
         1
         2
        [torch.LongTensor of size 2]
        >>> encoder.vocab
        ['token_a', 'token_b', 'token_c']

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reserved_tokens=[], **kwargs)
