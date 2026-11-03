from unittest.mock import patch
from data_generation.utils.prompts import (
    LENGTH_OPTIONS, STYLE_OPTIONS_SHORT, STYLE_OPTIONS_LONG,
    get_random_generation_params,
)


def test_length_targets_use_compatible_style_prompts():
    for index, length in enumerate(LENGTH_OPTIONS):
        with patch('data_generation.utils.prompts.random.randrange', return_value=index):
            params = get_random_generation_params()
        assert params['length_format'] == length
        assert params['style_instruction'] in (
            STYLE_OPTIONS_SHORT if index < 2 else STYLE_OPTIONS_LONG
        )
