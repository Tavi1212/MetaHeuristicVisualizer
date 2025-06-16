import pytest
import math
from scripts.utils import (
    parse_vectors_string,
    hamming_distance,
    euclidean_distance,
    manhattan_distance,
    compute_distance
)

# ---------- Test: parse_vectors_string in specific modes ----------
@pytest.mark.parametrize("input_str, mode, expected", [
    ("0101", 'binary', [[0, 1, 0, 1]]),
    ("ffffffdcffffffd000000032", 'hex', [[-36, -48, 50]]),
    ("0000000affffffd7ffffffd9", 'hex', [[10, -41, -39]]),
    ("1a2", 'categorical', [['1', 'a', '2']]),
    ("[1.1, 2.2, -3.3]", 'float', [[1.1, 2.2, -3.3]]),
    ("[[1,2], [3,4]]", 'float', [[1, 2], [3, 4]]),
    ("5.7", 'float', [[5.7]]),
])
def test_parse_vectors_string_modes(input_str, mode, expected):
    assert parse_vectors_string(input_str, mode=mode) == expected

# ---------- Test: parse_vectors_string in auto mode ----------
@pytest.mark.parametrize("input_str, expected", [
    ("0101", [[0, 1, 0, 1]]),
    ("ffffffdcffffffd000000032", [[-36, -48, 50]]),
    ("0000000affffffd7ffffffd9", [[10, -41, -39]]),
    ("1a2", [['1', 'a', '2']]),
    ("[1.1, 2.2, -3.3]", [[1.1, 2.2, -3.3]]),
    ("[[1,2], [3,4]]", [[1, 2], [3, 4]]),
    ("5.7", [[5.7]]),
])
def test_parse_vectors_string_auto(input_str, expected):
    assert parse_vectors_string(input_str, mode='auto') == expected


@pytest.mark.parametrize("v1_str, v2_str, expected, mode", [
    # Hamming-like binary test
    ("010", "011", 1, 'binary'),
    ("000", "000", 0, 'binary'),

    # Euclidean tests
    ("[0,0,0]", "[3,4,0]", 5.0, 'float'),
    ("[1,2]", "[4,6]", 5.0, 'float'),

    # Manhattan tests
    ("[1,2,3]", "[4,5,6]", 9, 'float'),
    ("[0,0,0]", "[1,1,1]", 3, 'float'),

    # Hex-encoded continuous vectors
    ("ffffffdcffffffd000000032", "0000000affffffd7ffffffd9",
     math.sqrt(((-36 - 10) ** 2) + ((-48 - -41) ** 2) + ((50 - -39) ** 2)), 'hex'),
])
def test_distance_from_parsed_strings(v1_str, v2_str, expected, mode):
    v1 = parse_vectors_string(v1_str, mode=mode)[0]
    v2 = parse_vectors_string(v2_str, mode=mode)[0]
    if isinstance(expected, float):
        assert math.isclose(euclidean_distance(v1, v2), expected, rel_tol=1e-6)
    else:
        assert manhattan_distance(v1, v2) == expected

