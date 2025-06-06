import ast
import math

def sol_to_vector(sol: str):
    try:
        parsed = ast.literal_eval(sol)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, str):
            return list(parsed)  # e.g. "'ABCD'" → ['A','B','C','D']
        else:
            raise ValueError  # force fallback
    except (ValueError, SyntaxError):
        return list(sol)  # fallback: treat raw string as character vector

def is_binary_vector(vec):
    if not isinstance(vec, list):
        return False
    return all(str(x) in ('0', '1') for x in vec)

def is_permutation_vector(vec):
    if not isinstance(vec, list):
        return False
    if len(vec) != len(set(vec)):
        return False
    if not all(isinstance(x, (int, str)) for x in vec):
        return False
    return True

def are_permutations_of_same_set(vectors):
    if not vectors:
        return True  # empty list is trivially valid

    reference_set = set(vectors[0])
    reference_length = len(vectors[0])

    for vec in vectors[1:]:
        if len(vec) != reference_length:
            return False  # not same length
        if set(vec) != reference_set:
            return False  # not same elements
        if len(set(vec)) != len(vec):
            return False  # duplicate in current vec

    return True

def is_distance_aplicable(distance, encoding):
    if distance == "hamming":
        return True
    elif distance in ("euclidean", "manhattan"):
        return encoding == "binary"
    return False

def is_entropy_applicable(solution_type: str):
    solution_type = solution_type.lower()
    return solution_type in ("binary", "categorical")


def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")

    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(v1, v2)))
    except ValueError as e:
        raise TypeError("Euclidean distance requires numeric vectors") from e




