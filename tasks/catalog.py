"""Small seed curriculum. Variants of a skill never cross split boundaries."""
import ast
import copy
import random

from scripts.prepare_corpus import canonical, digest

VERSION = "python-functions/v1"
SPLITS = {
    "stable_unique": "train", "sum_above": "train",
    "word_counts": "train", "clamp_repair": "train",
    "runs": "development", "chunk_totals": "development",
    "bounded_parentheses": "evaluation", "merge_counts": "evaluation",
}


def structure_hash(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = "name"
        elif isinstance(node, ast.arg):
            node.arg = "arg"
        elif isinstance(node, ast.FunctionDef):
            node.name = "function"
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            node.value = 0
    return digest(ast.dump(tree, include_attributes=False).encode())


def make_task(family, n, seed):
    if family == "stable_unique":
        description = f"Return the first {n} distinct integers from values in order of first appearance. Return all distinct values if there are fewer."
        body = f"    result = []\n    for value in values:\n        if value not in result:\n            result.append(value)\n        if len(result) == {n}:\n            break\n    return result\n"
        inputs = [[], [2, 2, 1, 2], list(range(n + 2)), [3, -1, 3, 0, -1]]
        expected = [list(dict.fromkeys(x))[:n] for x in inputs]
        mutants = [f"    return sorted(set(values))[:{n}]\n", f"    return values[:{n}]\n"]
    elif family == "sum_above":
        description = f"Return the sum of integers in values strictly greater than {n}. Return zero when none qualify."
        body = f"    total = 0\n    for value in values:\n        if value > {n}:\n            total += value\n    return total\n"
        inputs = [[], [n, n - 1], [n, n + 1, n + 2, -3], [n + 2, n + 2]]
        expected = [sum(filter(lambda x: x > n, values)) for values in inputs]
        mutants = [f"    return sum(x for x in values if x >= {n})\n", "    return sum(values)\n"]
    elif family == "word_counts":
        description = f"Split the string values on whitespace. Return a dictionary counting case-sensitive words of length at least {n}. Ignore shorter words."
        body = f"    counts = {{}}\n    for word in values.split():\n        if len(word) >= {n}:\n            counts[word] = counts.get(word, 0) + 1\n    return counts\n"
        word = "a" * n
        inputs = ["", f"{word} {word}\t{word.upper()}", " ", f"{word} {'b' * (n + 1)} {'c' * max(0, n - 1)}",
                  f"{'9' * n} {word}! {'é' * n} {word}!"]
        def oracle(text):
            words = text.split()
            return {w: words.count(w) for w in set(words) if len(w) >= n}
        expected = [oracle(x) for x in inputs]
        mutants = [f"    return {{w: 1 for w in values.split() if len(w) >= {n}}}\n", f"    return {{w: values.split().count(w) for w in values.split() if len(w) > {n}}}\n",
                   f"    return {{w: values.split().count(w) for w in values.split() if len(w) >= {n} and w.isalpha()}}\n"]
    elif family == "clamp_repair":
        description = f"Repair solve so each integer in values is clamped to the inclusive interval [-{n}, {n}]. Preserve list length and order, including values already inside the interval."
        body = f"    result = []\n    for value in values:\n        if value < -{n}:\n            value = -{n}\n        elif value > {n}:\n            value = {n}\n        result.append(value)\n    return result\n"
        inputs = [[], [-n, 0, n], [-n - 1, n + 1, -n * 3], [0, 1, -1]]
        expected = [[sorted((-n, x, n))[1] for x in xs] for xs in inputs]
        mutants = [f"    return [min({n}, x) for x in values]\n", f"    return [max(-{n}, x) for x in values]\n"]
    elif family == "runs":
        description = f"Return [character, count] pairs for consecutive runs in the string values whose length is at least {n}. Preserve run order and character case."
        body = f"    result = []\n    i = 0\n    while i < len(values):\n        j = i + 1\n        while j < len(values) and values[j] == values[i]:\n            j += 1\n        if j - i >= {n}:\n            result.append([values[i], j - i])\n        i = j\n    return result\n"
        inputs = ["", "a" * n, "a" * n + "B" * (n + 1) + "a" * n, "z" * max(0, n - 1)]
        import itertools
        expected = [[[c, len(list(group))] for c, group in itertools.groupby(x)] for x in inputs]
        expected = [[r for r in runs if r[1] >= n] for runs in expected]
        mutants = [f"    return [[c, values.count(c)] for c in dict.fromkeys(values) if values.count(c) >= {n}]\n", "    return []\n"]
    elif family == "chunk_totals":
        description = f"Return cumulative sums after every {n} integers in values. Include the final cumulative sum for a partial final chunk. Empty input returns an empty list."
        body = f"    result = []\n    total = 0\n    for i, value in enumerate(values):\n        total += value\n        if (i + 1) % {n} == 0:\n            result.append(total)\n    if len(values) % {n}:\n        result.append(total)\n    return result\n"
        inputs = [[], [2] * (n + 1), list(range(1, n * 2 + 1)), [-2, 4, -1]]
        expected = [[sum(xs[:i]) for i in range(n, len(xs) + 1, n)] + ([sum(xs)] if len(xs) % n else []) for xs in inputs]
        mutants = [f"    return [sum(values[i:i+{n}]) for i in range(0, len(values), {n})]\n", "    return [sum(values)]\n"]
    elif family == "bounded_parentheses":
        description = f"The string values contains only '(' and ')'. Return whether it is balanced and its nesting depth never exceeds {n}. Empty input is valid."
        body = f"    depth = 0\n    for char in values:\n        depth += 1 if char == '(' else -1\n        if depth < 0 or depth > {n}:\n            return False\n    return depth == 0\n"
        inputs = ["", ")(", "(" * n + ")" * n, "(" * (n + 1) + ")" * (n + 1), "(()", "()" * (n + 1)]
        def oracle(text):
            levels = [text[:i].count("(") - text[:i].count(")") for i in range(len(text) + 1)]
            return levels[-1] == 0 and min(levels) >= 0 and max(levels) <= n
        expected = [oracle(x) for x in inputs]
        mutants = ["    return values.count('(') == values.count(')')\n", "    return False\n",
                   f"    depth = 0\n    for char in values:\n        depth += 1 if char == '(' else -1\n        if depth < 0:\n            return False\n    return depth == 0 and values.count('(') <= {n}\n"]
    else:
        description = f"Combine a list values of dictionaries with string keys and integer counts. Sum each key across dictionaries, then return only keys whose final total is at least {n}."
        body = f"    totals = {{}}\n    for row in values:\n        for key, count in row.items():\n            totals[key] = totals.get(key, 0) + count\n    return {{key: count for key, count in totals.items() if count >= {n}}}\n"
        inputs = [[], [{"a": n - 1}, {"a": 1}], [{"a": n + 3}, {"a": -4, "b": n}], [{}, {"z": n + 1}]]
        def oracle(rows):
            keys = set().union(*(r.keys() for r in rows))
            sums = {k: sum(row.get(k, 0) for row in rows) for k in keys}
            return {k: v for k, v in sums.items() if v >= n}
        expected = [oracle(x) for x in inputs]
        mutants = [f"    return {{k: v for row in values for k, v in row.items() if v >= {n}}}\n", "    return {}\n"]
    reference = "def solve(values):\n" + body
    starter = "def solve(values):\n    pass\n"
    if family == "clamp_repair":
        starter = "def solve(values):\n" + mutants[0]
    identity = {"generator": VERSION, "family": family, "parameter": n, "seed": seed}
    return {
        "schema": "python-task/v1", "task_id": digest(canonical(identity).encode())[:24],
        "family": family, "split": SPLITS[family], "structure_id": structure_hash(reference),
        "generator": VERSION, "seed": seed, "parameter": n, "description": description,
        "starter": starter, "reference": reference, "mutants": ["def solve(values):\n" + x for x in mutants],
        "cases": [{"args": [copy.deepcopy(x)], "expected": y} for x, y in zip(inputs, expected)],
        "provenance": {"kind": "authored-generator", "path": "tasks/catalog.py", "reference_review": "pending-human-review"},
    }


def generate(variants=2, seed=17):
    if type(variants) is not int or not 1 <= variants <= 16 or type(seed) is not int or seed < 0:
        raise ValueError("variants must be 1..16 and seed must be nonnegative")
    rng = random.Random(seed)
    for family in SPLITS:
        # n=1 makes chunk totals and per-chunk sums equivalent, so use n>=2.
        for n in rng.sample(range(2, 34), variants):
            yield make_task(family, n, seed)
