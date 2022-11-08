import random
import string
import pytest
from typing import Dict,Any

from ..src.util.common import rand_color,make_signs,build_log_dict
MAXINT = 2**32
FLOATMAX = float(2**32)
LETTERS = string.ascii_letters + string.digits + string.punctuation

def test_random_color():
    expected = [
        "#c5d714", "#442082", "#1c2e2b", "#7942bd", "#789b34", "#82b70e", "#29f885", "#a54dca", "#74bdc0", "#edbf88",
    ]
    for i in range(10):
        random.seed(i)
        col = rand_color()
        assert col == expected[i]
        assert len(col) == 7
        assert col[0] == "#"
        assert "#" not in col[1:]

    random.seed(random.randint(0,MAXINT))
    col = rand_color()
    assert len(col) == 7
    assert col[0] == "#"
    assert "#" not in col[1:]
    for c in col[1:]:
        assert c.isalpha() or c.isdigit()

def test_make_signs_exception():
    with pytest.raises(AssertionError) as excinfo:
        x = make_signs(0,0)
        assert x == excinfo

def test_make_signs():
    r1 = random.randint(0,2**4)
    r2 = random.randint(0,2**4)
    positives = [1.0] * r1
    negatives = [-1.0] * r2
    assert make_signs(r1,r2) == (positives + negatives)

def test_build_log_dict():
    random.seed(random.randint(0,MAXINT))

    tqdm_dict = {
        "total": random.randint(0,MAXINT),
        "elapsed": random.uniform(0.0,FLOATMAX),
    }
    func = ''.join(random.choice(LETTERS) for i in range(1,100))
    positive = random.randint(0,MAXINT)
    negative = random.randint(0,MAXINT)
    loss = random.uniform(0.0,FLOATMAX)
    
    expected = {
        "Function": func,
        "Iterations": str(tqdm_dict["total"]),
        "Positive": str(positive),
        "Negative": str(negative),
        "Error": str(loss),
        "Duration": str(tqdm_dict["elapsed"]),
        "Iters per Second": str((tqdm_dict["total"])/tqdm_dict["elapsed"]), 
    }

    assert expected == build_log_dict(
        tqdm_dict=tqdm_dict,
        loss=loss,
        func=func,
        positive=positive,
        negative=negative
    )
    