import numpy as np

from roadvision.evaluation.metrics import ap50_per_class, bootstrap_f1, confusion, error_taxonomy, prf_at
from roadvision.models.ensemble import weighted_boxes_fusion

GT = {"a": (np.array([[10, 10, 50, 50], [100, 100, 150, 140]], float), np.array([0, 3]))}


def test_perfect_prediction():
    P = {"a": (GT["a"][0].copy(), np.array([0.9, 0.8]), np.array([0, 3]))}
    m = prf_at(P, GT, 0.5)
    assert m["f1"] == 1.0 and m["fp"] == 0 and m["fn"] == 0
    assert np.allclose(np.array(ap50_per_class(P, GT))[[0, 3]], 1.0)


def test_wrong_class_is_fp_and_fn():
    P = {"a": (GT["a"][0][:1], np.array([0.9]), np.array([1]))}
    m = prf_at(P, GT, 0.5)
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 2
    M = confusion(P, GT, 0.5)
    assert M[0, 1] == 1 and M[3, 4] == 1
    e = error_taxonomy(P, GT, 0.5, {"a": 200 * 200}, {"small": 0.005, "medium": 0.05})
    assert e["fp_types"] == {"classification": 1}


def test_threshold_filters():
    P = {"a": (GT["a"][0].copy(), np.array([0.9, 0.1]), np.array([0, 3]))}
    assert prf_at(P, GT, 0.5)["fn"] == 1


def test_duplicate_detection():
    b = np.array([[10, 10, 50, 50], [11, 11, 50, 50]], float)
    P = {"a": (b, np.array([0.9, 0.8]), np.array([0, 0]))}
    e = error_taxonomy(P, GT, 0.5, {"a": 40000}, {"small": 0.005, "medium": 0.05})
    assert e["fp_types"].get("duplicate") == 1


def test_bootstrap_runs():
    P = {"a": (GT["a"][0].copy(), np.array([0.9, 0.8]), np.array([0, 3]))}
    r = bootstrap_f1(P, P, GT, 0.5, 0.5, n=50)
    assert r["f1"] == 1.0 and r["delta_f1"] == 0.0


def test_wbf_merges_agreeing_boxes():
    p1 = (np.array([[10, 10, 50, 50]], float), np.array([0.9]), np.array([0]))
    p2 = (np.array([[12, 12, 52, 52]], float), np.array([0.7]), np.array([0]))
    b, s, l = weighted_boxes_fusion([p1, p2], [2, 1], 0.55)
    assert len(b) == 1 and 10 < b[0, 0] < 12 and s[0] > 0.7
    b, s, l = weighted_boxes_fusion([p1, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))], [2, 1])
    assert len(b) == 1 and s[0] < 0.9  # penalised: only one model agrees
