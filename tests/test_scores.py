import numpy as np

from lensdiff.readouts import Readout, list_positions, load_readouts, save_readouts
from lensdiff.scores import best_per_row, rank_by_score, score
from lensdiff.tokens import TokenGroups, is_latin


def test_token_rules():
    groups = TokenGroups(["whale", "whales", " Whale", ".whale", "box", "boxes", "hates", "hat", "hate", "123", "鲸"])
    assert {groups.group(k) for k in ("whale", "whales", " Whale", ".whale")} == {"whale"}
    assert groups.group("boxes") == "box"
    assert groups.group("hates") == "hate"  # -es only after s, x, z, ch, sh
    assert groups.group("123") is None and groups.group("鲸") is None
    assert is_latin("café") and not is_latin("...")


def test_theta_counts_each_group_once_and_ties_against_the_target():
    keys = ["whale", "whales", "ocean", "boat", "sea"]
    groups = TokenGroups(keys)
    # two layers; positions are 0-based, -1 = not ranked
    positions = np.array([[0, 1, 2, 2, -1],
                          [3, 0, 2, 2, -1]])
    s = score(positions, groups, ["whale"])
    assert s.theta == 1  # whale and whales are one group, at its best member
    assert s.best == 1 and s.median == 1.0
    tie = score(np.array([[1, -1, 1, 0, -1]]), groups, ["ocean"])
    assert tie.theta == 3  # boat first, then ocean tied with whale: ties count against the target
    assert score(positions, groups, ["sea"]).theta is None


def test_median_needs_every_row():
    groups = TokenGroups(["a1x", "b2y"])
    s = score(np.array([[0, 1], [-1, 0]]), groups, ["a1x"])
    assert s.best == 1 and s.median is None
    assert np.isnan(best_per_row(np.array([[0, 1], [-1, 0]]), groups, ["a1x"])[1])


def test_rank_by_score():
    groups = TokenGroups([" Bad", "Magic", "bad", "dog"])
    assert rank_by_score(np.array([0.1, 0.9, 0.5, 0.2]), groups, ["Magic"]) == 1
    assert rank_by_score(np.array([0.1, 0.9, 0.5, 0.2]), groups, [" Bad"]) == 2  # " Bad" and "bad" are one group
    assert rank_by_score(np.ones(4), groups, ["Magic"]) is None  # a constant score ranks nothing


def test_readout_files(tmp_path):
    keys = np.array(["a", "b", "c"])
    r = Readout(np.array([3, 5]), np.array([[0, -1, 1], [2, 0, 1]], dtype=np.int32), np.ones((2, 3), np.int8))
    save_readouts(tmp_path / "m.npz", keys, {"delta_w": r})
    loaded_keys, readouts = load_readouts(tmp_path / "m.npz")
    assert loaded_keys == ["a", "b", "c"]
    assert np.array_equal(readouts["delta_w"].layers(5, 5).positions, [[2, 0, 1]])


def test_list_positions_keeps_each_key_at_its_first_place():
    key_of_token = np.array([0, 1, 1, 2])  # tokens 1 and 2 share key 1
    allowed = np.array([True, True, True, False])
    assert list_positions(np.array([2, 3, 0, 1]), key_of_token, allowed, 3).tolist() == [1, 0, -1]
