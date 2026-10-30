import pandas as pd

from src.leakage import group_mae, has_train_sibling


def test_has_train_sibling_uses_gap():
    train = ["dish_1000", "dish_5000"]
    ing = {"dish_1000": [("rice", 1, 1), ("chicken", 1, 1)], "dish_5000": [("salad", 1, 1)],
           "dish_1300": [("rice", 1, 1)],                      # zaman komşusu + alt küme → kardeş
           "dish_1400": [("pasta", 1, 1)],                     # zaman komşusu ama malzeme farklı → değil
           "dish_2000": [("rice", 1, 1)],                      # malzeme iç içe ama 600 sn dışında → değil
           "dish_5600": [("salad", 1, 1), ("tomato", 1, 1)]}   # üst küme, 600 sn içinde → kardeş
    sib = has_train_sibling(["dish_1300", "dish_1400", "dish_2000", "dish_5600"], train, ing, gap=600)
    assert sib == {"dish_1300": True, "dish_1400": False, "dish_2000": False, "dish_5600": True}


def test_group_mae_splits_by_sibling():
    preds = pd.DataFrame({"id": ["a", "b", "c", "d"], "y_true": [100, 200, 300, 400], "y_pred": [110, 220, 300, 450]})
    r = group_mae(preds, {"a": True, "b": True, "c": False, "d": False})
    assert r["kardesli"][0] == 15 and r["kardesli"][3] == 2
    assert r["kardessiz"][0] == 25 and r["kardessiz"][3] == 2
