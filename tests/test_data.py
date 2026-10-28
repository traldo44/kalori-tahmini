import json

import pytest
from PIL import Image

from src.config import N5K_DIR
from src.data import (BAD_DISHES, FOOD101_CLASSES, Food101Subset, dish_time, load_n5k_split,
                      make_val_split, read_dish_metadata, time_groups)


def fake_ids(times):
    return [f"dish_{1556572657 + t}" for t in times]


def test_time_groups_split_on_long_gaps():
    groups = time_groups(fake_ids([0, 30, 70, 5000, 5040, 20000]), gap=600)
    assert [len(g) for g in groups] == [3, 2, 1]


def test_val_split_never_breaks_a_group():
    times, t = [], 0
    for _ in range(200):  # 200 grup, her biri 1–8 tabak
        for _ in range(1 + t % 8):
            times.append(t)
            t += 40
        t += 3600
    ids = fake_ids(times)
    train, val = make_val_split(ids)
    assert set(train) | set(val) == set(ids) and not set(train) & set(val)
    assert 0.08 < len(val) / len(ids) < 0.14
    for g in time_groups(ids):
        assert set(g) <= set(train) or set(g) <= set(val)
    assert make_val_split(ids) == (train, val)  # tohum sabit → hep aynı ayrım


def test_food101_subset(tmp_path):
    base = tmp_path / "food-101"
    meta = {c: [f"{c}/{i}" for i in range(2)] for c in FOOD101_CLASSES + ["beignets"]}
    (base / "meta").mkdir(parents=True)
    (base / "meta" / "train.json").write_text(json.dumps(meta))
    for c in FOOD101_CLASSES:
        (base / "images" / c).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (8, 8)).save(base / "images" / c / f"{i}.jpg")
    ds = Food101Subset("train", transform=lambda x: x, root=tmp_path)
    assert len(ds) == 2 * len(FOOD101_CLASSES)          # listede olmayan sınıf alınmadı
    assert ds[0][1] == 0 and ds[len(ds) - 1][1] == len(FOOD101_CLASSES) - 1
    assert len(FOOD101_CLASSES) == len(set(FOOD101_CLASSES)) == 25


needs_n5k = pytest.mark.skipif(not (N5K_DIR / "metadata").exists(), reason="Nutrition5k indirilmemiş")


@needs_n5k
def test_real_metadata():
    meta = read_dish_metadata()
    assert len(meta) > 4700
    d = meta["dish_1561662216"]
    assert d["calories"] == pytest.approx(300.794281) and d["mass"] == 193


@needs_n5k
def test_real_splits_have_no_overlap_and_no_bad_dishes():
    train, test = load_n5k_split("train"), load_n5k_split("test")
    assert not set(train) & set(test)
    assert not set(train) & set(BAD_DISHES)
    assert all(dish_time(d) > 0 for d in train[:10])
