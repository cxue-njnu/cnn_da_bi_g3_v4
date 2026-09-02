# -*- coding: utf-8 -*-
"""source_adapted/folds_v4.py - V2.2 patient-specific event-LOSO fold loading.

STRICT REUSE: the 117 folds under
    <root>/v2_2_literature_aligned/folds/patient_event_loso/
are read AS-IS. This module never rebuilds folds, never resamples TEST negatives
and never alters train.csv / dev.csv / test.csv, clip_id, case_id, edf_file,
label, held-out event or the patient-specific event-LOSO structure.
"""
import csv
import os

FOLD_SUBPATH = ("v2_2_literature_aligned", "folds", "patient_event_loso")


def fold_root(v22_root):
    return os.path.join(v22_root, *FOLD_SUBPATH)


def enumerate_folds(v22_root):
    """Return the sorted list of fold directory names (expected: 117)."""
    root = fold_root(v22_root)
    if not os.path.isdir(root):
        raise FileNotFoundError("fold root not found: " + root)
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_fold_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            r = dict(row)
            r["label"] = int(float(r["label"]))
            r["clip_id"] = str(r["clip_id"])
            r["case_id"] = str(r["case_id"])
            r["edf_file"] = str(r["edf_file"])
            rows.append(r)
    return rows


def load_fold(v22_root, fold):
    """Load (train, dev, test) record lists for one fold, unmodified."""
    fold_dir = os.path.join(fold_root(v22_root), fold)
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError("fold dir not found: " + fold_dir)
    train = load_fold_csv(os.path.join(fold_dir, "train.csv"))
    dev = load_fold_csv(os.path.join(fold_dir, "dev.csv"))
    test = load_fold_csv(os.path.join(fold_dir, "test.csv"))
    return train, dev, test


def assert_no_overlap(a, b, name_a, name_b):
    sa = set(x["clip_id"] for x in a)
    sb = set(x["clip_id"] for x in b)
    inter = sa & sb
    if inter:
        raise RuntimeError("OVERLAP %s-%s: %d clip_ids (e.g. %s)"
                           % (name_a, name_b, len(inter), sorted(inter)[:3]))
    return 0


def assert_fold_disjoint(train_rows, dev_rows, test_rows):
    n = 0
    n += assert_no_overlap(train_rows, dev_rows, "train", "dev")
    n += assert_no_overlap(train_rows, test_rows, "train", "test")
    n += assert_no_overlap(dev_rows, test_rows, "dev", "test")
    return n
