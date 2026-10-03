"""Trains the shared model on FAKE contract files and checks it beats a constant guess."""

import numpy as np
import pandas as pd
import pytest

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.known_amr_table import KnownAmrTable
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.data.unitig_store import UnitigStore
from genome2mic.models.cross_validation import CrossValidation
from genome2mic.models.mic_predictor import MicPredictor
from genome2mic.models.model_artifact import ModelArtifact
from genome2mic.models.train_config import TrainConfig
from tests.fixtures.synthetic_contract_data import SyntheticContractData


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("processed")
    frames = SyntheticContractData(n_per_species=250, seed=1).write(data_dir)
    labels = LabelMatrix.from_labels(frames["labels"])
    known = KnownAmrTable(frames["known_amr"])
    unitigs = UnitigStore.from_processed_dir(data_dir, list(SPECIES_KEYS))
    config = TrainConfig(max_epochs=40, patience=8, batch_size=64, unitig_max_columns=100, hidden_dim=64, known_dim=32, unitig_dim=16)
    runner = CrossValidation(labels, known, unitigs, frames["splits"], config)
    artifact, predictions, history = runner.run()
    return {"frames": frames, "labels": labels, "runner": runner, "artifact": artifact, "predictions": predictions,
            "history": history, "tmp": tmp_path_factory.mktemp("artifact")}


def test_test_split_never_enters_training(trained):
    splits = trained["frames"]["splits"]
    test_ids = set(splits.loc[splits["split"] == "test", "genome_id"])
    train_ids = set(trained["labels"].genome_ids[trained["runner"].train_positions])
    assert not (test_ids & train_ids)
    assert not (set(trained["predictions"]["genome_id"]) & test_ids)


def test_out_of_fold_predictions_beat_constant_guess(trained):
    preds = trained["predictions"]
    exact = preds[(preds["lab_lower"] > 0) & np.isfinite(preds["lab_upper"])]
    lab_log2 = np.log2(exact["lab_upper"])
    model_gap = np.abs(np.log2(exact["pred_mic"]) - lab_log2)
    constant_gap = np.abs(np.ceil(lab_log2.median()) - lab_log2)
    assert (model_gap <= 1).mean() > 0.6
    assert model_gap.mean() < constant_gap.mean() * 0.8


def test_band_covers_about_ninety_percent_out_of_fold(trained):
    preds = trained["predictions"]
    covered = (preds["band_high"] > preds["lab_lower"]) & (preds["band_low"] <= preds["lab_upper"])
    assert 0.85 <= covered.mean() <= 1.0
    assert (preds["band_low"] <= preds["pred_mic"]).all() and (preds["pred_mic"] <= preds["band_high"]).all()


def test_every_drug_predicted_from_one_forward_pass(trained):
    artifact = trained["artifact"]
    predictor = MicPredictor(artifact)
    known = predictor.known_vector({"gene_blakpc_2": 1, "gene_blactx_m": 1, "n_class_beta_lactam": 2}).reshape(1, -1)
    unitigs = predictor.unitig_vector("KPNEU", set())
    out = predictor.predict("KPNEU", known, unitigs)
    assert sorted(out["drug"]) == ["ceftriaxone", "ciprofloxacin", "meropenem"]
    carbapenemase_mero = out.set_index("drug").loc["meropenem", "mu_log2"]
    clean = predictor.predict("KPNEU", predictor.known_vector({}).reshape(1, -1), unitigs)
    assert carbapenemase_mero > clean.set_index("drug").loc["meropenem", "mu_log2"] + 3


def test_artifact_round_trip_gives_same_predictions(trained):
    artifact = trained["artifact"]
    artifact.save(trained["tmp"])
    loaded = ModelArtifact.load(trained["tmp"])
    predictor_a, predictor_b = MicPredictor(artifact), MicPredictor(loaded)
    known = predictor_a.known_vector({"point_gyra_s83l": 1}).reshape(1, -1)
    unitigs = predictor_a.unitig_vector("ECOLI", {"u_000001"})
    a = predictor_a.predict("ECOLI", known, unitigs)["mu_log2"].to_numpy()
    b = predictor_b.predict("ECOLI", known, unitigs)["mu_log2"].to_numpy()
    assert np.allclose(a, b, atol=1e-5)
    assert loaded.drugs_by_species["ECOLI"] == ["ceftriaxone", "ciprofloxacin", "meropenem"]
