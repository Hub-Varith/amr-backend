"""FAKE contract files for tests. Planted gene effects let tests check the model learns something."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse

DRUGS = ["ceftriaxone", "ciprofloxacin", "meropenem"]
KNOWN_COLUMNS = ["gene_blactx_m", "gene_blakpc_2", "gene_blashv", "point_gyra_s83l", "point_parc_s80i", "n_class_beta_lactam"]
# Planted log2 effects: column -> per-drug shift in doubling steps.
KNOWN_EFFECTS = {
    "gene_blactx_m": {"ceftriaxone": 6.0},
    "gene_blakpc_2": {"meropenem": 7.0, "ceftriaxone": 4.0},
    "point_gyra_s83l": {"ciprofloxacin": 4.0},
    "point_parc_s80i": {"ciprofloxacin": 2.0},
}
BASELINE_LOG2 = {"KPNEU": {"ceftriaxone": -3.0, "ciprofloxacin": -4.0, "meropenem": -4.0},
                 "ECOLI": {"ceftriaxone": -3.5, "ciprofloxacin": -5.0, "meropenem": -4.5}}
PANEL_FLOOR_LOG2 = {"ceftriaxone": -2.0, "ciprofloxacin": -4.0, "meropenem": -4.0}
PANEL_CEILING_LOG2 = {"ceftriaxone": 5.0, "ciprofloxacin": 2.0, "meropenem": 4.0}
N_UNITIGS = 300
N_HIDDEN_UNITIGS = 4
HIDDEN_UNITIG_EFFECT = 1.5


class SyntheticContractData:
    """Writes labels, known_amr, lineages, splits and unitig files to a directory."""

    def __init__(self, n_per_species: int = 300, n_clusters: int = 12, missing_rate: float = 0.1, seed: int = 0) -> None:
        self.n_per_species = n_per_species
        self.n_clusters = n_clusters
        self.missing_rate = missing_rate
        self.rng = np.random.default_rng(seed)

    def write(self, directory: Path) -> dict[str, pd.DataFrame]:
        directory.mkdir(parents=True, exist_ok=True)
        label_frames, known_frames, lineage_frames, split_frames = [], [], [], []
        for species_key in BASELINE_LOG2:
            genome_ids = np.array([f"{species_key}_{index:04d}" for index in range(self.n_per_species)])
            clusters = self.rng.integers(0, self.n_clusters, self.n_per_species)
            known = self.sample_known(clusters)
            unitigs = self.rng.random((self.n_per_species, N_UNITIGS)) < 0.3
            true_log2 = self.true_log2(species_key, known, unitigs)
            label_frames.append(self.observe(genome_ids, species_key, true_log2))

            known_frame = pd.DataFrame(known, columns=KNOWN_COLUMNS[:-1])
            known_frame["n_class_beta_lactam"] = known_frame[["gene_blactx_m", "gene_blakpc_2", "gene_blashv"]].sum(axis=1)
            known_frame = known_frame.astype(np.int8)
            known_frame.insert(0, "species", species_key)
            known_frame.insert(0, "genome_id", genome_ids)
            known_frames.append(known_frame)

            cluster_names = np.array([f"{species_key}_PP_{cluster}" for cluster in clusters])
            lineage_frames.append(pd.DataFrame({"genome_id": genome_ids, "species": species_key,
                                                "lineage_cluster": cluster_names, "st": "NA", "cluster_method": "poppunk"}))
            split_frames.append(self.splits(genome_ids, species_key, clusters))

            sparse.save_npz(directory / f"unitigs_{species_key}.npz", sparse.csr_matrix(unitigs.astype(np.int8)))
            pd.DataFrame({"genome_id": genome_ids}).to_parquet(directory / f"unitigs_{species_key}_rows.parquet")
            pd.DataFrame({
                "col_index": np.arange(N_UNITIGS),
                "pattern_id": [f"u_{index:06d}" for index in range(N_UNITIGS)],
                "n_unitigs": 1,
                "unitig_sequences": [["ACGT"] for _ in range(N_UNITIGS)],
                "train_frequency": unitigs.mean(axis=0),
            }).to_parquet(directory / f"unitigs_{species_key}_index.parquet")

        frames = {
            "labels": pd.concat(label_frames, ignore_index=True),
            "known_amr": pd.concat(known_frames, ignore_index=True),
            "lineages": pd.concat(lineage_frames, ignore_index=True),
            "splits": pd.concat(split_frames, ignore_index=True),
        }
        for name, frame in frames.items():
            frame.to_parquet(directory / f"{name}.parquet", index=False)
        return frames

    def sample_known(self, clusters: np.ndarray) -> np.ndarray:
        """Gene presence depends on lineage cluster, mimicking real population structure."""
        n_genes = len(KNOWN_COLUMNS) - 1
        cluster_rates = self.rng.beta(0.6, 1.4, size=(self.n_clusters, n_genes))
        return (self.rng.random((len(clusters), n_genes)) < cluster_rates[clusters]).astype(np.int8)

    def true_log2(self, species_key: str, known: np.ndarray, unitigs: np.ndarray) -> np.ndarray:
        truth = np.zeros((len(known), len(DRUGS)))
        for drug_position, drug in enumerate(DRUGS):
            truth[:, drug_position] = BASELINE_LOG2[species_key][drug]
            for column, effects in KNOWN_EFFECTS.items():
                if drug in effects:
                    truth[:, drug_position] += effects[drug] * known[:, KNOWN_COLUMNS.index(column)]
        truth[:, DRUGS.index("ciprofloxacin")] += HIDDEN_UNITIG_EFFECT * unitigs[:, :N_HIDDEN_UNITIGS].any(axis=1)
        return truth + self.rng.normal(0.0, 0.5, size=truth.shape)

    def observe(self, genome_ids: np.ndarray, species_key: str, true_log2: np.ndarray) -> pd.DataFrame:
        """Turn a true value into one contract interval row per genome x drug, with random gaps."""
        rows = []
        for row in range(len(genome_ids)):
            for drug_position, drug in enumerate(DRUGS):
                if self.rng.random() < self.missing_rate:
                    continue
                step = float(np.ceil(true_log2[row, drug_position]))
                if step <= PANEL_FLOOR_LOG2[drug]:
                    lower, upper, censor = 0.0, 2.0 ** PANEL_FLOOR_LOG2[drug], "left"
                    raw = f"<={upper:g}"
                elif step > PANEL_CEILING_LOG2[drug]:
                    lower, upper, censor = 2.0 ** PANEL_CEILING_LOG2[drug], float("inf"), "right"
                    raw = f">{lower:g}"
                else:
                    lower, upper, censor = 2.0 ** (step - 1), 2.0**step, "interval"
                    raw = f"={upper:g}"
                rows.append({
                    "genome_id": genome_ids[row], "biosample": f"SAMN{row:08d}", "species": species_key, "drug": drug,
                    "mic_lower": lower, "mic_upper": upper, "censor": censor, "sir": None, "raw_result": raw,
                    "method": "dilution", "standard": "CLSI", "standard_year": 2020, "source": "BVBRC",
                    "isolation_source": "blood", "country": "FAKE", "year": 2020,
                })
        return pd.DataFrame(rows)

    def splits(self, genome_ids: np.ndarray, species_key: str, clusters: np.ndarray) -> pd.DataFrame:
        """Whole clusters go to test; remaining clusters are dealt round-robin into 5 folds."""
        cluster_order = self.rng.permutation(self.n_clusters)
        test_clusters = set(cluster_order[: max(1, self.n_clusters // 6)].tolist())
        fold_of_cluster = {cluster: fold % 5 for fold, cluster in enumerate(cluster_order) if cluster not in test_clusters}
        split = np.where(np.isin(clusters, list(test_clusters)), "test", "train")
        fold = pd.array([None if cluster in test_clusters else fold_of_cluster[cluster] for cluster in clusters], dtype="Int64")
        return pd.DataFrame({"genome_id": genome_ids, "species": species_key, "split": split, "fold": fold,
                             "external_set": None, "lolo_lineage": None})
