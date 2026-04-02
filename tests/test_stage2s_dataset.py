import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path



def _load_stage2s_module(module_basename):
    package_name = "vlnce_baselines"
    subpackage_name = "vlnce_baselines.stage2s"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(Path("vlnce_baselines"))]
        sys.modules[package_name] = pkg
    if subpackage_name not in sys.modules:
        subpkg = types.ModuleType(subpackage_name)
        subpkg.__path__ = [str(Path("vlnce_baselines/stage2s"))]
        sys.modules[subpackage_name] = subpkg

    module_name = f"{subpackage_name}.{module_basename}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path("vlnce_baselines/stage2s") / f"{module_basename}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module



def _write_sample_records(path: Path):
    contracts = _load_stage2s_module("contracts")
    logging_mod = _load_stage2s_module("logging")
    record_a = logging_mod.build_candidate_set_record(
        candidate_set_id="ep-1:3",
        latent_state=contracts.StructuredLatentState(history_latent=[0.1, 0.2]),
        candidates=[
            contracts.CandidateRecord(candidate_index=0),
            contracts.CandidateRecord(candidate_index=1),
        ],
        episode_id="ep-1",
        step_id=3,
    )
    record_b = logging_mod.build_candidate_set_record(
        candidate_set_id="ep-2:4",
        latent_state=contracts.StructuredLatentState(history_latent=[0.3, 0.4]),
        candidates=[contracts.CandidateRecord(candidate_index=0)],
        episode_id="ep-2",
        step_id=4,
    )
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(record_a.to_dict()) + "\n")
        f.write(json.dumps(record_b.to_dict()) + "\n")



def test_grouped_dataset_preserves_candidate_set_membership(tmp_path):
    dataset_mod = _load_stage2s_module("dataset")
    path = tmp_path / "records.jsonl.gz"
    _write_sample_records(path)
    dataset = dataset_mod.GroupedCounterfactualDataset(path)
    assert dataset.group_count == 2
    assert dataset.records[0].candidate_set_id == "ep-1:3"
    assert len(dataset.records[0].candidates) == 2



def test_dataset_builds_rollout_targets_without_row_shuffle(tmp_path):
    dataset_mod = _load_stage2s_module("dataset")
    path = tmp_path / "records.jsonl.gz"
    _write_sample_records(path)
    dataset = dataset_mod.GroupedCounterfactualDataset(path)
    sample = dataset.build_group_sample(0)
    assert sample["candidate_set_id"] == "ep-1:3"
    assert len(sample["candidates"]) == 2
    assert sample["candidates"][0]["candidate_index"] == 0



def test_append_candidate_set_record_writes_gzip_jsonl(tmp_path):
    contracts = _load_stage2s_module("contracts")
    logging_mod = _load_stage2s_module("logging")
    path = tmp_path / "append.jsonl.gz"
    record = logging_mod.build_candidate_set_record(
        candidate_set_id="ep-3:5",
        latent_state=contracts.StructuredLatentState(history_latent=[1.0]),
        candidates=[contracts.CandidateRecord(candidate_index=0)],
        episode_id="ep-3",
        step_id=5,
    )
    logging_mod.append_candidate_set_record(path, record)
    with gzip.open(path, "rt") as f:
        row = json.loads(next(f))
    assert row["candidate_set_id"] == "ep-3:5"
    assert row["episode_id"] == "ep-3"
