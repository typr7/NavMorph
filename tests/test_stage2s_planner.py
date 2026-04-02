import importlib.util
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


def _fake_branch_bundle():
    return {
        "baseline_index": 0,
        "candidates": [
            {
                "candidate_index": 0,
                "semantic": 0.95,
                "exec_prob": 0.15,
                "progress": 0.30,
                "blocking": 0.85,
                "uncertainty": 0.40,
                "gmap_index": 5,
            },
            {
                "candidate_index": 1,
                "semantic": 0.70,
                "exec_prob": 0.92,
                "progress": 0.65,
                "blocking": 0.10,
                "uncertainty": 0.08,
                "gmap_index": 8,
            },
            {
                "candidate_index": 2,
                "semantic": 0.50,
                "exec_prob": 0.75,
                "progress": 0.40,
                "blocking": 0.20,
                "uncertainty": 0.10,
                "gmap_index": 9,
            },
        ],
    }


def test_branch_planner_can_override_semantic_top1_when_physics_is_bad():
    planner_mod = _load_stage2s_module("planner")
    planner = planner_mod.Stage2SBranchPlanner(top_k=2, depth=2)
    decision = planner.select(_fake_branch_bundle())
    assert decision["selected_index"] == 1


def test_branch_planner_reports_intervention_metadata():
    planner_mod = _load_stage2s_module("planner")
    planner = planner_mod.Stage2SBranchPlanner(top_k=3, depth=2)
    decision = planner.select(_fake_branch_bundle())
    assert "changed" in decision
    assert "branch_scores" in decision
    assert decision["changed"] is True
    assert decision["selected_gmap_index"] == 8


def test_branch_planner_keeps_semantic_top1_when_physics_is_similar():
    planner_mod = _load_stage2s_module("planner")
    planner = planner_mod.Stage2SBranchPlanner(top_k=2, depth=2)
    branch_bundle = _fake_branch_bundle()
    branch_bundle["candidates"][0].update({
        "exec_prob": 0.88,
        "progress": 0.62,
        "blocking": 0.12,
        "uncertainty": 0.09,
    })
    decision = planner.select(branch_bundle)
    assert decision["selected_index"] == 0
    assert decision["changed"] is False



def _class_has_method(path, class_name, method_name):
    import ast
    module = ast.parse(Path(path).read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(isinstance(item, ast.FunctionDef) and item.name == method_name for item in node.body)
    return False


def test_ss_trainer_references_stage2s_branch_planner():
    trainer_text = Path("vlnce_baselines/ss_trainer_ETP.py").read_text()
    assert "Stage2SBranchPlanner" in trainer_text
    assert "_stage2s_online_eval_enabled" in trainer_text


def test_memory_vft_exposes_event_push_interface():
    assert _class_has_method("utils_p/memory.py", "Memory_vft", "push_event")
    assert _class_has_method("utils_p/memory.py", "Memory_vft", "get_event_count")


def test_stage2s_eval_wrapper_exists():
    assert Path("scripts/stage2s/eval_stage2s_navmorph.sh").exists()
