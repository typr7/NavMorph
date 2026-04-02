from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from .logging import load_candidate_set_records


class GroupedCounterfactualDataset:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.records = load_candidate_set_records(self.path)
        self.group_count = len(self.records)

    def __len__(self) -> int:
        return self.group_count

    def build_group_sample(self, index: int) -> Dict:
        record = self.records[index]
        return record.to_dict()
