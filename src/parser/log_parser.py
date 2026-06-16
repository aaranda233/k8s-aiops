"""
Capa 1 — Online Log Parsing con Drain3.

Abstrae tokens dinamicos y produce templates + cluster IDs.
"""

from dataclasses import dataclass

from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig


@dataclass(frozen=True)
class ParsedLog:
    cluster_id: int
    template: str
    raw: str
    namespace: str = ""
    timestamp: float = 0.0
    level: str = ""   # nivel/severidad de origen (ERROR, FATAL, CRITICAL, ...)


class LogParser:
    def __init__(self):
        cfg = TemplateMinerConfig()
        cfg.drain_sim_th = 0.4
        cfg.drain_depth = 4
        cfg.drain_max_children = 100
        cfg.masking_instructions = [
            MaskingInstruction(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?", "IP"),
            MaskingInstruction(r"[0-9a-f]{8,}-[0-9a-f\-]{8,}", "UUID"),
            MaskingInstruction(r"(-[a-z0-9]{5,10}){2,}", "POD_SUFFIX"),
            MaskingInstruction(r"\b\d+\b", "NUM"),
            MaskingInstruction(r":\d+\.\d+[\.\d]*", "VER"),
        ]

        self._miner = TemplateMiner(config=cfg)

    def parse(self, raw: str, namespace: str = "", timestamp: float = 0.0,
              level: str = "") -> ParsedLog:
        result = self._miner.add_log_message(raw)
        return ParsedLog(
            cluster_id=result["cluster_id"],
            template=result["template_mined"],
            raw=raw,
            namespace=namespace,
            timestamp=timestamp,
            level=level,
        )

    @property
    def cluster_count(self) -> int:
        return len(self._miner.drain.clusters)

    def get_templates(self) -> dict[int, str]:
        return {c.cluster_id: c.get_template() for c in self._miner.drain.clusters}
