"""YirenIngest：把《异人之下》攻略视频提取为结构化攻略文本。

使用方式：
    from yiren_ingest import ingest_series_material

    result = ingest_series_material(
        series_id="zhangchulan_guide",
        upload_paths=["ep01.mp4", "ep02.mp4"],
        output_dir="data/series/zhangchulan_guide",
    )
"""

from .ingest_orchestrator.orchestrator import (
    generate_transcript_id,
    generate_upload_id,
    ingest_series_material,
    ingest_single_file,
    series_id_to_compact,
    setup_series_dirs,
)
from .ingest_orchestrator.task_runner import (
    query_ingest_progress,
    submit_ingest_task,
)
from .utils import atomic_write_json

__all__ = [
    "ingest_series_material",
    "ingest_single_file",
    "submit_ingest_task",
    "query_ingest_progress",
    "generate_upload_id",
    "generate_transcript_id",
    "series_id_to_compact",
    "setup_series_dirs",
    "atomic_write_json",
]
