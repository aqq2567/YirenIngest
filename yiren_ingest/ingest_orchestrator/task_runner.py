"""
Yiren 异步任务执行器。

职责：决定"怎么跑"——异步任务生命周期管理。
   - submit_ingest_task()  — 提交后台任务，立即返回 task_id
   - query_ingest_progress() — 查询任务进度

"做什么"由 orchestrator 定义，task_runner 只负责把 orchestrator
包在后台线程里运行，并通过 progress_tracker 对外暴露进度。

对齐 docs/modules/Yiren_ingest.md §3.5 异步任务接口规范。
"""

import os
import threading
import logging
from typing import Callable

from . import progress_tracker
from .orchestrator import (
    ingest_series_material,
    generate_upload_id,
    setup_series_dirs,
)

logger = logging.getLogger(__name__)

# 进度回调签名: 对齐文档 §3.5 阶段名
# file_validating → audio_extracting → audio_segmenting →
# asr_transcribing → refine_started → text_cleaning → done
ProgressCallback = Callable[[str, float | None], None] | None


def submit_ingest_task(
    series_id: str,
    upload_paths: list[str],
    output_dir: str,
    config: dict | None = None,
) -> str:
    """提交异步摄取任务，立即返回 task_id。

    后台线程执行 ingest_series_material()，
    进度通过 query_ingest_progress(task_id) 查询。

    Args:
        series_id:   系列 ID，如 zhangchulan_guide
        upload_paths: 上传文件路径列表（视频/音频）
        output_dir:   输出根目录，推荐 data/series/{series_id}/
        config:       处理配置（同 orchestrator.ingest_series_material）

    Returns:
        task_id: 格式 TASK_{timestamp}_{rand}，用于 query_ingest_progress()
    """
    # 配置默认值由 orchestrator._DEFAULT_CONFIG 统一管理，
    # 这里只透传调用方传入的 config，不重复维护默认值。
    cfg = dict(config or {})

    from .. import utils
    task_id = f"TASK_{utils.utc_timestamp()}_{utils.random_suffix()}"

    tracker = progress_tracker.get_tracker()
    tracker.register(task_id)
    tracker.update(task_id, status=progress_tracker.STATUS_RUNNING)

    def _run():
        try:
            # 构造 progress 回调——每次阶段变更更新 tracker
            def _on_stage_change(stage: str, progress: float | None = None):
                tracker.update(task_id, stage=stage, progress=progress)

            result = ingest_series_material(
                series_id=series_id,
                upload_paths=upload_paths,
                output_dir=output_dir,
                config=cfg,
                on_progress=_on_stage_change,
            )
            tracker.complete(task_id, result)
        except Exception as e:
            logger.error(f"后台任务异常 [{task_id}]: {e}")
            tracker.fail(task_id, str(e))

    thread = threading.Thread(target=_run, name=f"ingest-{task_id}", daemon=True)
    thread.start()

    logger.info(f"异步任务已提交: task_id={task_id}, files={len(upload_paths)}")
    return task_id


def query_ingest_progress(task_id: str) -> dict | None:
    """查询异步任务进度。

    Args:
        task_id: submit_ingest_task() 返回的任务 ID

    Returns:
        None — task_id 不存在
        {
            "task_id": str,
            "status": "pending" | "running" | "success" | "failed" | "cancelled",
            "progress": 0.0~1.0,
            "stage": "file_validating" | "audio_extracting" | ... | "done",
            "result": {...} | null,      # 完成后填充
            "error": str | null,         # 失败时填充
            "created_at": "ISO8601",
            "updated_at": "ISO8601",
        }
    """
    tracker = progress_tracker.get_tracker()
    return tracker.get(task_id)


# ============================================================
# 本地测试入口
# ============================================================
if __name__ == "__main__":
    import tempfile
    import wave
    import struct
    import time
    import shutil as _shutil

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(levelname)s] %(message)s")

    print("=== task_runner unit tests ===\n")

    tmp_dir = tempfile.mkdtemp()
    test_output = os.path.join(tmp_dir, "seriess", "T_TR_001")

    # Create a test WAV
    def _make_wav(path, sec=1):
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
            for _ in range(16000 * sec): wf.writeframes(struct.pack('<h', 0))

    wav1 = os.path.join(tmp_dir, "audio_1.wav")
    _make_wav(wav1, 1)

    # Test 1: submit and query
    print("\n[Test 1] submit_ingest_task + query_ingest_progress...")
    task_id = submit_ingest_task(
        "T_TR_001", [wav1], test_output,
        config={"enable_refine": False, "enable_summary": False},
    )
    assert task_id.startswith("TASK_"), f"task_id format: {task_id}"
    print(f"  task_id: {task_id}")

    # Poll until done
    for _ in range(30):
        info = query_ingest_progress(task_id)
        assert info is not None, "task not found"
        if info["status"] in ("success", "failed"):
            break
        time.sleep(0.5)
    else:
        assert False, "task did not complete in 15s"

    assert info["status"] == "success", f"status={info['status']}, error={info.get('error')}"
    assert info["progress"] == 1.0
    assert info["stage"] == "done"
    assert info["result"] is not None
    print(f"  status={info['status']}, stage={info['stage']}")
    print(f"  result: transcripts={info['result']['transcripts']}")

    # Test 2: list_tasks via tracker
    tracker = progress_tracker.get_tracker()
    tasks = tracker.list_tasks()
    assert len(tasks) >= 1
    print(f"\n[Test 2] list_tasks: {len(tasks)} tasks")

    # Test 3: cancel via tracker
    task_id2 = submit_ingest_task(
        "T_TR_002", [], os.path.join(tmp_dir, "seriess", "T_TR_002"),
    )
    ok = tracker.cancel(task_id2)
    assert ok is not None
    info = query_ingest_progress(task_id2)
    assert info["status"] == progress_tracker.STATUS_CANCELLED
    print(f"\n[Test 3] cancel: status={info['status']}")

    # Cleanup
    _shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\n[PASS] All task_runner tests passed")
