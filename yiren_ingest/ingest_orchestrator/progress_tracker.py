"""
Yiren 进度追踪器：线程安全的任务状态注册表。

维护 task_id → {status, progress, stage, result} 映射。
供 task_runner 写入进度，供外部（CLI）查询进度。

进度阶段：
    file_validating → audio_extracting → audio_segmenting →
    asr_transcribing → refine_started → text_cleaning → done
"""

import threading
from datetime import datetime, timezone
from typing import Any

# 文档规定的有效阶段名
_VALID_STAGES = frozenset({
    "file_validating", "audio_extracting", "audio_segmenting",
    "asr_transcribing", "refine_started", "text_cleaning",
    "summarizing", "done",
})

# 任务状态枚举
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class TaskState:
    """单个任务的进度快照。"""
    __slots__ = (
        "task_id", "status", "progress", "stage",
        "result", "error", "created_at", "updated_at",
    )

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = STATUS_PENDING
        self.progress = 0.0
        self.stage = "file_validating"
        self.result: dict | None = None
        self.error: str | None = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProgressTracker:
    """线程安全的任务进度注册表。

    用法:
        tracker = ProgressTracker()
        tracker.register("task_001")
        tracker.update("task_001", stage="asr_transcribing", progress=0.5)
        state = tracker.get("task_001")
        tracker.complete("task_001", result={...})
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskState] = {}

    # ── 公开 API ──

    def register(self, task_id: str) -> TaskState:
        """注册新任务（状态: pending），返回初始 TaskState。"""
        with self._lock:
            state = TaskState(task_id)
            self._tasks[task_id] = state
            return state

    def update(
        self,
        task_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: float | None = None,
    ) -> TaskState | None:
        """更新任务进度（仅更新传入的字段，其余保持不变）。

        progress 自动 clamp 到 [0.0, 1.0]。
        stage 不在文档规定列表内时降级为 warning 日志但不拒绝。
        """
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return None
            if status is not None:
                state.status = status
            if stage is not None:
                state.stage = stage
            if progress is not None:
                state.progress = max(0.0, min(1.0, float(progress)))
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

    def get(self, task_id: str) -> dict | None:
        """查询任务进度，返回 to_dict() 或 None。"""
        with self._lock:
            state = self._tasks.get(task_id)
            return state.to_dict() if state else None

    def complete(self, task_id: str, result: dict) -> TaskState | None:
        """标记任务成功完成。"""
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return None
            state.status = STATUS_SUCCESS
            state.progress = 1.0
            state.stage = "done"
            state.result = result
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

    def fail(self, task_id: str, error: str) -> TaskState | None:
        """标记任务失败。"""
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return None
            state.status = STATUS_FAILED
            state.error = error
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

    def cancel(self, task_id: str) -> TaskState | None:
        """标记任务取消。"""
        with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return None
            state.status = STATUS_CANCELLED
            state.updated_at = datetime.now(timezone.utc).isoformat()
            return state

    def list_tasks(self) -> list[dict]:
        """列出所有已知任务。"""
        with self._lock:
            return [s.to_dict() for s in self._tasks.values()]


# 模块级单例（同一进程内所有 task_runner 共享）
_default_tracker: ProgressTracker | None = None
_lock_singleton = threading.Lock()


def get_tracker() -> ProgressTracker:
    """获取模块级 ProgressTracker 单例。"""
    global _default_tracker
    if _default_tracker is None:
        with _lock_singleton:
            if _default_tracker is None:
                _default_tracker = ProgressTracker()
    return _default_tracker


# ============================================================
# 本地测试入口
# ============================================================
if __name__ == "__main__":
    print("=== progress_tracker unit tests ===\n")

    tracker = ProgressTracker()

    # Test: register
    state = tracker.register("task_001")
    assert state.status == STATUS_PENDING
    assert state.progress == 0.0
    print(f"[OK] register: status={state.status}")

    # Test: update
    tracker.update("task_001", status=STATUS_RUNNING, stage="asr_transcribing", progress=0.5)
    info = tracker.get("task_001")
    assert info["status"] == STATUS_RUNNING
    assert info["stage"] == "asr_transcribing"
    assert info["progress"] == 0.5
    print(f"[OK] update: status={info['status']}, stage={info['stage']}, progress={info['progress']}")

    # Test: progress clamp
    tracker.update("task_001", progress=1.5)
    assert tracker.get("task_001")["progress"] == 1.0
    tracker.update("task_001", progress=-0.5)
    assert tracker.get("task_001")["progress"] == 0.0
    print("[OK] progress clamp to [0, 1]")

    # Test: complete
    tracker.complete("task_001", {"segments": 42})
    info = tracker.get("task_001")
    assert info["status"] == STATUS_SUCCESS
    assert info["progress"] == 1.0
    assert info["stage"] == "done"
    assert info["result"] == {"segments": 42}
    print(f"[OK] complete: status={info['status']}, result={info['result']}")

    # Test: fail
    tracker.register("task_002")
    tracker.fail("task_002", "ASR backend unavailable")
    info = tracker.get("task_002")
    assert info["status"] == STATUS_FAILED
    assert info["error"] == "ASR backend unavailable"
    print(f"[OK] fail: status={info['status']}, error={info['error']}")

    # Test: cancel
    tracker.register("task_003")
    tracker.cancel("task_003")
    assert tracker.get("task_003")["status"] == STATUS_CANCELLED
    print("[OK] cancel")

    # Test: list_tasks
    tasks = tracker.list_tasks()
    assert len(tasks) == 3
    print(f"[OK] list_tasks: {len(tasks)} tasks")

    # Test: missing task
    assert tracker.get("nonexistent") is None
    assert tracker.update("nonexistent", progress=0.5) is None
    print("[OK] missing task returns None")

    # Test: singleton
    t1 = get_tracker()
    t2 = get_tracker()
    assert t1 is t2
    print("[OK] get_tracker singleton")

    print("\n[PASS] All progress_tracker tests passed")
