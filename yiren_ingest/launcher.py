"""
Yiren Pipeline Web Launcher — 浏览器 UI 控制台。

通过 Web 界面选择文件夹、配置参数、一键启动流水线并实时监控进度。
内核复用 ingest_series_material()，监控面板复用 PipelineWindow 状态机。

用法:
    python -m yiren_ingest.launcher
    python -m yiren_ingest.launcher --port 8080 --no-browser
"""

import os
import sys
import json
import time
import queue
import threading
import webbrowser
import logging
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from yiren_ingest.utils import load_project_env

load_project_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Yiren.launcher")


# ═══════════════════════════════════════════════════════════════════════════
# 端口工具
# ═══════════════════════════════════════════════════════════════════════════

def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ═══════════════════════════════════════════════════════════════════════════
# LauncherWindow — 配置加载 + 管线生命周期管理
# ═══════════════════════════════════════════════════════════════════════════

class LauncherWindow:
    """管理 Web 启动器状态：配置加载、管线启动/停止、SSE 队列桥接。"""

    def __init__(self):
        self.pipeline_win = None       # PipelineWindow 实例（启动后）
        self.pipeline_thread = None    # 管线后台线程
        self.running = False
        self.current_config = {}       # 最后一次启动使用的配置
        self.start_time = 0.0
        self.file_names: list[str] = []
        self.last_status = "idle"      # idle / running / complete / error / aborted
        self.last_output_dir = ""      # 输出目录（完成后可打开）
        self.last_series_id = ""      # 最后一次启动的 series_id
        self.log_queue: list[dict] = []  # [{ts, level, msg}, ...] 实时日志

    # ── 硬件配置加载（复用 run.py） ──────────────────────────────────
    def load_hardware_config(self) -> dict:
        """返回合并后的硬件配置（含 flat 键映射供前端使用）。"""
        try:
            from yiren_ingest.run import _load_hardware_config, _hw_to_flat_config
            hw = _load_hardware_config()
            flat = _hw_to_flat_config(hw) if hw else {}
            return {"nested": hw, "flat": flat}
        except Exception as e:
            logger.debug(f"硬件配置加载失败: {e}")
            return {"nested": {}, "flat": {}}

    # ── 源文件扫描 ──────────────────────────────────────────────────
    @staticmethod
    def _scan_dir(path: str) -> list[str]:
        """递归扫描目录下所有支持的媒体文件。"""
        from glob import glob
        exts = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm", "*.flv",
                "*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac"]
        results = []
        try:
            for ext in exts:
                results.extend(glob(os.path.join(path, "**", ext),
                                   recursive=True))
        except Exception:
            logger.debug(f"glob 扫描失败: {path}", exc_info=True)
        return sorted(results)

    def preview_paths(self, paths: list[str]) -> dict:
        """扫描路径列表，返回文件计数和样本文件名。"""
        all_files = []
        errors = []
        for p in paths:
            if not os.path.exists(p):
                errors.append({"path": p, "error": "路径不存在"})
                continue
            if os.path.isfile(p):
                all_files.append(p)
            elif os.path.isdir(p):
                all_files.extend(self._scan_dir(p))
            else:
                errors.append({"path": p, "error": "未知路径类型"})

        total_size = 0
        for f in all_files:
            try:
                total_size += os.path.getsize(f)
            except OSError:
                logger.debug(f"获取文件大小失败: {f}")

        return {
            "total_files": len(all_files),
            "total_size_bytes": total_size,
            "sample_files": [os.path.basename(f) for f in all_files[:8]],
            "all_paths": all_files,
            "errors": errors,
        }

    # ── 配置合并 ────────────────────────────────────────────────────
    def _merge_config(self, user_config: dict) -> dict:
        """组装最终配置：_DEFAULT_CONFIG → hardware_config → user_config。"""
        from yiren_ingest.ingest_orchestrator.orchestrator import _DEFAULT_CONFIG
        hw = self.load_hardware_config()
        cfg = dict(_DEFAULT_CONFIG)
        if hw.get("flat"):
            cfg.update(hw["flat"])
        if user_config:
            cfg.update(user_config)
        return cfg

    # ── 管线生命周期 ────────────────────────────────────────────────
    def start(self, series_id: str, source_paths: list[str],
              output_dir: str, config: dict | None = None) -> dict:
        """校验参数、合并配置、启动管线。返回 {ok: bool, errors?: [...]}。

        成功后可通过 sse_queue 属性获取 SSE 事件队列。
        """
        # 校验
        errors = []
        if not series_id or not series_id.strip():
            errors.append({"field": "series_id", "message": "Series ID 不能为空"})
        if not source_paths:
            errors.append({"field": "source_paths", "message": "至少需要一个源文件"})
        if not output_dir or not output_dir.strip():
            errors.append({"field": "output_dir", "message": "输出目录不能为空"})
        # 检查源文件存在
        missing = [p for p in source_paths if not os.path.exists(p)]
        if missing:
            errors.append({"field": "source_paths",
                           "message": f"{len(missing)} 个文件不存在: {missing[:3]}..."})
        if errors:
            return {"ok": False, "errors": errors}

        # 合并配置
        merged_cfg = self._merge_config(config or {})
        self.current_config = merged_cfg

        # 收集文件名（用于 PipelineWindow 初始化）
        self.file_names = [os.path.basename(p) for p in source_paths]

        # 创建 PipelineWindow 并启动
        from yiren_ingest.pipeline_window import PipelineWindow
        from yiren_ingest.ingest_orchestrator.orchestrator import ingest_series_material

        self.pipeline_win = PipelineWindow(self.file_names)
        self.start_time = time.time()

        self.pipeline_thread = self.pipeline_win.start_pipeline(
            ingest_series_material,
            series_id=series_id,
            upload_paths=source_paths,
            output_dir=output_dir,
            config=merged_cfg,
        )
        self.running = True
        self.last_status = "running"
        self.last_output_dir = output_dir
        self.last_series_id = series_id
        self.log_queue.clear()
        self._push_log("INFO", f"管线启动: {series_id}, {len(source_paths)} 个文件")
        logger.info(f"管线已启动: series_id={series_id}, "
                     f"files={len(source_paths)}, "
                     f"asr_workers={merged_cfg.get('asr_workers')}, "
                     f"postprocess_workers={merged_cfg.get('postprocess_workers')}")
        return {"ok": True}

    def _push_log(self, level: str, msg: str):
        """推送日志到 SSE 队列（供前端实时日志面板消费）。"""
        entry = {"ts": time.time(), "level": level, "msg": msg}
        self.log_queue.append(entry)
        # 限制内存占用（最多保留 200 条）
        if len(self.log_queue) > 200:
            self.log_queue = self.log_queue[-150:]
        # 同时写入 SSE 队列
        if self.pipeline_win and self.pipeline_win.sse_queue:
            try:
                self.pipeline_win.sse_queue.put_nowait(("log", entry))
            except Exception:
                logger.debug("SSE 队列写入失败", exc_info=True)

    def abort(self):
        """请求中止管线。"""
        if self.pipeline_win:
            self.pipeline_win.stop()
        self.running = False
        self.last_status = "aborted"
        self._push_log("WARN", "管线已被用户中止")

    def is_done(self) -> bool:
        """管线是否已完成。线程结束后释放 running，允许启动下一次任务。"""
        if self.pipeline_thread is None:
            return True
        done = not self.pipeline_thread.is_alive()
        if done:
            if self.last_status == "running":
                # 线程结束但状态未更新：检查结果
                if self.pipeline_win and hasattr(self.pipeline_win, 'result'):
                    r = self.pipeline_win.result
                    if isinstance(r, dict) and r.get("status") == "error":
                        self.last_status = "error"
                    else:
                        self.last_status = "complete"
                else:
                    self.last_status = "complete"
            self.running = False
        return done

    @property
    def sse_queue(self):
        """暴露 PipelineWindow 的 SSE 队列给 HTTP handler。"""
        if self.pipeline_win:
            return self.pipeline_win.sse_queue
        return None

    def init_payload(self) -> dict:
        """SSE init 事件数据。"""
        return {"files": self.file_names}

    def snapshot_payload(self) -> dict:
        """当前状态快照（SSE tick 回落用）。"""
        if self.pipeline_win:
            return self.pipeline_win.snapshot_payload()
        return {
            "n_total": len(self.file_names),
            "phase_label": "Waiting...",
            "overall_pct": 0.0,
            "tasks": [],
        }

    def elapsed_sec(self) -> float:
        """从启动到现在的秒数。"""
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time


# ═══════════════════════════════════════════════════════════════════════════
# HTTP Handler
# ═══════════════════════════════════════════════════════════════════════════

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yiren 流水线启动器</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS (same as pipeline_window)
   ═══════════════════════════════════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-root:      #FFFFFF;
  --bg-surface:   #F8F9FA;
  --bg-card:      #FFFFFF;
  --border:       #E5E6E8;
  --border-light: #EEEEF0;
  --cyan:         #1A8FBA; --cyan-bg: #1A8FBA12; --cyan-border: #1A8FBA33;
  --amber:        #C4922A; --amber-bg: #C4922A12; --amber-border: #C4922A33;
  --green:        #4A9E3E; --green-bg: #4A9E3E12; --green-border: #4A9E3E33;
  --red:          #D94444; --red-bg:  #D9444412;
  --text-primary:  #1A1A1A; --text-secondary: #5A5D63;
  --text-dim:      #9A9DA3; --text-subtle: #C5C8CC;
  --font-mono: "JetBrains Mono","Cascadia Code","Fira Code","Consolas",monospace;
  --font-sans: system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  --radius-sm: 6px; --radius-md: 8px; --radius-lg: 10px; --radius-pill: 999px;
}
/* 暗色主题 */
[data-theme="dark"]{
  --bg-root:      #1A1A1E;
  --bg-surface:   #222228;
  --bg-card:      #2A2A32;
  --border:       #383840;
  --border-light: #303038;
  --text-primary:  #E8E8EC;
  --text-secondary: #A0A0A8;
  --text-dim:      #686870;
  --text-subtle:   #484850;
}
[data-theme="dark"] .status-bar{background:var(--bg-surface)}
[data-theme="dark"] .mon-header{background:var(--bg-surface)}
[data-theme="dark"] .mon-footer{background:var(--bg-surface)}
[data-theme="dark"] .launcher-footer{background:var(--bg-surface)}
[data-theme="dark"] .btn-browse{background:var(--cyan-bg)}
[data-theme="dark"] .btn-start{color:#fff}
[data-theme="dark"] .badge-wait{background:rgba(255,255,255,0.04);color:var(--text-dim)}
[data-theme="dark"] .bar-track{background:rgba(255,255,255,0.06)}
[data-theme="dark"] .overall-track{background:rgba(255,255,255,0.06)}
html,body{height:100%;overflow:hidden}
body{
  font-family:var(--font-sans);background:var(--bg-root);color:var(--text-secondary);
  display:flex;flex-direction:column;-webkit-font-smoothing:antialiased;
}
/* subtle dot grid */
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:radial-gradient(circle,rgba(0,0,0,0.025) 0.5px,transparent 0.5px);
  background-size:18px 18px;}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS BAR (always visible)
   ═══════════════════════════════════════════════════════════════════════════ */
.status-bar{position:relative;z-index:1;display:flex;align-items:center;justify-content:space-between;
  padding:10px 24px;background:var(--bg-surface);border-bottom:1px solid var(--border);flex-shrink:0;}
.status-logo{font-family:var(--font-mono);font-size:16px;font-weight:600;
  letter-spacing:0.06em;color:var(--text-primary);}
.status-info{font-family:var(--font-mono);font-size:13px;color:var(--text-dim);}
.status-phase{font-family:var(--font-mono);font-size:11px;font-weight:600;
  letter-spacing:0.05em;padding:3px 12px;border-radius:var(--radius-pill);}
.phase-ready{color:var(--text-dim);background:rgba(0,0,0,0.04);}
.phase-running{color:var(--cyan);background:var(--cyan-bg);border:1px solid var(--cyan-border);}
.phase-done{color:var(--green);background:var(--green-bg);border:1px solid var(--green-border);}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN AREA
   ═══════════════════════════════════════════════════════════════════════════ */
.main-area{position:relative;z-index:1;flex:1;overflow:hidden;}

/* ── LAUNCHER SECTION ── */
.launcher-section{display:flex;flex-direction:column;height:100%;
  transition:opacity 0.3s ease,transform 0.3s ease;}
.launcher-section.hiding{opacity:0;transform:translateY(-20px);pointer-events:none;}
.launcher-scroll{flex:1;overflow-y:auto;padding:16px 24px;}
.launcher-scroll::-webkit-scrollbar{width:4px}
.launcher-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── ACCORDION ── */
.accordion{margin-bottom:10px;border:1px solid var(--border);border-radius:var(--radius-md);
  background:var(--bg-card);overflow:hidden;}
.accordion-header{display:flex;align-items:center;gap:10px;padding:12px 16px;
  cursor:pointer;user-select:none;transition:background 0.15s;}
.accordion-header:hover{background:var(--bg-surface);}
.accordion-chevron{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);
  transition:transform 0.2s;width:14px;text-align:center;}
.accordion.open .accordion-chevron{transform:rotate(90deg);}
.accordion-title{font-family:var(--font-mono);font-size:13px;font-weight:600;
  letter-spacing:0.06em;color:var(--text-primary);width:120px;flex-shrink:0;}
.accordion-summary{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);
  flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.accordion-body{display:none;padding:0 16px 14px;border-top:1px solid var(--border-light);}
.accordion.open .accordion-body{display:block;}
.accordion-body .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 20px;padding-top:12px;}
.accordion-body .form-grid.col-1{grid-template-columns:1fr;}

/* ── FORM ELEMENTS ── */
.form-field{display:flex;flex-direction:column;gap:4px;}
.form-label{font-family:var(--font-mono);font-size:11px;font-weight:500;
  letter-spacing:0.04em;color:var(--text-dim);text-transform:uppercase;}
.form-input,.form-select{font-family:var(--font-mono);font-size:13px;
  padding:7px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);
  background:var(--bg-root);color:var(--text-primary);outline:none;
  transition:border-color 0.2s;}
.form-input:focus,.form-select:focus{border-color:var(--cyan);box-shadow:0 0 0 2px var(--cyan-bg);}
.form-input.invalid{border-color:var(--red);}
.form-hint{font-family:var(--font-sans);font-size:11px;color:var(--text-subtle);}
.form-error{font-family:var(--font-sans);font-size:11px;color:var(--red);display:none;}
.form-error.show{display:block;}

/* toggle switch */
.form-toggle-row{display:flex;align-items:center;gap:10px;padding:4px 0;}
.toggle-switch{position:relative;width:40px;height:22px;flex-shrink:0;}
.toggle-switch input{opacity:0;width:0;height:0;}
.toggle-slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,0.12);border-radius:var(--radius-pill);transition:0.25s;}
.toggle-slider::before{content:"";position:absolute;height:16px;width:16px;
  left:3px;bottom:3px;background:#fff;border-radius:50%;transition:0.25s;}
.toggle-switch input:checked+.toggle-slider{background:var(--cyan);}
.toggle-switch input:checked+.toggle-slider::before{transform:translateX(18px);}
.toggle-label{font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);}

/* tag list for source folders */
.tag-list{display:flex;flex-wrap:wrap;gap:6px;min-height:28px;align-items:center;}
.tag-item{display:flex;align-items:center;gap:6px;
  font-family:var(--font-mono);font-size:12px;padding:3px 10px;
  border-radius:var(--radius-pill);background:var(--cyan-bg);border:1px solid var(--cyan-border);
  color:var(--cyan);}
.tag-remove{cursor:pointer;font-size:14px;line-height:1;opacity:0.6;}
.tag-remove:hover{opacity:1;}

/* ── BUTTONS ── */
.btn{font-family:var(--font-mono);font-size:13px;font-weight:600;
  letter-spacing:0.05em;padding:8px 18px;border-radius:var(--radius-sm);
  cursor:pointer;border:1px solid transparent;transition:all 0.2s;outline:none;}
.btn-browse{color:var(--cyan);border-color:var(--cyan-border);background:var(--cyan-bg);}
.btn-browse:hover{background:#1A8FBA22;}
.btn-start{display:block;width:100%;padding:14px;font-size:15px;letter-spacing:0.08em;
  color:#fff;background:var(--green);border:none;border-radius:var(--radius-md);
  box-shadow:0 2px 8px #4A9E3E33;}
.btn-start:hover{background:#3E8A34;}
.btn-start:disabled{background:var(--text-subtle);box-shadow:none;cursor:not-allowed;}
.btn-abort{color:var(--red);border-color:#D9444433;background:var(--red-bg);}
.btn-abort:hover{background:#D9444422;}

/* launcher footer */
.launcher-footer{padding:14px 24px;border-top:1px solid var(--border);
  background:var(--bg-surface);flex-shrink:0;}

/* ── MONITOR SECTION ── */
.monitor-section{display:none;flex-direction:column;height:100%;
  transition:opacity 0.3s ease,transform 0.3s ease;}
.monitor-section.showing{display:flex;opacity:0;transform:translateY(20px);}
.monitor-section.visible{display:flex;opacity:1;transform:translateY(0);}

/* monitor header (same as pipeline_window) */
.mon-header{display:flex;align-items:center;justify-content:space-between;
  padding:12px 24px;background:var(--bg-surface);border-bottom:1px solid var(--border);flex-shrink:0;}
.mon-header-left{display:flex;align-items:baseline;gap:14px;}
.mon-header-right{display:flex;align-items:center;gap:18px;}
.stat-group{display:flex;align-items:center;gap:6px;}
.stat-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.stat-dot.cyan{background:var(--cyan);box-shadow:0 0 6px #1A8FBA44}
.stat-dot.amber{background:var(--amber);box-shadow:0 0 6px #C4922A44}
.stat-dot.green{background:var(--green);box-shadow:0 0 6px #4A9E3E44}
.stat-label{font-family:var(--font-mono);font-size:12px;color:var(--text-dim);letter-spacing:0.04em;}
.stat-value{font-family:var(--font-mono);font-size:16px;font-weight:600;color:var(--text-primary);}
.overall-wrap{display:flex;align-items:center;gap:8px;}
.overall-track{width:140px;height:6px;border-radius:10px;background:rgba(0,0,0,0.06);overflow:hidden;}
.overall-fill{height:100%;border-radius:10px;transition:width 0.4s ease;background:var(--cyan);}
.overall-fill.complete{background:var(--green);}
.overall-pct{font-family:var(--font-mono);font-size:14px;font-weight:500;
  color:var(--text-secondary);width:42px;text-align:right;}

/* monitor task list */
.task-scroll{flex:1;overflow-y:auto;padding:12px 24px;}
.task-scroll::-webkit-scrollbar{width:4px}
.task-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.task-card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);padding:10px 14px;margin-bottom:8px;
  transition:opacity 0.5s ease,transform 0.4s ease,max-height 0.5s ease,padding 0.4s ease,margin 0.4s ease;
  overflow:hidden;max-height:120px;}
.task-card.fading{opacity:0.3;transform:translateX(6px);}
.task-card.removing{opacity:0;max-height:0;padding-top:0;padding-bottom:0;margin-bottom:0;border-width:0;}
.card-top{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.task-index{font-family:var(--font-mono);font-size:13px;font-weight:500;color:var(--text-dim);width:28px;}
.task-name{font-family:var(--font-sans);font-size:15px;font-weight:600;color:var(--text-primary);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.task-name.dim{color:var(--text-dim);font-weight:400;}
.status-badge{font-family:var(--font-mono);font-size:10px;font-weight:600;
  letter-spacing:0.06em;padding:2px 10px;border-radius:var(--radius-pill);flex-shrink:0;}
.badge-wait{color:var(--text-dim);background:rgba(0,0,0,0.04)}
.badge-asr{color:var(--cyan);background:var(--cyan-bg)}
.badge-ref{color:var(--amber);background:var(--amber-bg)}
.badge-done{color:var(--green);background:var(--green-bg)}
.badge-fail{color:var(--red);background:var(--red-bg)}
.card-bars{display:flex;flex-direction:column;gap:5px;}
.bar-row{display:flex;align-items:center;gap:8px;}
.bar-label{font-family:var(--font-mono);font-size:10px;font-weight:500;letter-spacing:0.06em;
  width:28px;text-align:right;color:var(--text-dim);}
.bar-track{flex:1;height:7px;border-radius:10px;background:rgba(0,0,0,0.06);overflow:hidden;}
.bar-fill{height:100%;border-radius:10px;transition:width 0.35s ease;}
.bar-fill.asr{background:var(--cyan);box-shadow:0 0 8px #1A8FBA33}
.bar-fill.ref{background:var(--amber);box-shadow:0 0 8px #C4922A33}
.bar-fill.done{background:var(--green);box-shadow:0 0 6px #4A9E3E22}
.bar-pct{font-family:var(--font-mono);font-size:11px;width:36px;text-align:left;color:var(--text-dim);}
/* sweep animation for opaque ref progress */
.bar-fill.sweep{width:25%!important;animation:sweep 1.4s ease-in-out infinite;border-radius:10px 3px 3px 10px;}
@keyframes sweep{0%{transform:translateX(-100%)}100%{transform:translateX(400%)}}

/* monitor footer */
.mon-footer{display:flex;align-items:center;justify-content:space-between;
  padding:8px 24px;background:var(--bg-surface);border-top:1px solid var(--border);flex-shrink:0;}
.mon-elapsed{font-family:var(--font-mono);font-size:13px;color:var(--text-dim);}
.scan-overlay{position:fixed;inset:0;z-index:100;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.01) 2px,rgba(0,0,0,0.01) 4px);}

/* 条件字段显隐 */
.conditional-group{overflow:hidden;transition:opacity 0.25s,max-height 0.3s;max-height:200px;opacity:1;}
.conditional-group.hidden{max-height:0;opacity:0;pointer-events:none;}

/* 主题切换按钮 */
.theme-btn{font-family:var(--font-mono);font-size:12px;cursor:pointer;
  padding:4px 10px;border-radius:var(--radius-pill);border:1px solid var(--border);
  background:var(--bg-card);color:var(--text-secondary);transition:all 0.2s;}
.theme-btn:hover{border-color:var(--cyan-border);color:var(--cyan);}

/* 预设管理 */
.preset-row{display:flex;align-items:center;gap:8px;padding:6px 0;}
.preset-select{font-family:var(--font-mono);font-size:12px;padding:5px 8px;
  border:1px solid var(--border);border-radius:var(--radius-sm);
  background:var(--bg-card);color:var(--text-primary);flex:1;max-width:260px;}
.preset-btn{font-family:var(--font-mono);font-size:11px;font-weight:500;
  padding:4px 10px;border-radius:var(--radius-sm);cursor:pointer;
  border:1px solid var(--border);background:var(--bg-card);color:var(--text-secondary);}
.preset-btn:hover{background:var(--bg-surface);}

/* 完成摘要卡片 */
.summary-card{background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius-md);padding:14px 18px;margin-bottom:14px;}
.summary-title{font-family:var(--font-mono);font-size:14px;font-weight:600;
  color:var(--text-primary);margin-bottom:10px;}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px 16px;}
.summary-item{display:flex;flex-direction:column;gap:2px;}
.summary-key{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;
  letter-spacing:0.05em;color:var(--text-dim);}
.summary-val{font-family:var(--font-mono);font-size:14px;font-weight:600;color:var(--text-primary);}
.summary-val.err{color:var(--red);font-size:12px;font-weight:400;}

/* 任务卡片上的错误信息 */
.card-error{font-family:var(--font-sans);font-size:11px;color:var(--red);
  margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}

/* ETA 标签 */
.eta-label{font-family:var(--font-mono);font-size:11px;color:var(--text-dim);margin-left:8px;}

/* SSE 重连横幅 */
.reconnect-banner{position:fixed;top:0;left:0;right:0;z-index:200;
  padding:10px 20px;text-align:center;font-family:var(--font-mono);font-size:13px;
  background:var(--amber-bg);color:var(--amber);border-bottom:1px solid var(--amber-border);
  display:none;animation:reconnectPulse 1.5s infinite;}
.reconnect-banner.show{display:block;}
@keyframes reconnectPulse{0%,100%{opacity:1}50%{opacity:0.6}}

/* 操作按钮组（完成后显示） */
.action-bar{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;}
.action-btn{font-family:var(--font-mono);font-size:13px;font-weight:500;
  padding:8px 18px;border-radius:var(--radius-sm);cursor:pointer;
  border:1px solid var(--border);background:var(--bg-card);color:var(--text-primary);
  transition:background 0.15s;}
.action-btn:hover{background:var(--bg-surface);}
.action-btn.primary{background:var(--cyan);color:#fff;border-color:var(--cyan);}
.action-btn.primary:hover{background:var(--cyan-dark,#0d7377);}

/* 实时日志面板 */
.log-panel{margin-top:16px;border:1px solid var(--border);border-radius:var(--radius-sm);
  background:var(--bg-root);overflow:hidden;}
.log-panel-header{display:flex;align-items:center;justify-content:space-between;
  padding:8px 14px;background:var(--bg-surface);border-bottom:1px solid var(--border-light);
  font-family:var(--font-mono);font-size:12px;font-weight:600;color:var(--text-secondary);
  cursor:pointer;user-select:none;}
.log-panel-header:hover{background:var(--border-light);}
.log-panel-body{max-height:200px;overflow-y:auto;padding:6px 0;font-family:var(--font-mono);font-size:11px;}
.log-line{padding:3px 14px;white-space:pre-wrap;word-break:break-all;
  border-bottom:1px solid var(--border-light);}
.log-line:last-child{border-bottom:none;}
.log-line .log-ts{color:var(--text-dim);margin-right:8px;}
.log-line.warn{color:var(--amber);}
.log-line.error{color:var(--red);}

/* Preset 导入导出按钮 */
.preset-io{display:flex;gap:6px;margin-left:6px;}
.preset-io-btn{font-family:var(--font-mono);font-size:10px;padding:3px 8px;
  border-radius:var(--radius-sm);cursor:pointer;border:1px solid var(--border);
  background:var(--bg-card);color:var(--text-dim);}
.preset-io-btn:hover{color:var(--text-primary);border-color:var(--text-dim);}

/* 警告弹窗 */
.warn-overlay{position:fixed;inset:0;z-index:300;background:rgba(0,0,0,0.4);
  display:flex;align-items:center;justify-content:center;}
.warn-dialog{background:var(--bg-card);border:1px solid var(--amber-border);
  border-radius:var(--radius-md);padding:24px 28px;max-width:480px;width:90%;
  box-shadow:0 8px 32px rgba(0,0,0,0.18);}
.warn-dialog h3{font-family:var(--font-mono);font-size:15px;color:var(--amber);margin:0 0 12px;}
.warn-dialog p{font-family:var(--font-sans);font-size:13px;color:var(--text-secondary);margin:0 0 16px;line-height:1.6;}
.warn-dialog .warn-buttons{display:flex;gap:10px;justify-content:flex-end;}
.warn-dialog button{font-family:var(--font-mono);font-size:12px;font-weight:500;
  padding:7px 16px;border-radius:var(--radius-sm);cursor:pointer;border:1px solid var(--border);}
.warn-dialog .warn-cancel{background:var(--bg-card);color:var(--text-secondary);}
.warn-dialog .warn-proceed{background:var(--amber);color:#fff;border-color:var(--amber);}

</style>
</head>
<body>

<!-- ═══════════ STATUS BAR ═══════════ -->
<div class="status-bar">
  <span class="status-logo">YIREN PIPELINE</span>
  <span class="status-info" id="statusInfo">就绪</span>
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="status-phase phase-ready" id="statusPhase">配置中</span>
    <button class="theme-btn" id="themeBtn" onclick="toggleTheme()" title="切换亮色/暗色模式">🌙</button>
  </div>
</div>

<!-- ═══════════ MAIN AREA ═══════════ -->
<div class="main-area" id="mainArea">

<!-- ── LAUNCHER ── -->
<div class="launcher-section" id="launcherSection">
<div class="launcher-scroll">

  <!-- BASIC accordion -->
  <div class="accordion open" id="accBasic">
    <div class="accordion-header" data-section="basic">
      <span class="accordion-chevron">▶</span>
      <span class="accordion-title">基本设置</span>
      <span class="accordion-summary" id="summaryBasic"></span>
    </div>
    <div class="accordion-body">
      <div class="form-grid">
        <div class="form-field">
          <span class="form-label">系列 ID</span>
          <input class="form-input" id="fldSeriesId" placeholder="zhangchulan_guide" autocomplete="off">
          <span class="form-error" id="errSeriesId"></span>
        </div>
        <div class="form-field">
          <span class="form-label">输出目录</span>
          <div style="display:flex;gap:6px;">
            <input class="form-input" id="fldOutputDir" placeholder="data/series/zhangchulan_guide" style="flex:1;" autocomplete="off">
            <button class="btn btn-browse" onclick="browseNativeFolder(p=>{setField('fldOutputDir',p)})">浏览</button>
          </div>
          <span class="form-error" id="errOutputDir"></span>
        </div>
      </div>
      <div class="form-grid" style="margin-top:10px;">
        <div class="form-field">
          <span class="form-label">源文件夹 / 文件</span>
          <div style="display:flex;gap:6px;align-items:center;">
            <div class="tag-list" id="tagList"></div>
            <button class="btn btn-browse" onclick="browseNativeFolder(addSourceFolder)" style="flex-shrink:0;">+ 添加</button>
          </div>
          <span class="form-error" id="errSourcePaths"></span>
          <span class="form-hint" id="hintSource">未选择文件</span>
        </div>
      </div>
      <div class="form-toggle-row" style="margin-top:8px;">
        <label class="toggle-switch"><input type="checkbox" id="fldForceReprocess"><span class="toggle-slider"></span></label>
        <span class="toggle-label">强制重新处理（忽略幂等缓存）</span>
      </div>
    </div>
  </div>

  <!-- ASR accordion -->
  <div class="accordion" id="accAsr">
    <div class="accordion-header" data-section="asr">
      <span class="accordion-chevron">▶</span>
      <span class="accordion-title">ASR 引擎</span>
      <span class="accordion-summary" id="summaryAsr"></span>
    </div>
    <div class="accordion-body">
      <div class="form-grid">
        <!-- ── 后端选择（始终可见） ── -->
        <div class="form-field">
          <span class="form-label">后端引擎</span>
          <select class="form-select" id="fldAsrBackend" onchange="updateConditionalFields()">
            <option value="funasr" selected>funasr（本地，默认）</option>
            <option value="whisper">whisper（本地）</option>
            <option value="cloud">cloud（云端 API）</option>
          </select>
        </div>

        <!-- ═══════ Whisper 专属 ═══════ -->
        <div class="conditional-group" id="grpWhisper">
          <div class="form-field">
            <span class="form-label">模型规格</span>
            <select class="form-select" id="fldWhisperModel">
              <option value="large-v3">large-v3（最强，~10GB）</option>
              <option value="large-v2">large-v2</option>
              <option value="medium">medium（~5GB）</option>
              <option value="small">small（~2GB）</option>
              <option value="tiny">tiny（~1GB）</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">计算设备</span>
            <select class="form-select" id="fldWhisperDevice" onchange="updateConditionalFields()">
              <option value="auto" selected>auto（自动检测）</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">计算精度</span>
            <select class="form-select" id="fldWhisperCompute">
              <option value="auto" selected>auto（GPU: float16 / CPU: int8）</option>
              <option value="float16">float16（GPU 推荐）</option>
              <option value="int8">int8（CPU 推荐）</option>
              <option value="int8_float16">int8_float16</option>
            </select>
          </div>
          <div class="form-field conditional-group" id="grpWhisperCpu">
            <span class="form-label">CPU 线程数</span>
            <input class="form-input" id="fldWhisperCpuThreads" type="number" min="1" max="64" value="4">
            <span class="form-hint">仅 device=cpu 时生效</span>
          </div>
          <div class="form-field">
            <span class="form-label">Beam Size</span>
            <input class="form-input" id="fldWhisperBeamSize" type="number" min="1" max="10" value="5">
            <span class="form-hint">束搜索宽度，越大越准但越慢</span>
          </div>
          <div class="form-field">
            <span class="form-label">VAD 过滤</span>
            <label class="toggle-switch"><input id="fldWhisperVad" type="checkbox" checked><span class="toggle-track"></span></label>
            <span class="form-hint">过滤静音/噪音片段</span>
          </div>
        </div>

        <!-- ═══════ FunASR 专属 ═══════ -->
        <div class="conditional-group hidden" id="grpFunasr">
          <div class="form-field">
            <span class="form-label">模型</span>
            <select class="form-select" id="fldFunasrModel">
              <option value="paraformer-zh">paraformer-zh（中文）</option>
              <option value="paraformer-en">paraformer-en（英文）</option>
              <option value="paraformer-zh-spk">paraformer-zh-spk（说话人分离）</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">计算设备</span>
            <select class="form-select" id="fldFunasrDevice" onchange="updateConditionalFields()">
              <option value="auto" selected>auto（自动检测）</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </select>
          </div>
          <div class="form-field conditional-group hidden" id="grpFunasrCpu">
            <span class="form-label">CPU 线程数</span>
            <input class="form-input" id="fldFunasrNcpu" type="number" min="1" max="64" value="16">
            <span class="form-hint">仅 device=cpu 时生效</span>
          </div>
          <div class="form-field">
            <span class="form-label">VAD 模型</span>
            <select class="form-select" id="fldFunasrVadModel">
              <option value="fsmn-vad">fsmn-vad</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">标点模型</span>
            <select class="form-select" id="fldFunasrPuncModel">
              <option value="ct-punc">ct-punc</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">启用 VAD</span>
            <label class="toggle-switch"><input id="fldFunasrUseVad" type="checkbox" checked><span class="toggle-track"></span></label>
            <span class="form-hint">语音活动检测</span>
          </div>
        </div>

        <!-- ═══════ Cloud 专属 ═══════ -->
        <div class="conditional-group hidden" id="grpCloud">
          <div class="form-field">
            <span class="form-label">服务商</span>
            <select class="form-select" id="fldCloudProvider">
              <option value="aliyun">aliyun（阿里云百炼）</option>
              <option value="tencent">tencent（腾讯云）</option>
              <option value="azure">azure（微软 Azure）</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">模型</span>
            <select class="form-select" id="fldCloudModel">
              <option value="qwen3-asr-flash" selected>qwen3-asr-flash（推荐，Chat Completions）</option>
            </select>
          </div>
          <div class="form-field">
            <span class="form-label">API Key</span>
            <input class="form-input" id="fldCloudApiKey" type="password" placeholder="sk-... 或留空读环境变量 DASHSCOPE_API_KEY">
            <span class="form-hint">留空则从环境变量读取</span>
          </div>
          <div class="form-field">
            <span class="form-label">Base URL</span>
            <input class="form-input" id="fldCloudBaseUrl" placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1">
            <span class="form-hint">API 端点地址</span>
          </div>
        </div>

        <!-- ── 通用字段 ── -->
        <div class="form-field">
          <span class="form-label">并发数</span>
          <input class="form-input" id="fldAsrWorkers" type="number" min="0" max="8" value="0">
          <span class="form-hint">0=自动（保守:1），GPU 可设 3-4</span>
        </div>
        <div class="form-field">
          <span class="form-label">语言</span>
          <input class="form-input" id="fldLanguage" value="zh" maxlength="8">
        </div>
        <div class="form-field">
          <span class="form-label">片段最大时长（秒）</span>
          <input class="form-input" id="fldSegMaxDur" type="number" min="60" max="7200" value="1800">
        </div>
      </div>
    </div>
  </div>

  <!-- REFINE accordion -->
  <div class="accordion" id="accRefine">
    <div class="accordion-header" data-section="refine">
      <span class="accordion-chevron">▶</span>
      <span class="accordion-title">LLM 精修 / 章节摘要</span>
      <span class="accordion-summary" id="summaryRefine"></span>
    </div>
    <div class="accordion-body">
      <div class="form-toggle-row">
        <label class="toggle-switch"><input type="checkbox" id="fldRefineEnable" checked onchange="updateConditionalFields()"><span class="toggle-slider"></span></label>
        <span class="toggle-label">启用 LLM 文本精修</span>
      </div>
      <div class="conditional-group" id="grpRefineSettings">
      <div class="form-grid">
        <div class="form-field">
          <span class="form-label">模型名称</span>
          <input class="form-input" id="fldRefineModel" value="deepseek-v4-pro">
        </div>
        <div class="form-field">
          <span class="form-label">后处理并发数</span>
          <input class="form-input" id="fldRefineWorkers" type="number" min="0" max="50" value="0">
          <span class="form-hint">精修+摘要共用，0=自动</span>
        </div>
        <div class="form-field">
          <span class="form-label">API 地址</span>
          <input class="form-input" id="fldRefineApiUrl" placeholder="https://api.deepseek.com/v1/chat/completions">
        </div>
        <div class="form-field">
          <span class="form-label">API 密钥</span>
          <input class="form-input" id="fldRefineApiKey" type="password" placeholder="sk-...">
        </div>
        <div class="form-field">
          <span class="form-label">超时（秒）</span>
          <input class="form-input" id="fldRefineTimeout" type="number" min="30" max="600" value="180">
        </div>
        <div class="form-field">
          <span class="form-label">最大输出 Token 数</span>
          <input class="form-input" id="fldRefineMaxTokens" type="number" min="1024" max="131072" value="32768">
        </div>
      </div>
      <div class="form-toggle-row">
        <label class="toggle-switch"><input type="checkbox" id="fldRefineThinking"><span class="toggle-slider"></span></label>
        <span class="toggle-label">启用思考模式（仅 DeepSeek V4）</span>
      </div>
      </div>
      <div class="form-toggle-row" style="margin-top:8px;">
        <label class="toggle-switch"><input type="checkbox" id="fldSummaryEnable" checked onchange="updateConditionalFields()"><span class="toggle-slider"></span></label>
        <span class="toggle-label">启用章节 + 全文总览（第二次 LLM 调用）</span>
      </div>
      <div class="conditional-group" id="grpSummarySettings">
        <div class="form-grid">
          <div class="form-field">
            <span class="form-label">摘要模型</span>
            <input class="form-input" id="fldSummaryModel" placeholder="留空则与精修相同">
          </div>
          <div class="form-field">
            <span class="form-label">摘要超时（秒）</span>
            <input class="form-input" id="fldSummaryTimeout" type="number" min="30" max="600" value="180">
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /launcher-scroll -->

<div class="launcher-footer">
  <div class="preset-row" style="margin-bottom:10px;">
    <span style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim);white-space:nowrap;">预设</span>
    <select class="preset-select" id="presetSelect" onchange="onPresetSelect()">
      <option value="">— 选择预设 —</option>
    </select>
    <button class="preset-btn" onclick="savePreset()" title="保存当前配置">保存</button>
    <button class="preset-btn" onclick="deletePreset()" title="删除选中预设">删除</button>
    <span class="preset-io">
      <button class="preset-io-btn" onclick="exportPresets()" title="导出预设为 JSON 文件">导出</button>
      <button class="preset-io-btn" onclick="importPresets()" title="从 JSON 文件导入预设">导入</button>
    </span>
  </div>
  <button class="btn btn-start" id="btnStart" onclick="onStartClick()">启动流水线</button>
</div>
</div><!-- /launcher-section -->

<!-- ── MONITOR (hidden) ── -->
<div class="reconnect-banner" id="reconnectBanner">⚠ 连接断开，正在重连…</div>

<div class="monitor-section" id="monitorSection">
  <div class="mon-header">
    <div class="mon-header-left">
      <span class="header-logo" style="font-family:var(--font-mono);font-size:16px;font-weight:600;color:var(--text-primary);">YIREN PIPELINE</span>
      <span class="header-count" id="headerCount" style="font-family:var(--font-mono);font-size:14px;color:var(--text-dim);"></span>
      <span class="phase-badge asr" id="phaseLabel" style="font-family:var(--font-mono);font-size:11px;font-weight:600;letter-spacing:0.05em;padding:3px 12px;border-radius:var(--radius-pill);color:var(--cyan);background:var(--cyan-bg);border:1px solid var(--cyan-border);">语音识别</span>
    </div>
    <div class="mon-header-right">
      <div class="stat-group"><span class="stat-dot cyan"></span><span class="stat-label">识别中</span><span class="stat-value" id="statAsr">0</span></div>
      <div class="stat-group"><span class="stat-dot amber"></span><span class="stat-label">精修/摘要</span><span class="stat-value" id="statRef">0</span></div>
      <div class="stat-group"><span class="stat-dot green"></span><span class="stat-label">已完成</span><span class="stat-value" id="statDone">0</span></div>
      <div class="overall-wrap"><div class="overall-track"><div class="overall-fill" id="overallBar" style="width:0%"></div></div><span class="overall-pct" id="overallPct">0%</span></div>
    </div>
  </div>
  <div class="task-scroll" id="taskList">
    <div class="summary-card" id="summaryCard" style="display:none;"></div>
  </div>
  <div class="mon-footer">
    <span class="mon-elapsed" id="elapsedLabel">0s</span>
    <span class="eta-label" id="etaLabel"></span>
    <span class="mon-elapsed" id="statTotalLabel" style="margin-left:auto;"></span>
    <div class="action-bar" id="actionBar" style="display:none;">
      <button class="action-btn primary" onclick="resetToForm()">＋ 新任务</button>
      <button class="action-btn" onclick="openOutputDir()">📂 打开输出目录</button>
    </div>
  </div>
  <div class="log-panel" id="logPanel" style="display:none;">
    <div class="log-panel-header" onclick="toggleLogPanel()">
      <span>📋 运行日志 <span id="logCount">0</span> 条</span>
      <span id="logToggleIcon">▼</span>
    </div>
    <div class="log-panel-body" id="logBody"></div>
  </div>
  <div class="scan-overlay"></div>
</div><!-- /monitor-section -->

</div><!-- /main-area -->


<script>
// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
const S = {
  hwDefaults: {},
  sourceFolders: [],   // [{path, count}]
  previewCache: null,
  dirCallback: null,   // fn(path) called on modal confirm
  dirPath: '',
  dirSelected: '',
  // pipeline
  tasks: [],
  running: false,
  done: false,
  t0: 0,
};

// ═══════════════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════════════
async function init() {
  console.log('[UI] 页面初始化开始');
  restoreTheme();
  try {
    const r = await fetch('/api/hardware_config');
    const hw = await r.json();
    S.hwDefaults = hw.flat || {};
    console.log('[UI] 硬件配置已加载', Object.keys(S.hwDefaults).length, '个参数');
    applyDefaults();
  } catch(e) { console.warn('[UI] 加载硬件配置失败', e); }
  restoreFormState();
  attachAccordions();
  updateSummaries();
  updateConditionalFields();
  loadPresets();
  attachFormListeners();
  attachKeyboardShortcuts();
  // 检查是否有正在运行的管线（页面刷新恢复）
  checkRunningPipeline();
  console.log('[UI] 页面初始化完成');
}
document.addEventListener('DOMContentLoaded', init);

async function checkRunningPipeline() {
  try {
    var r = await fetch('/api/status');
    var s = await r.json();
    if (s.running && !s.done) {
      console.log('[UI] 检测到正在运行的管线，自动切换到监控面板');
      S.running = true; S.done = false;
      updateStatusBar('运行中', 'phase-running', (s.series_id || '') + ' — ' + (s.file_count || 0) + ' 个文件');
      transitionToMonitor();
      connectSSE();
    }
  } catch(e) { /* ignore */ }
}

function applyDefaults() {
  const d = S.hwDefaults;
  // Whisper 默认（来自 hardware_config）
  setVal('fldWhisperModel', d.asr_model || 'large-v3');
  setVal('fldWhisperDevice', d.asr_device || 'auto');
  setVal('fldWhisperCompute', d.asr_compute_type || 'auto');
  setVal('fldWhisperCpuThreads', d.asr_cpu_threads || 4);
  setVal('fldWhisperBeamSize', 5);
  setVal('fldWhisperVad', true);
  // FunASR 默认（来自 asr_pipeline/config.py FUNASR_DEFAULTS）
  setVal('fldFunasrModel', 'paraformer-zh');
  setVal('fldFunasrDevice', d.asr_device || 'auto');
  setVal('fldFunasrNcpu', 16);
  setVal('fldFunasrVadModel', 'fsmn-vad');
  setVal('fldFunasrPuncModel', 'ct-punc');
  setVal('fldFunasrUseVad', true);
  // Cloud 默认（来自 asr_pipeline/config.py CLOUD_DEFAULTS）
  setVal('fldCloudProvider', 'aliyun');
  setVal('fldCloudModel', 'qwen3-asr-flash');
  setVal('fldCloudApiKey', '');
  setVal('fldCloudBaseUrl', '');
  // 通用
  setVal('fldAsrWorkers', d.asr_workers !== undefined ? d.asr_workers : 0);
  setVal('fldRefineWorkers', d.postprocess_workers !== undefined ? d.postprocess_workers : 0);
  setVal('fldSummaryEnable', true);
  setVal('fldSummaryModel', '');
  setVal('fldSummaryTimeout', 180);
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.type === 'checkbox') el.checked = !!val;
  else el.value = val;
}
// setField 别名供 dir browser 回调使用
const setField = setVal;

// JS 字符串安全转义（用于嵌入 HTML 属性 onclick/ondblclick）
function escJs(s) {
  return String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// ═══════════════════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════════════════
const THEME_KEY = 'm1_theme';
function restoreTheme() {
  const t = localStorage.getItem(THEME_KEY) || 'light';
  document.documentElement.setAttribute('data-theme', t);
  document.getElementById('themeBtn').textContent = t === 'dark' ? '☀' : '🌙';
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  document.getElementById('themeBtn').textContent = next === 'dark' ? '☀' : '🌙';
  localStorage.setItem(THEME_KEY, next);
}

// ═══════════════════════════════════════════════════════════════════════════
// CONDITIONAL FIELD VISIBILITY
// ═══════════════════════════════════════════════════════════════════════════
function updateConditionalFields() {
  var backend = getVal('fldAsrBackend') || 'funasr';
  // ASR 后端条件组
  var grpW = document.getElementById('grpWhisper');
  var grpF = document.getElementById('grpFunasr');
  var grpC = document.getElementById('grpCloud');
  if (grpW) grpW.classList.toggle('hidden', backend !== 'whisper');
  if (grpF) grpF.classList.toggle('hidden', backend !== 'funasr');
  if (grpC) grpC.classList.toggle('hidden', backend !== 'cloud');

  // Whisper CPU 线程数（仅 device=cpu）
  var wcpu = document.getElementById('grpWhisperCpu');
  if (wcpu) wcpu.classList.toggle('hidden', getVal('fldWhisperDevice') !== 'cpu');

  // FunASR CPU 线程数（仅 device=cpu）
  var fcpu = document.getElementById('grpFunasrCpu');
  if (fcpu) fcpu.classList.toggle('hidden', getVal('fldFunasrDevice') !== 'cpu');

  // Refine 精修设置
  var refGrp = document.getElementById('grpRefineSettings');
  if (refGrp) refGrp.classList.toggle('hidden', !getVal('fldRefineEnable'));

  // 章节摘要设置
  var sumGrp = document.getElementById('grpSummarySettings');
  if (sumGrp) sumGrp.classList.toggle('hidden', !getVal('fldSummaryEnable'));
}

// ═══════════════════════════════════════════════════════════════════════════
// FORM STATE PERSISTENCE (localStorage)
// ═══════════════════════════════════════════════════════════════════════════
const FORM_KEY = 'm1_form_state';
var _formSaveTimer = 0;

function saveFormState() {
  clearTimeout(_formSaveTimer);
  _formSaveTimer = setTimeout(function() {
    var state = {
      seriesId: getVal('fldSeriesId'),
      outputDir: getVal('fldOutputDir'),
      sourceFolders: S.sourceFolders,
      forceReprocess: getVal('fldForceReprocess'),
      asrBackend: getVal('fldAsrBackend'),
      asrWorkers: getVal('fldAsrWorkers'),
      language: getVal('fldLanguage'),
      segMaxDur: getVal('fldSegMaxDur'),
      // Whisper
      whisperModel: getVal('fldWhisperModel'),
      whisperDevice: getVal('fldWhisperDevice'),
      whisperCompute: getVal('fldWhisperCompute'),
      whisperCpuThreads: getVal('fldWhisperCpuThreads'),
      whisperBeamSize: getVal('fldWhisperBeamSize'),
      whisperVad: getVal('fldWhisperVad'),
      // FunASR
      funasrModel: getVal('fldFunasrModel'),
      funasrDevice: getVal('fldFunasrDevice'),
      funasrNcpu: getVal('fldFunasrNcpu'),
      funasrVadModel: getVal('fldFunasrVadModel'),
      funasrPuncModel: getVal('fldFunasrPuncModel'),
      funasrUseVad: getVal('fldFunasrUseVad'),
      // Cloud
      cloudProvider: getVal('fldCloudProvider'),
      cloudModel: getVal('fldCloudModel'),
      cloudApiKey: getVal('fldCloudApiKey'),
      cloudBaseUrl: getVal('fldCloudBaseUrl'),
      refineEnable: getVal('fldRefineEnable'),
      refineModel: getVal('fldRefineModel'),
      refineWorkers: getVal('fldRefineWorkers'),
      refineApiUrl: getVal('fldRefineApiUrl'),
      refineTimeout: getVal('fldRefineTimeout'),
      refineMaxTokens: getVal('fldRefineMaxTokens'),
      refineThinking: getVal('fldRefineThinking'),
      summaryEnable: getVal('fldSummaryEnable'),
      summaryModel: getVal('fldSummaryModel'),
      summaryTimeout: getVal('fldSummaryTimeout'),
    };
    localStorage.setItem(FORM_KEY, JSON.stringify(state));
  }, 200);
}

function restoreFormState() {
  try {
    var raw = localStorage.getItem(FORM_KEY);
    if (!raw) return;
    var s = JSON.parse(raw);
    if (s.seriesId) setVal('fldSeriesId', s.seriesId);
    if (s.outputDir) setVal('fldOutputDir', s.outputDir);
    if (s.sourceFolders && s.sourceFolders.length) {
      S.sourceFolders = s.sourceFolders;
      renderTagList();
      refreshPreview();
    }
    setVal('fldForceReprocess', s.forceReprocess);
    if (s.asrBackend) setVal('fldAsrBackend', s.asrBackend);
    if (s.asrWorkers !== undefined) setVal('fldAsrWorkers', s.asrWorkers);
    if (s.language) setVal('fldLanguage', s.language);
    if (s.segMaxDur) setVal('fldSegMaxDur', s.segMaxDur);
    // Whisper
    if (s.whisperModel) setVal('fldWhisperModel', s.whisperModel);
    if (s.whisperDevice) setVal('fldWhisperDevice', s.whisperDevice);
    if (s.whisperCompute) setVal('fldWhisperCompute', s.whisperCompute);
    if (s.whisperCpuThreads) setVal('fldWhisperCpuThreads', s.whisperCpuThreads);
    if (s.whisperBeamSize) setVal('fldWhisperBeamSize', s.whisperBeamSize);
    setVal('fldWhisperVad', s.whisperVad !== undefined ? s.whisperVad : true);
    // FunASR
    if (s.funasrModel) setVal('fldFunasrModel', s.funasrModel);
    if (s.funasrDevice) setVal('fldFunasrDevice', s.funasrDevice);
    if (s.funasrNcpu) setVal('fldFunasrNcpu', s.funasrNcpu);
    if (s.funasrVadModel) setVal('fldFunasrVadModel', s.funasrVadModel);
    if (s.funasrPuncModel) setVal('fldFunasrPuncModel', s.funasrPuncModel);
    setVal('fldFunasrUseVad', s.funasrUseVad !== undefined ? s.funasrUseVad : true);
    // Cloud
    if (s.cloudProvider) setVal('fldCloudProvider', s.cloudProvider);
    if (s.cloudModel) setVal('fldCloudModel', s.cloudModel);
    if (s.cloudApiKey) setVal('fldCloudApiKey', s.cloudApiKey);
    if (s.cloudBaseUrl) setVal('fldCloudBaseUrl', s.cloudBaseUrl);
    setVal('fldRefineEnable', s.refineEnable !== undefined ? s.refineEnable : true);
    if (s.refineModel) setVal('fldRefineModel', s.refineModel);
    if (s.refineWorkers !== undefined) setVal('fldRefineWorkers', s.refineWorkers);
    if (s.refineApiUrl) setVal('fldRefineApiUrl', s.refineApiUrl);
    if (s.refineTimeout) setVal('fldRefineTimeout', s.refineTimeout);
    if (s.refineMaxTokens) setVal('fldRefineMaxTokens', s.refineMaxTokens);
    setVal('fldRefineThinking', !!s.refineThinking);
    setVal('fldSummaryEnable', s.summaryEnable !== undefined ? s.summaryEnable : true);
    if (s.summaryModel) setVal('fldSummaryModel', s.summaryModel);
    if (s.summaryTimeout) setVal('fldSummaryTimeout', s.summaryTimeout);
    console.log('[UI] 表单状态已恢复');
  } catch(e) { console.warn('[UI] 表单状态恢复失败', e); }
}

function attachFormListeners() {
  var ids = ['fldSeriesId','fldOutputDir','fldForceReprocess',
    'fldAsrBackend','fldAsrWorkers','fldLanguage','fldSegMaxDur',
    'fldWhisperModel','fldWhisperDevice','fldWhisperCompute','fldWhisperCpuThreads','fldWhisperBeamSize','fldWhisperVad',
    'fldFunasrModel','fldFunasrDevice','fldFunasrNcpu','fldFunasrVadModel','fldFunasrPuncModel','fldFunasrUseVad',
    'fldCloudProvider','fldCloudModel','fldCloudApiKey','fldCloudBaseUrl',
    'fldRefineEnable','fldRefineModel','fldRefineWorkers',
    'fldRefineApiUrl','fldRefineTimeout','fldRefineMaxTokens','fldRefineThinking',
    'fldSummaryEnable','fldSummaryModel','fldSummaryTimeout'];
  ids.forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', function() {
      saveFormState();
      updateConditionalFields();
    });
    el.addEventListener('input', function() {
      saveFormState();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════════
function attachKeyboardShortcuts() {
  document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      onStartClick();
    }
    if (e.ctrlKey && e.key === 'o') {
      e.preventDefault();
      browseNativeFolder(addSourceFolder);
    }
    if (e.key === 'Escape' && S.running && !S.done) {
      e.preventDefault();
      fetch('/api/abort', {method:'POST'}).then(function() {
        S.running = false; S.done = true;
      });
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG PRESETS
// ═══════════════════════════════════════════════════════════════════════════
const PRESET_KEY = 'm1_presets';

function loadPresets() {
  var sel = document.getElementById('presetSelect');
  sel.innerHTML = '<option value="">— 选择预设 —</option>';
  try {
    var presets = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]');
    presets.forEach(function(p, i) {
      var opt = document.createElement('option');
      opt.value = String(i);
      opt.textContent = p.name;
      sel.appendChild(opt);
    });
  } catch(e) {}
}

function savePreset() {
  var name = prompt('预设名称:');
  if (!name || !name.trim()) return;
  var values = {};
  var ids = ['fldSeriesId','fldOutputDir','fldForceReprocess',
    'fldAsrBackend','fldAsrWorkers','fldLanguage','fldSegMaxDur',
    'fldWhisperModel','fldWhisperDevice','fldWhisperCompute','fldWhisperCpuThreads','fldWhisperBeamSize','fldWhisperVad',
    'fldFunasrModel','fldFunasrDevice','fldFunasrNcpu','fldFunasrVadModel','fldFunasrPuncModel','fldFunasrUseVad',
    'fldCloudProvider','fldCloudModel','fldCloudApiKey','fldCloudBaseUrl',
    'fldRefineEnable','fldRefineModel','fldRefineWorkers',
    'fldRefineApiUrl','fldRefineTimeout','fldRefineMaxTokens','fldRefineThinking',
    'fldSummaryEnable','fldSummaryModel','fldSummaryTimeout'];
  ids.forEach(function(id) { values[id] = getVal(id); });
  var presets = [];
  try { presets = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch(e) {}
  presets.push({name: name.trim(), values: values});
  localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
  loadPresets();
}

function deletePreset() {
  var sel = document.getElementById('presetSelect');
  if (!sel.value) return;
  if (!confirm('删除预设「' + sel.options[sel.selectedIndex].text + '」？')) return;
  var presets = [];
  try { presets = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch(e) {}
  presets.splice(parseInt(sel.value), 1);
  localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
  loadPresets();
}

function onPresetSelect() {
  var sel = document.getElementById('presetSelect');
  if (!sel.value) return;
  var presets = [];
  try { presets = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch(e) {}
  var p = presets[parseInt(sel.value)];
  if (!p || !p.values) return;
  Object.keys(p.values).forEach(function(id) {
    setVal(id.replace('fld',''), p.values[id]);
  });
  updateConditionalFields();
  updateSummaries();
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETION SUMMARY
// ═══════════════════════════════════════════════════════════════════════════
function renderSummary(data, elapsedSec) {
  var card = document.getElementById('summaryCard');
  var nDone = 0, nFail = 0, failNames = [];
  S.tasks.forEach(function(t) {
    if (t.status === 'FAIL') { nFail++; failNames.push(t.name); }
    else if (t.status === 'DONE') nDone++;
  });
  var mm = Math.floor(elapsedSec / 60), ss = Math.floor(elapsedSec % 60);
  var elapsedStr = mm > 0 ? mm + '分' + ss + '秒' : ss + '秒';

  var html = '<div class="summary-title">' + (data.status === 'error' ? '流水线执行出错' : '流水线执行完成') + '</div>';
  html += '<div class="summary-grid">';
  html += '<div class="summary-item"><span class="summary-key">总耗时</span><span class="summary-val">' + elapsedStr + '</span></div>';
  html += '<div class="summary-item"><span class="summary-key">文件总数</span><span class="summary-val">' + S.tasks.length + '</span></div>';
  html += '<div class="summary-item"><span class="summary-key">成功</span><span class="summary-val">' + nDone + '</span></div>';
  html += '<div class="summary-item"><span class="summary-key">失败</span><span class="summary-val" style="' + (nFail > 0 ? 'color:var(--red)' : '') + '">' + nFail + '</span></div>';
  if (data.message) {
    html += '<div class="summary-item" style="grid-column:1/-1"><span class="summary-key">详情</span><span class="summary-val err">' + escHtml(data.message) + '</span></div>';
  }
  if (failNames.length > 0) {
    html += '<div class="summary-item" style="grid-column:1/-1"><span class="summary-key">失败文件</span><span class="summary-val err">' + failNames.join(', ') + '</span></div>';
  }
  html += '</div>';
  card.innerHTML = html;
  card.style.display = 'block';
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ═══════════════════════════════════════════════════════════════════════════
// ACCORDIONS
// ═══════════════════════════════════════════════════════════════════════════
function attachAccordions() {
  document.querySelectorAll('.accordion-header').forEach(h => {
    h.addEventListener('click', () => {
      const acc = h.parentElement;
      acc.classList.toggle('open');
    });
  });
}

function updateSummaries() {
  var backend = getVal('fldAsrBackend') || 'funasr';
  var asrWorkers = getVal('fldAsrWorkers');
  var asrModel;
  if (backend === 'funasr') asrModel = getVal('fldFunasrModel');
  else if (backend === 'cloud') asrModel = getVal('fldCloudModel');
  else asrModel = getVal('fldWhisperModel');  // auto / whisper

  const refineOn = getVal('fldRefineEnable');
  const refineModel = getVal('fldRefineModel');
  const refineWorkers = getVal('fldRefineWorkers');

  document.getElementById('summaryAsr').textContent =
    backend + ' / ' + asrModel + ' / ' + (asrWorkers||'自动') + ' 路并行';
  const summaryOn = getVal('fldSummaryEnable');
  document.getElementById('summaryRefine').textContent =
    (refineOn ? refineModel + ' / ' + (refineWorkers||'自动') + ' 路并行' : '精修已禁用')
    + (summaryOn ? ' + 章节摘要' : '');
  document.getElementById('summaryBasic').textContent =
    S.sourceFolders.length + ' 个源已选择';
}

function getVal(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (el.type === 'checkbox') return el.checked;
  return el.value;
}

// attach change listeners
document.querySelectorAll('.form-input,.form-select,.toggle-switch input').forEach(el => {
  el.addEventListener('change', updateSummaries);
  el.addEventListener('input', updateSummaries);
});

// ═══════════════════════════════════════════════════════════════════════════
// NATIVE FOLDER PICKER（通过后端 API 调用 Windows 原生对话框）
// ═══════════════════════════════════════════════════════════════════════════
async function browseNativeFolder(callback) {
  console.log('[UI] 打开原生文件夹对话框...');
  try {
    const r = await fetch('/api/browse_folder', {method:'POST'});
    const data = await r.json();
    if (data.path) {
      console.log('[UI] 文件夹已选择:', data.path);
      callback(data.path);
    } else if (data.cancelled) {
      console.log('[UI] 文件夹选择已取消');
    }
  } catch(e) {
    console.error('[UI] 文件夹对话框调用失败:', e);
    alert('无法打开系统文件夹对话框，请手动输入路径。\n\n错误: ' + e.message);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SOURCE FOLDER MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════
function addSourceFolder(path) {
  if (S.sourceFolders.find(f => f.path === path)) return;
  console.log('[UI] 添加源文件夹:', path);
  S.sourceFolders.push({path, count: 0});
  renderTagList();
  refreshPreview();
}

function removeSourceFolder(path) {
  console.log('[UI] 移除源文件夹:', path);
  S.sourceFolders = S.sourceFolders.filter(f => f.path !== path);
  renderTagList();
  refreshPreview();
}

function renderTagList() {
  const container = document.getElementById('tagList');
  container.innerHTML = S.sourceFolders.map(f =>
    `<span class="tag-item">${f.path}<span class="tag-remove" onclick="removeSourceFolder('${escJs(f.path)}')">×</span></span>`
  ).join('');
  updateSummaries();
}

async function refreshPreview() {
  const hint = document.getElementById('hintSource');
  if (S.sourceFolders.length === 0) {
    hint.textContent = '未选择文件';
    return;
  }
  console.log('[UI] 扫描源文件预览...', S.sourceFolders.length, '个路径');
  hint.textContent = '扫描中...';
  try {
    const r = await fetch('/api/preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({paths: S.sourceFolders.map(f => f.path)}),
    });
    const data = await r.json();
    S.previewCache = data;
    console.log('[UI] 扫描结果:', data.total_files, '个文件,', formatBytes(data.total_size_bytes));
    hint.textContent = `${data.total_files} 个文件 (${formatBytes(data.total_size_bytes)})`;
  } catch(e) {
    console.error('[UI] 扫描失败:', e);
    hint.textContent = '扫描失败';
  }
}

function formatBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n/1024).toFixed(1) + ' KB';
  if (n < 1073741824) return (n/1048576).toFixed(1) + ' MB';
  return (n/1073741824).toFixed(2) + ' GB';
}

// ═══════════════════════════════════════════════════════════════════════════
// FORM VALIDATION & SUBMIT
// ═══════════════════════════════════════════════════════════════════════════
function clearErrors() {
  document.querySelectorAll('.form-error').forEach(e => e.classList.remove('show'));
  document.querySelectorAll('.form-input.invalid').forEach(e => e.classList.remove('invalid'));
}

function showError(fieldId, msg) {
  const errEl = document.getElementById('err' + fieldId);
  if (errEl) { errEl.textContent = msg; errEl.classList.add('show'); }
  const inputEl = document.getElementById('fld' + fieldId);
  if (inputEl) inputEl.classList.add('invalid');
}

function collectConfig() {
  const cfg = {};
  // Basic
  cfg.force_reprocess = getVal('fldForceReprocess');
  // ASR
  var backend = getVal('fldAsrBackend') || 'funasr';
  cfg.asr_backend = backend;
  cfg.asr_workers = parseInt(getVal('fldAsrWorkers')) || 0;
  // Whisper 参数
  if (getVal('fldWhisperModel')) cfg.model_size = getVal('fldWhisperModel');
  if (getVal('fldWhisperCompute'))
    cfg.compute_type = getVal('fldWhisperCompute');
  cfg.beam_size = parseInt(getVal('fldWhisperBeamSize')) || 5;
  cfg.vad_filter = getVal('fldWhisperVad');
  // 设备 & CPU 线程数（按后端区分）
  if (backend === 'funasr') {
    if (getVal('fldFunasrDevice')) cfg.asr_device = getVal('fldFunasrDevice');
    cfg.ncpu = parseInt(getVal('fldFunasrNcpu')) || 16;
  } else {
    // whisper / auto / cloud
    if (getVal('fldWhisperDevice'))
      cfg.asr_device = getVal('fldWhisperDevice');
    cfg.cpu_threads = parseInt(getVal('fldWhisperCpuThreads')) || 4;
  }
  // FunASR 专属参数
  if (getVal('fldFunasrVadModel')) cfg.vad_model = getVal('fldFunasrVadModel');
  if (getVal('fldFunasrPuncModel')) cfg.punc_model = getVal('fldFunasrPuncModel');
  cfg.use_vad = getVal('fldFunasrUseVad');
  // Cloud 参数
  if (getVal('fldCloudProvider')) cfg.provider = getVal('fldCloudProvider');
  if (getVal('fldCloudApiKey')) cfg.api_key = getVal('fldCloudApiKey');
  if (getVal('fldCloudBaseUrl')) cfg.base_url = getVal('fldCloudBaseUrl');
  // 模型名（按后端二选一，避免 FunASR 覆盖 Cloud）
  if (backend === 'cloud') {
    if (getVal('fldCloudModel')) cfg.model = getVal('fldCloudModel');
  } else if (backend === 'funasr') {
    if (getVal('fldFunasrModel')) cfg.model = getVal('fldFunasrModel');
  }
  cfg.language = getVal('fldLanguage') || 'zh';
  cfg.segment_max_duration_sec = parseInt(getVal('fldSegMaxDur')) || 1800;
  // Refine
  cfg.enable_refine = getVal('fldRefineEnable');
  cfg.refine_model = getVal('fldRefineModel') || 'deepseek-v4-pro';
  cfg.postprocess_workers = parseInt(getVal('fldRefineWorkers')) || 0;
  cfg.refine_timeout = parseInt(getVal('fldRefineTimeout')) || 180;
  cfg.refine_max_output_tokens = parseInt(getVal('fldRefineMaxTokens')) || 32768;
  cfg.refine_enable_thinking = getVal('fldRefineThinking');
  if (getVal('fldRefineApiUrl')) cfg.refine_api_url = getVal('fldRefineApiUrl');
  if (getVal('fldRefineApiKey')) cfg.refine_api_key = getVal('fldRefineApiKey');
  // Summary
  cfg.enable_summary = getVal('fldSummaryEnable');
  if (getVal('fldSummaryModel')) cfg.summary_model = getVal('fldSummaryModel');
  cfg.summary_timeout = parseInt(getVal('fldSummaryTimeout')) || 180;
  return cfg;
}

async function onStartClick() {
  console.log('[UI] 点击「启动流水线」按钮');
  clearErrors();
  const seriesId = document.getElementById('fldSeriesId').value.trim();
  const outputDir = document.getElementById('fldOutputDir').value.trim();

  let hasErr = false;
  if (!seriesId) { showError('SeriesId', '必填'); hasErr = true; }
  if (!outputDir) { showError('OutputDir', '必填'); hasErr = true; }
  if (S.sourceFolders.length === 0) {
    showError('SourcePaths', '请至少添加一个源文件夹');
    hasErr = true;
  }
  if (hasErr) { console.warn('[UI] 表单校验未通过'); return; }

  // Collect all file paths
  if (!S.previewCache || !S.previewCache.all_paths) {
    await refreshPreview();
  }
  const allPaths = S.previewCache ? (S.previewCache.all_paths || []) : [];
  if (allPaths.length === 0) {
    showError('SourcePaths', '所选文件夹中未找到媒体文件');
    return;
  }

  console.log('[UI] 启动请求: series=%s, files=%d, output=%s', seriesId, allPaths.length, outputDir);
  var config = collectConfig();

  // API Key 前置检查
  var warnings = checkApiKeys();
  if (warnings.length > 0) {
    showApiKeyWarning(warnings, function() { doStart(seriesId, allPaths, outputDir, config); });
    return;
  }
  doStart(seriesId, allPaths, outputDir, config);
}

async function doStart(seriesId, allPaths, outputDir, config) {
  var btn = document.getElementById('btnStart');
  btn.disabled = true;
  btn.textContent = '正在启动...';

  try {
    const r = await fetch('/api/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        series_id: seriesId,
        source_paths: allPaths,
        output_dir: outputDir,
        config: config,
      }),
    });
    const result = await r.json();
    if (!result.ok) {
      console.error('[UI] 启动失败:', result.errors);
      (result.errors || []).forEach(e => {
        if (e.field === 'series_id') showError('SeriesId', e.message);
        else if (e.field === 'output_dir') showError('OutputDir', e.message);
        else if (e.field === 'source_paths') showError('SourcePaths', e.message);
        else alert(e.field + ': ' + e.message);
      });
      btn.disabled = false;
      btn.textContent = '启动流水线';
      return;
    }
    console.log('[UI] 启动成功, 即将切换到监控面板');
  } catch(e) {
    console.error('[UI] 启动请求异常:', e);
    alert('启动失败：' + e.message);
    btn.disabled = false;
    btn.textContent = '启动流水线';
    return;
  }

  // Success — transition to monitor
  S.running = true;
  S.done = false;
  S.t0 = performance.now();
  updateStatusBar('运行中', 'phase-running', `${seriesId} — ${allPaths.length} 个文件`);
  transitionToMonitor();
  connectSSE();
}

function updateStatusBar(text, phaseClass, infoText) {
  document.getElementById('statusInfo').textContent = infoText || text;
  const ph = document.getElementById('statusPhase');
  ph.textContent = text;
  ph.className = 'status-phase ' + (phaseClass || 'phase-ready');
}

// ═══════════════════════════════════════════════════════════════════════════
// TRANSITION
// ═══════════════════════════════════════════════════════════════════════════
function transitionToMonitor() {
  const launch = document.getElementById('launcherSection');
  const monitor = document.getElementById('monitorSection');

  launch.classList.add('hiding');
  setTimeout(() => {
    launch.style.display = 'none';
    monitor.classList.add('showing');
    monitor.style.display = 'flex';
    requestAnimationFrame(() => {
      monitor.classList.remove('showing');
      monitor.classList.add('visible');
    });
  }, 300);
}

// ═══════════════════════════════════════════════════════════════════════════
// MONITOR JS (reuses pipeline_window logic)
// ═══════════════════════════════════════════════════════════════════════════
const BADGE_MAP = {WAIT:['badge-wait','等待'],ASR:['badge-asr','识别中'],
  REFINE:['badge-ref','精修中'],DONE:['badge-done','已完成'],FAIL:['badge-fail','失败']};

function initTasks(names) {
  const list = document.getElementById('taskList');
  list.innerHTML = '';
  S.tasks = [];
  const frag = document.createDocumentFragment();
  names.forEach((name, i) => {
    const t = {name:name.slice(0,56),asrPct:0,refinePct:0,status:'WAIT',
               fading:false,removed:false,_dom:{}};
    S.tasks.push(t);
    const card = document.createElement('div'); card.className = 'task-card';
    const top = document.createElement('div'); top.className = 'card-top';
    const idxEl = document.createElement('span'); idxEl.className = 'task-index'; idxEl.textContent = '#' + (i+1);
    const nameEl = document.createElement('span'); nameEl.className = 'task-name dim'; nameEl.textContent = t.name;
    const badgeEl = document.createElement('span'); badgeEl.className = 'status-badge badge-wait'; badgeEl.textContent = '等待';
    top.append(idxEl, nameEl, badgeEl);
    const bars = document.createElement('div'); bars.className = 'card-bars';
    // ASR row
    const asrRow = document.createElement('div'); asrRow.className = 'bar-row';
    const asrLbl = document.createElement('span'); asrLbl.className = 'bar-label'; asrLbl.textContent = '识别';
    const asrTrack = document.createElement('div'); asrTrack.className = 'bar-track';
    const asrFill = document.createElement('div'); asrFill.className = 'bar-fill asr'; asrFill.style.width = '0%';
    asrTrack.appendChild(asrFill);
    const asrPctEl = document.createElement('span'); asrPctEl.className = 'bar-pct'; asrPctEl.textContent = '0%';
    asrRow.append(asrLbl, asrTrack, asrPctEl);
    // REF row
    const refRow = document.createElement('div'); refRow.className = 'bar-row';
    const refLbl = document.createElement('span'); refLbl.className = 'bar-label'; refLbl.textContent = '后处理';
    const refTrack = document.createElement('div'); refTrack.className = 'bar-track';
    const refFill = document.createElement('div'); refFill.className = 'bar-fill ref'; refFill.style.width = '0%';
    refTrack.appendChild(refFill);
    const refPctEl = document.createElement('span'); refPctEl.className = 'bar-pct'; refPctEl.textContent = '0%';
    refRow.append(refLbl, refTrack, refPctEl);
    bars.append(asrRow, refRow);
    var errEl = document.createElement('div'); errEl.className = 'card-error'; errEl.style.display = 'none';
    card.append(top, bars, errEl);
    t._dom = {card, idxEl, nameEl, badgeEl, asrFill, asrPctEl, refFill, refPctEl, errEl};
    frag.appendChild(card);
  });
  list.appendChild(frag);
  document.getElementById('statTotalLabel').textContent = names.length + ' 个文件';
}

function patchCard(t) {
  const d = t._dom;
  if (!d || t.removed) return;
  let cls = 'task-card';
  if (t.fading) cls += ' fading';
  if (t.removed) cls += ' removing';
  d.card.className = cls;
  const active = t.status !== 'DONE' && t.status !== 'WAIT';
  d.nameEl.className = 'task-name' + (active ? '' : ' dim');
  d.nameEl.textContent = t.name;
  const [bc, bn] = BADGE_MAP[t.status] || ['badge-wait', '?'];
  d.badgeEl.className = 'status-badge ' + bc;
  d.badgeEl.textContent = bn;
  d.asrFill.style.width = (t.asrPct * 100) + '%';
  if (t.status === 'DONE' || (t.status === 'ASR' && t.asrPct >= 1)) {
    d.asrFill.className = 'bar-fill done';
  } else {
    d.asrFill.className = 'bar-fill asr';
  }
  d.asrPctEl.textContent = Math.round(t.asrPct * 100) + '%';
  const rp = t.refinePct;
  if (t.status === 'REFINE' && rp <= 0) {
    d.refFill.className = 'bar-fill ref sweep';
    d.refFill.style.width = '';
    d.refPctEl.textContent = '...';
  } else {
    d.refFill.style.width = (rp * 100) + '%';
    d.refPctEl.textContent = Math.round(rp * 100) + '%';
    d.refFill.className = t.status === 'DONE' ? 'bar-fill done' : 'bar-fill ref';
  }
  // 错误信息
  if (d.errEl) {
    if (t.status === 'FAIL' && t.error) {
      d.errEl.textContent = t.error; d.errEl.style.display = '';
    } else {
      d.errEl.style.display = 'none';
    }
  }
}

function reindexCards() {
  let vi = 0;
  S.tasks.forEach(t => { if (!t.removed) { vi++; t._dom.idxEl.textContent = '#' + vi; } });
}

function tickTimer() {
  if (!S.running) return;
  const elapsed = (performance.now() - S.t0) / 1000;
  const mm = Math.floor(elapsed / 60), ss = Math.floor(elapsed % 60);
  document.getElementById('elapsedLabel').textContent = mm > 0 ? mm + 'm ' + ss + 's' : ss + 's';
  requestAnimationFrame(tickTimer);
}

var _sseRetryCount = 0;
var _sseMaxRetry = 10;

function connectSSE() {
  console.log('[SSE] 连接中...');
  requestAnimationFrame(tickTimer);
  var es = new EventSource('/events');

  es.addEventListener('init', e => {
    _sseRetryCount = 0;
    hideReconnectBanner();
    var data = JSON.parse(e.data);
    console.log('[SSE] init 收到, 任务数:', data.files ? data.files.length : 0);
    S.t0 = performance.now();
    initTasks(data.files);
    document.getElementById('logPanel').style.display = '';
    document.getElementById('logBody').innerHTML = '';
    S._logLines = [];
  });

  var tickSeq = 0;
  es.addEventListener('tick', e => {
    tickSeq++;
    if (!S.t0) S.t0 = performance.now();  // 重连后恢复 t0
    var d = JSON.parse(e.data);
    document.getElementById('headerCount').textContent = d.n_total + ' 个文件';

    // phase badge
    var phase = d.phase_label;
    var pl = document.getElementById('phaseLabel');
    if (phase.includes('All done') || phase.includes('Done')) {
      pl.textContent = '已完成'; pl.style.color = 'var(--green)'; pl.style.background = 'var(--green-bg)'; pl.style.borderColor = 'var(--green-border)';
    } else if (phase.includes('Refine') || phase.includes('ASR +')) {
      pl.textContent = '识别+精修'; pl.style.color = 'var(--amber)'; pl.style.background = 'var(--amber-bg)'; pl.style.borderColor = 'var(--amber-border)';
    } else {
      pl.textContent = '语音识别'; pl.style.color = 'var(--cyan)'; pl.style.background = 'var(--cyan-bg)'; pl.style.borderColor = 'var(--cyan-border)';
    }

    document.getElementById('overallPct').textContent = Math.round(d.overall_pct * 100) + '%';
    var ob = document.getElementById('overallBar');
    ob.style.width = (d.overall_pct * 100) + '%';
    ob.className = d.overall_pct >= 0.99 ? 'overall-fill complete' : 'overall-fill';

    var tstates = d.tasks;
    var nDone = 0, nAsr = 0, nRef = 0, needReindex = false;

    for (var i = 0; i < S.tasks.length && i < tstates.length; i++) {
      var cur = S.tasks[i], upd = tstates[i];
      cur.status = upd.status;
      cur.asrPct = upd.asr_pct;
      cur.refinePct = upd.refine_pct;
      cur.error = upd.error || '';
      cur.fading = upd.exit_alpha < 0.5 && upd.status === 'DONE';
      if (upd.exit_alpha < 0.02 && cur.status === 'DONE' && !cur.removed) {
        cur.removed = true;
        cur._dom.card.className = 'task-card removing';
        needReindex = true;
      }
      if (!cur.removed) patchCard(cur);
      if (cur.status === 'DONE') nDone++;
      if (cur.status === 'ASR') nAsr++;
      if (cur.status === 'REFINE') nRef++;
    }
    if (needReindex) reindexCards();
    document.getElementById('statDone').textContent = nDone;
    document.getElementById('statAsr').textContent = nAsr;
    document.getElementById('statRef').textContent = nRef;

    // ETA 估算
    var etaEl = document.getElementById('etaLabel');
    if (d.overall_pct > 0.01 && S.t0) {
      var elapsed = (performance.now() - S.t0) / 1000;
      var etaSec = elapsed / d.overall_pct - elapsed;
      if (etaSec > 0 && isFinite(etaSec)) {
        var em = Math.floor(etaSec / 60), es = Math.floor(etaSec % 60);
        etaEl.textContent = '剩余 ≈ ' + (em > 0 ? em + '分' + es + '秒' : es + '秒');
      }
    }
  });

  // 实时日志事件
  es.addEventListener('log', e => {
    var entry = JSON.parse(e.data);
    appendLog(entry);
  });

  es.addEventListener('done', e => {
    var data = JSON.parse(e.data);
    var elapsed = ((performance.now() - S.t0) / 1000).toFixed(1);
    var isError = data.status === 'error';
    console.log('[SSE] 管线完成, 总耗时:', elapsed + 's, 状态:', data.status, 'tickSeq:', tickSeq);
    S.done = true; S.running = false;
    es.close();

    if (isError) {
      updateStatusBar('失败', 'phase-fail', data.message || '流水线处理出错');
      var pl2 = document.getElementById('phaseLabel');
      pl2.textContent = '处理失败'; pl2.style.color = 'var(--red, #c0392b)';
      pl2.style.background = 'var(--red-bg, #c0392b18)';
      pl2.style.borderColor = 'var(--red-border, #c0392b33)';
      appendLog({level:'ERROR',msg:'流水线失败: ' + (data.message || '未知')});
      var statTotal = document.getElementById('statTotalLabel');
      if (statTotal) statTotal.textContent = '错误: ' + (data.message || '未知');
    } else {
      updateStatusBar('已完成', 'phase-done', '全部文件处理完毕');
      var pl3 = document.getElementById('phaseLabel');
      pl3.textContent = '已完成'; pl3.style.color = 'var(--green)'; pl3.style.background = 'var(--green-bg)'; pl3.style.borderColor = 'var(--green-border)';
      appendLog({level:'INFO',msg:'全部文件处理完毕 (' + data.total + ' 个文件)'});
    }
    document.getElementById('overallPct').textContent = data.total > 0 ? '100%' : '—';
    document.getElementById('overallBar').style.width = data.total > 0 ? '100%' : '0%';
    document.getElementById('overallBar').className = isError ? 'overall-fill' : 'overall-fill complete';
    S.tasks.forEach(function(t) {
      if (t.status !== 'DONE' && t.status !== 'FAIL') {
        t.status = isError ? 'FAIL' : 'DONE'; t.asrPct = 1; t.refinePct = 1;
      }
      if (!t.removed) patchCard(t);
    });
    document.getElementById('statDone').textContent = isError ? '0' : String(S.tasks.length);
    document.getElementById('statAsr').textContent = '0';
    document.getElementById('statRef').textContent = '0';
    var elapsedSec = (performance.now() - S.t0) / 1000;
    renderSummary(data, elapsedSec);
    document.getElementById('elapsedLabel').textContent = elapsedSec.toFixed(1) + 's';
    document.getElementById('etaLabel').textContent = '';
    // 显示操作按钮
    document.getElementById('actionBar').style.display = 'flex';
    // 自动展开日志面板
    document.getElementById('logPanel').style.display = '';
    var logBody = document.getElementById('logBody');
    if (logBody) logBody.scrollTop = logBody.scrollHeight;
  });

  // SSE 断线重连
  es.onerror = function(ev) {
    console.warn('[SSE] 连接异常 (readyState=' + es.readyState + ')');
    if (es.readyState === EventSource.CLOSED) {
      es.close();
      _sseRetryCount++;
      if (_sseRetryCount <= _sseMaxRetry && !S.done) {
        showReconnectBanner(_sseRetryCount);
        var delay = Math.min(1000 * Math.pow(2, _sseRetryCount - 1), 30000);
        console.log('[SSE] 将在 ' + delay + 'ms 后重连 (第' + _sseRetryCount + '次)');
        setTimeout(function() {
          if (!S.done) recoverAndReconnect();
        }, delay);
      } else if (_sseRetryCount > _sseMaxRetry) {
        showReconnectBanner(-1);  // 放弃
      }
    }
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// SSE 重连 & 日志面板
// ═══════════════════════════════════════════════════════════════════════════
function showReconnectBanner(retry) {
  var b = document.getElementById('reconnectBanner');
  if (retry < 0) {
    b.textContent = '⚠ 无法重连，请刷新页面';
    b.style.background = 'var(--red-bg)'; b.style.color = 'var(--red)';
    b.classList.add('show');
  } else {
    b.textContent = '⚠ 连接断开，正在重连… (' + retry + '/' + _sseMaxRetry + ')';
    b.style.background = 'var(--amber-bg)'; b.style.color = 'var(--amber)';
    b.classList.add('show');
  }
}
function hideReconnectBanner() {
  document.getElementById('reconnectBanner').classList.remove('show');
}

async function recoverAndReconnect() {
  try {
    var r = await fetch('/api/snapshot');
    var snap = await r.json();
    if (snap.ok && snap.snapshot && S.tasks.length > 0) {
      console.log('[SSE] 重连成功，恢复快照');
      // 用快照恢复界面
      var d = snap.snapshot;
      S.t0 = performance.now();  // 重置计时
      // 恢复日志
      if (snap.logs && snap.logs.length > 0) {
        var body = document.getElementById('logBody');
        body.innerHTML = '';
        S._logLines = [];
        snap.logs.forEach(function(e) { appendLog(e); });
      }
    }
  } catch(e) { console.warn('[SSE] 快照恢复失败', e); }
  connectSSE();
}

// ═══════════════════════════════════════════════════════════════════════════
// 实时日志面板
// ═══════════════════════════════════════════════════════════════════════════
var _logCollapsed = false;
function toggleLogPanel() {
  _logCollapsed = !_logCollapsed;
  var body = document.getElementById('logBody');
  var icon = document.getElementById('logToggleIcon');
  if (_logCollapsed) {
    body.style.display = 'none'; icon.textContent = '▶';
  } else {
    body.style.display = ''; icon.textContent = '▼';
  }
}

function appendLog(entry) {
  if (!entry) return;
  S._logLines = S._logLines || [];
  S._logLines.push(entry);
  if (S._logLines.length > 200) S._logLines = S._logLines.slice(-150);
  var body = document.getElementById('logBody');
  if (!body || _logCollapsed) return;
  var ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString() : '';
  var css = (entry.level === 'ERROR' || entry.level === 'WARN') ? entry.level.toLowerCase() : '';
  var line = document.createElement('div');
  line.className = 'log-line' + (css ? ' ' + css : '');
  line.innerHTML = '<span class="log-ts">' + escHtml(ts) + '</span>' + escHtml(entry.msg || entry.message || '');
  body.appendChild(line);
  body.scrollTop = body.scrollHeight;
  document.getElementById('logCount').textContent = S._logLines.length;
}

// ═══════════════════════════════════════════════════════════════════════════
// 完成后操作：新任务 & 打开输出目录
// ═══════════════════════════════════════════════════════════════════════════
function resetToForm() {
  S.done = false; S.running = false; S.tasks = [];
  const monitor = document.getElementById('monitorSection');
  const launcher = document.getElementById('launcherSection');
  monitor.style.display = 'none';
  monitor.classList.remove('showing', 'visible');
  launcher.style.display = '';
  launcher.classList.remove('hiding');
  document.getElementById('actionBar').style.display = 'none';
  document.getElementById('summaryCard').style.display = 'none';
  document.getElementById('logPanel').style.display = 'none';
  updateStatusBar('就绪', 'phase-ready', '');
  hideReconnectBanner();
  _sseRetryCount = 0;
}

function openOutputDir() {
  fetch('/api/open_output').then(function(r) { return r.json(); }).then(function(d) {
    if (!d.ok) alert('无法打开: ' + (d.message || ''));
  }).catch(function() {});
}

// ═══════════════════════════════════════════════════════════════════════════
// Preset 导入/导出
// ═══════════════════════════════════════════════════════════════════════════
function exportPresets() {
  var raw = localStorage.getItem(PRESET_KEY) || '[]';
  var blob = new Blob([raw], {type:'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'm1_presets_' + new Date().toISOString().slice(0,10) + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function importPresets() {
  var inp = document.createElement('input');
  inp.type = 'file'; inp.accept = '.json';
  inp.onchange = function() {
    var file = inp.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function() {
      try {
        var imported = JSON.parse(reader.result);
        if (!Array.isArray(imported)) throw new Error('不是有效的预设文件');
        var existing = [];
        try { existing = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); } catch(e) {}
        var merged = {};
        existing.forEach(function(p) { merged[p.name] = p; });
        imported.forEach(function(p) { merged[p.name] = p; });
        var result = Object.values(merged);
        localStorage.setItem(PRESET_KEY, JSON.stringify(result));
        loadPresets();
        alert('已导入 ' + imported.length + ' 个预设 (合并后共 ' + result.length + ' 个)');
      } catch(e) {
        alert('导入失败: ' + e.message);
      }
    };
    reader.readAsText(file);
  };
  inp.click();
}

// ═══════════════════════════════════════════════════════════════════════════
// API Key 前置检查
// ═══════════════════════════════════════════════════════════════════════════
function checkApiKeys() {
  var warnings = [];
  var backend = getVal('fldAsrBackend') || 'funasr';
  // Cloud ASR
  if (backend === 'cloud' && !getVal('fldCloudApiKey')) {
    warnings.push('Cloud ASR 后端已选择但未填写 API Key，将读取环境变量 DASHSCOPE_API_KEY');
  }
  // Refine / Summary 共用 LLM Key
  if ((getVal('fldRefineEnable') || getVal('fldSummaryEnable')) && !getVal('fldRefineApiKey')) {
    warnings.push('LLM 精修/摘要已启用但未填写 API Key，将读取环境变量或使用默认配置');
  }
  return warnings;
}

function showApiKeyWarning(warnings, onProceed) {
  var overlay = document.createElement('div');
  overlay.className = 'warn-overlay';
  overlay.id = 'warnOverlay';
  var html = '<div class="warn-dialog"><h3>⚠ API Key 提醒</h3>';
  html += '<p>' + warnings.map(function(w) { return escHtml(w); }).join('<br><br>') + '</p>';
  html += '<p style="font-size:11px;color:var(--text-dim);">如已在 .env 中配置则可忽略，点击"继续启动"。</p>';
  html += '<div class="warn-buttons">';
  html += '<button class="warn-cancel" onclick="document.getElementById(\'warnOverlay\').remove()">返回修改</button>';
  html += '<button class="warn-proceed" id="warnProceedBtn">继续启动</button>';
  html += '</div></div>';
  overlay.innerHTML = html;
  document.body.appendChild(overlay);
  document.getElementById('warnProceedBtn').onclick = function() {
    overlay.remove();
    onProceed();
  };
}
</script>
</body>
</html>"""


class LauncherHandler(BaseHTTPRequestHandler):
    """HTTP 路由: / 页面, /api/* REST, /events SSE。"""

    # ── 请求日志 ────────────────────────────────────────────────────
    def log_message(self, format, *args):
        """覆盖基类方法：结构化记录每个 HTTP 请求。"""
        client = self.client_address[0]
        try:
            msg = format % args
        except Exception:
            msg = format
        logger.info("[HTTP] %s | %s %s → %s", client, self.command,
                     self.path, msg)

    def _log_api(self, action: str, **kwargs):
        """记录 API 级操作，便于追踪业务行为。"""
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        logger.info("[API] %s | %s", action, "  ".join(parts) if parts else "")

    # ── 路由分发 ────────────────────────────────────────────────────
    def do_GET(self):
        t0 = time.time()
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            self._serve_page()
        elif path == "/api/hardware_config":
            self._api_hardware_config()
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/snapshot":
            self._api_snapshot()
        elif path == "/api/open_output":
            self._api_open_output()
        elif path == "/events":
            self._log_api("SSE连接", client=self.client_address[0])
            self._serve_sse()
        else:
            logger.warning("[HTTP] 404 | %s %s", self.command, path)
            self.send_error(404)
        self._log_timing(path, t0)

    def do_POST(self):
        t0 = time.time()
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/start":
            self._api_start()
        elif path == "/api/preview":
            self._api_preview()
        elif path == "/api/abort":
            self._api_abort()
        elif path == "/api/browse_folder":
            self._api_browse_folder()
        else:
            logger.warning("[HTTP] 404 | %s %s", self.command, path)
            self.send_error(404)
        self._log_timing(path, t0)

    def _log_timing(self, path, t0):
        elapsed = (time.time() - t0) * 1000
        if elapsed > 100:
            logger.info("[HTTP] %s 耗时 %.0fms", path, elapsed)

    # ── 辅助 ────────────────────────────────────────────────────────
    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            data = self.rfile.read(length)
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_page(self):
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    # ── API: 硬件配置 ───────────────────────────────────────────────
    def _api_hardware_config(self):
        launcher: LauncherWindow = self.server.launcher
        self._json_response(launcher.load_hardware_config())

    # ── API: 目录预览 ───────────────────────────────────────────────
    def _api_preview(self):
        launcher: LauncherWindow = self.server.launcher
        data = self._read_json_body()
        paths = data.get("paths", [])
        self._log_api("预览扫描", paths_count=len(paths),
                       paths="[" + ", ".join(p[:60] for p in paths[:5]) + "]")
        result = launcher.preview_paths(paths)
        self._log_api("预览结果", total_files=result.get("total_files", 0))
        self._json_response(result)

    # ── API: 启动管线 ───────────────────────────────────────────────
    def _api_start(self):
        launcher: LauncherWindow = self.server.launcher
        if launcher.running and not launcher.is_done():
            logger.warning("[API] 启动被拒 — 管线已在运行中")
            self._json_response({"ok": False, "errors": [
                {"field": "_", "message": "管线已在运行中"}
            ]})
            return

        data = self._read_json_body()
        series_id = data.get("series_id", "").strip()
        source_paths = data.get("source_paths", [])
        output_dir = data.get("output_dir", "").strip()
        config = data.get("config", {})

        # 日志使用扁平键（与 JS collectConfig 一致）
        self._log_api("启动管线", series_id=series_id,
                       sources=len(source_paths), output=output_dir,
                       asr_model=config.get("asr_model", "?"),
                       asr_workers=config.get("asr_workers", "?"),
                       postprocess_workers=config.get("postprocess_workers", "?"),
                       enable_refine=config.get("enable_refine", True))

        try:
            result = launcher.start(series_id, source_paths, output_dir, config)
        except Exception as e:
            logger.error("[API] 管线启动异常: %s", e, exc_info=True)
            self._json_response({"ok": False, "errors": [
                {"field": "_", "message": f"内部错误: {e}"}
            ]}, status=500)
            return

        if result.get("ok"):
            self.server.sse_queue = launcher.sse_queue
            self.server.init_payload = launcher.init_payload()
            self.server.snapshot_payload = launcher.snapshot_payload
            self.server.running = True
            self._log_api("管线已启动", files=len(source_paths))
        else:
            logger.error("[API] 启动失败: %s", result.get("errors", "未知错误"))
        self._json_response(result)

    # ── API: 中止 ───────────────────────────────────────────────────
    def _api_abort(self):
        launcher: LauncherWindow = self.server.launcher
        elapsed = launcher.elapsed_sec()
        self._log_api("中止管线", elapsed_sec=f"{elapsed:.1f}")
        launcher.abort()
        self.server.running = False
        self._json_response({"ok": True})

    # ── API: 原生文件夹选择对话框 ───────────────────────────────────
    def _api_browse_folder(self):
        """通过 tkinter 打开系统原生文件夹选择对话框（标准库，无额外依赖）。"""
        self._log_api("打开文件夹对话框")
        import tkinter as tk
        from tkinter import filedialog

        try:
            root = tk.Tk()
            root.withdraw()                      # 隐藏主窗口
            root.attributes("-topmost", True)    # 对话框置顶
            root.lift()
            root.focus_force()
            path = filedialog.askdirectory(
                parent=root,
                title="选择文件夹",
                mustexist=True,
            )
            root.destroy()
            if path and os.path.isdir(path):
                norm = os.path.normpath(path)
                self._log_api("文件夹已选择", path=norm)
                self._json_response({"path": norm})
            else:
                self._log_api("文件夹选择已取消")
                self._json_response({"path": "", "cancelled": True})
        except Exception as e:
            logger.error("[API] 文件夹对话框异常: %s", e)
            self._json_response({"path": "", "error": str(e)}, status=500)

    # ── API: 状态轮询 + 快照 ──────────────────────────────────────
    def _api_status(self):
        launcher: LauncherWindow = self.server.launcher
        done = launcher.is_done() if launcher.running else True
        result = {
            "running": launcher.running,
            "done": done,
            "elapsed_sec": launcher.elapsed_sec(),
            "file_count": len(launcher.file_names),
            "last_status": launcher.last_status,
            "output_dir": launcher.last_output_dir if done else "",
            "series_id": launcher.last_series_id,
        }
        # 如果正在运行，附加当前快照
        if launcher.running:
            result["snapshot"] = launcher.snapshot_payload()
        self._json_response(result)

    def _api_snapshot(self):
        """SSE 重连后获取当前进度快照。"""
        launcher: LauncherWindow = self.server.launcher
        if launcher.running and launcher.pipeline_win:
            snap = launcher.snapshot_payload()
            # 同时返回最近的日志
            logs = launcher.log_queue[-50:] if launcher.log_queue else []
            self._json_response({"ok": True, "snapshot": snap, "logs": logs})
        else:
            self._json_response({"ok": False, "message": "没有正在运行的管线"})

    def _api_open_output(self):
        """在系统文件管理器中打开输出目录。"""
        launcher: LauncherWindow = self.server.launcher
        out_dir = launcher.last_output_dir
        if out_dir and os.path.isdir(out_dir):
            import subprocess
            try:
                if os.name == "nt":
                    cmd = ["explorer", os.path.normpath(out_dir)]
                elif sys.platform == "darwin":
                    cmd = ["open", out_dir]
                else:
                    cmd = ["xdg-open", out_dir]
                subprocess.Popen(cmd)
                self._json_response({"ok": True})
            except Exception as e:
                self._json_response({"ok": False, "message": str(e)})
        else:
            self._json_response({"ok": False, "message": "输出目录不存在"})

    # ── SSE ──────────────────────────────────────────────────────────
    def _serve_sse(self):
        launcher: LauncherWindow = self.server.launcher
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sse_q = getattr(self.server, "sse_queue", None)
        init_data = getattr(self.server, "init_payload", None)
        snapshot_fn = getattr(self.server, "snapshot_payload", None)

        if sse_q is None:
            self._log_api("SSE错误", reason="管线未启动")
            self._write_sse("error", {"message": "管线未启动"})
            return

        # 发送 init（init_payload 在 _api_start 中已预计算为 dict）
        if init_data:
            files = init_data.get("files", [])
            self._log_api("SSE:init", file_count=len(files))
            self._write_sse("init", init_data)

        # 流式推送
        tick_count = 0
        while getattr(self.server, "running", False):
            try:
                msg = sse_q.get(timeout=1.0)
            except queue.Empty:
                tick_count += 1
                if snapshot_fn:
                    self._write_sse("tick", snapshot_fn())
                continue
            if msg is None:
                break
            event_name, payload = msg
            if event_name == "done":
                done_files = payload.get("done", 0) if isinstance(payload, dict) else 0
                total = payload.get("total", 0) if isinstance(payload, dict) else 0
                err_msg = payload.get("message", "") if isinstance(payload, dict) else ""
                self._log_api("SSE:done", done=done_files, total=total,
                              ticks=tick_count, error=err_msg[:120] if err_msg else "")
            elif event_name == "error":
                self._log_api("SSE:error", payload=str(payload)[:120])
            self._write_sse(event_name, payload)
            if event_name == "done":
                self.server.running = False
                break
        self._log_api("SSE断开", ticks=tick_count, client=self.client_address[0])

    def _write_sse(self, event: str, data):
        try:
            payload = json.dumps(data, ensure_ascii=False, default=str)
            self.wfile.write(
                f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            raise


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Yiren Pipeline Web Launcher")
    parser.add_argument("--port", type=int, default=0,
                       help="HTTP 端口（0=自动分配）")
    parser.add_argument("--no-browser", action="store_true", default=False,
                       help="不自动打开浏览器")
    args = parser.parse_args()

    port = args.port or _pick_port()
    url = f"http://127.0.0.1:{port}"

    # 创建 LauncherWindow 并挂载到 server
    launcher = LauncherWindow()
    server = HTTPServer(("127.0.0.1", port), LauncherHandler)
    server.launcher = launcher
    server.sse_queue = None
    server.init_payload = None
    server.snapshot_payload = None
    server.running = False
    server.timeout = 0.5

    print(f"\n  Yiren Pipeline Launcher  →  {url}\n")
    print("  选择源文件夹、配置参数后点击「启动流水线」按钮\n")

    if not args.no_browser:
        def _open():
            try:
                webbrowser.open(url)
            except Exception:
                logger.debug("浏览器打开失败", exc_info=True)
        threading.Thread(target=_open, daemon=True).start()

    # 定时检查管线完成，自动清理
    def _idle_watchdog():
        """30 分钟无活动自动关闭。"""
        last_activity = time.time()
        while True:
            time.sleep(30)
            if launcher.running:
                last_activity = time.time()
            elif time.time() - last_activity > 1800:
                logger.info("30 分钟无活动，自动关闭")
                launcher.abort()
                break

    threading.Thread(target=_idle_watchdog, daemon=True).start()

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n关闭...")
    finally:
        launcher.abort()
        try:
            server.server_close()
        except Exception:
            logger.debug("服务器关闭异常", exc_info=True)


if __name__ == "__main__":
    main()
