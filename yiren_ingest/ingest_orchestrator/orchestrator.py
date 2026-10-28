"""YirenIngest 主编排器：ingest_series_material()

流水线：视频提取音频 → 长音频切分 → ASR 转写 → LLM 精修 → 章节/全文总览。
ASR 串行（GPU 独占），精修与摘要走网络 API、按文件后台并发。
任何阶段失败都降级保留已完成产物，不中断整个系列。
"""

import json
import logging
import os

from .. import utils
from ..audio_processing import (
    extract_audio_from_video,
    get_audio_duration_sec,
    segment_audio,
)

logger = logging.getLogger(__name__)

_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}
_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


# ═══════════════════════════════════════════
# ID 生成与目录
# ═══════════════════════════════════════════

def generate_upload_id() -> str:
    return f"UP_{utils.utc_timestamp()}_{utils.random_suffix()}"


def series_id_to_compact(series_id: str) -> str:
    return series_id.replace("_", "")


def generate_transcript_id(series_id: str, seq: int) -> str:
    return f"TR_{series_id_to_compact(series_id)}_{seq:03d}"


def setup_series_dirs(series_id: str, upload_id: str,
                      base_dir: str = "data") -> dict[str, str]:
    root = os.path.join(base_dir, "series", series_id)
    dirs = {
        "root": root,
        "uploads": os.path.join(root, "uploads", upload_id),
        "transcripts": os.path.join(root, "transcripts"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ═══════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════

def _classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    return "unknown"


def _emit(fn, stage, progress=None, **meta):
    """安全调用 on_progress 回调（回调异常不影响主流程）。"""
    if fn:
        try:
            fn(stage, progress, **meta)
        except Exception:
            logger.debug(f"on_progress 回调异常 stage={stage}", exc_info=True)


def _rm(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.debug(f"删除临时文件失败: {path}")


def _file_fingerprint(path: str) -> str:
    """文件指纹：文件名 + 大小 + 修改时间，用于幂等判断。"""
    try:
        stat = os.stat(path)
        return f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        logger.debug(f"无法获取文件指纹: {path}")
        return ""


def _load_ingest_index(transcripts_dir: str) -> dict:
    index_path = os.path.join(transcripts_dir, "_ingest_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning(f"摄取索引损坏，将重建: {index_path}")
    return {"entries": {}}


def _save_ingest_index(transcripts_dir: str, index: dict):
    utils.atomic_write_json(os.path.join(transcripts_dir, "_ingest_index.json"), index)


def _validate_transcript(transcript: dict) -> list[str]:
    """校验 transcript 必需字段，返回缺失字段列表。"""
    missing = []
    for key in ("transcript_id", "series_id", "source_audio", "language", "segments"):
        if key not in transcript:
            missing.append(key)
    if "segments" in transcript:
        if not isinstance(transcript["segments"], list):
            missing.append("segments(非数组)")
        else:
            for i, seg in enumerate(transcript["segments"]):
                for key in ("segment_id", "text"):
                    if key not in seg:
                        missing.append(f"segments[{i}].{key}")
    return missing


def _check_quality(transcript: dict) -> list[str]:
    """检测产出质量异常（仅警告，不阻止流程）。"""
    warnings = []
    segs = transcript.get("segments", [])
    quality = transcript.get("asr_quality", {})

    cer = quality.get("estimated_cer", 0)
    if cer >= 0.99:
        warnings.append(f"estimated_cer={cer}（≥99%，疑似 ASR 解析失败或音频异常）")

    if segs and all(
        s.get("start", 0) == 0.0 and s.get("end", 0) == 0.0 for s in segs
    ):
        warnings.append("所有 segment 时间戳均为 0，缺少时间锚点")

    missing_conf = sum(1 for s in segs if "confidence" not in s)
    if missing_conf > 0:
        warnings.append(f"{missing_conf}/{len(segs)} 个 segment 缺少 confidence 字段")

    low = quality.get("low_confidence_segments", 0)
    if segs and low > 0 and low / len(segs) >= 0.5:
        warnings.append(
            f"低置信度段占比 {low}/{len(segs)}（≥50%，内容可信度存疑）"
        )
    return warnings


def _make_asr_config(config):
    """把 orchestrator 平面配置映射为 ASR pipeline 键名。"""
    asr_cfg = {
        "asr_backend": config.get("asr_backend", "auto"),
        "model_size": config.get("model_size") or config.get("asr_model", "medium"),
        "model": config.get("model"),
        "device": config.get("asr_device") or "auto",
        "compute_type": config.get("compute_type") or config.get("asr_compute_type") or "auto",
        "cpu_threads": config.get("cpu_threads") or config.get("asr_cpu_threads", 4),
        "ncpu": config.get("ncpu"),
        "cache_dir": config.get("asr_cache_dir"),
        "language": config.get("language", "zh"),
    }
    for key in ("beam_size", "vad_filter", "vad_model", "punc_model",
                "use_vad", "hotwords", "provider", "api_key", "base_url"):
        if key in config and config[key] is not None:
            asr_cfg[key] = config[key]
    # None 不能进入引擎配置，否则会覆盖 FUNASR_DEFAULTS/WHISPER_DEFAULTS
    return {k: v for k, v in asr_cfg.items() if v is not None}


def _run_asr(audio_path, tid, series_id, src_name, config, lock):
    from ..asr_pipeline import transcribe_audio
    asr_cfg = _make_asr_config(config)
    if lock:
        with lock:
            return transcribe_audio(audio_path, tid, series_id, src_name, config=asr_cfg)
    return transcribe_audio(audio_path, tid, series_id, src_name, config=asr_cfg)


def _transcribe(audio_path, tid, series_id, src_name, config, tmp_dir, lock,
                on_progress=None):
    """ASR 转写；音频超长时切分后逐段转写再合并时间轴。"""
    seg_max = config.get("segment_max_duration_sec", 1800)

    dur = get_audio_duration_sec(audio_path)
    if dur <= seg_max:
        return _run_asr(audio_path, tid, series_id, src_name, config, lock)

    logger.info(f"音频过长 ({dur:.0f}s)，按 {seg_max}s 切分")
    _emit(on_progress, "audio_segmenting")
    seg_paths = segment_audio(audio_path, os.path.join(tmp_dir, f"segs_{tid}"),
                              max_duration_sec=seg_max)

    all_segs, combined, offset = [], None, 0.0
    for i, sp in enumerate(seg_paths):
        logger.info(f"转写分段 {i + 1}/{len(seg_paths)}: {os.path.basename(sp)}")
        seg_progress = round((i + 1) / len(seg_paths), 3) if seg_paths else 1.0
        _emit(on_progress, "asr_transcribing", seg_progress)
        r = _run_asr(sp, tid, series_id, src_name, config, lock)
        for s in r["segments"]:
            s["start"] = round(s["start"] + offset, 2)
            s["end"] = round(s["end"] + offset, 2)
            s["segment_id"] = f"seg_{len(all_segs) + 1:04d}"
            all_segs.append(s)
        if combined is None:
            combined = r
        try:
            offset += get_audio_duration_sec(sp)
        except Exception:
            logger.debug(f"获取分段时长失败，使用默认值 seg_max={seg_max}: {sp}")
            offset += seg_max

    if combined is None:
        raise RuntimeError("所有分段转写失败")
    combined["segments"] = all_segs
    return combined


def _refine_config(config: dict) -> dict:
    return {
        "api_url": config.get("refine_api_url"),
        "api_key": config.get("refine_api_key"),
        "model": config.get("refine_model", "deepseek-v4-pro"),
        "timeout": config.get("refine_timeout", 180),
        "enable_thinking": config.get("refine_enable_thinking", False),
        "max_output_tokens": config.get("refine_max_output_tokens", 32768),
    }


def _summary_config(config: dict) -> dict:
    base = _refine_config(config)
    return {
        "api_url": config.get("summary_api_url") or base["api_url"],
        "api_key": config.get("summary_api_key") or base["api_key"],
        "model": config.get("summary_model") or base["model"],
        "timeout": config.get("summary_timeout") or base["timeout"],
        "enable_thinking": config.get(
            "summary_enable_thinking", base["enable_thinking"]
        ),
        "max_output_tokens": config.get(
            "summary_max_output_tokens", base["max_output_tokens"]
        ),
    }


def _postprocess(t_path, tid, config):
    """单文件后处理：精修 → 章节摘要。失败降级，不抛异常。"""
    result = {
        "refined": False,
        "summarized": False,
        "chapters_count": 0,
        "errors": [],
    }

    if config.get("enable_refine"):
        try:
            from ..text_cleaning import refine
            r = refine(t_path, t_path, config=_refine_config(config))
            if r.get("status") == "success":
                result["refined"] = True
            else:
                result["errors"].append(f"refine: {r.get('message', 'unknown')}")
                logger.warning(f"精修失败 [{tid}]: {r.get('message', 'unknown')}")
        except Exception as e:
            result["errors"].append(f"refine: {e}")
            logger.warning(f"精修异常 [{tid}]: {e}")

    if config.get("enable_summary"):
        try:
            from ..summarizer import summarize_transcript
            r = summarize_transcript(t_path, config=_summary_config(config))
            if r.get("status") == "success":
                result["summarized"] = True
                result["chapters_count"] = r.get("chapters_count", 0)
            else:
                result["errors"].append(f"summary: {r.get('message', 'unknown')}")
                logger.warning(f"摘要失败 [{tid}]: {r.get('message', 'unknown')}")
        except Exception as e:
            result["errors"].append(f"summary: {e}")
            logger.warning(f"摘要异常 [{tid}]: {e}")

    return result


# ═══════════════════════════════════════════
# 处理器
# ═══════════════════════════════════════════

def _process_media(src, tid, series_id, dirs, config, lock, on_progress, is_video):
    """视频/音频：提取音频（仅视频）→ ASR → 校验 → 写 transcript。"""
    src_name = os.path.basename(src)

    if is_video:
        _emit(on_progress, "audio_extracting", 0.0)
        audio_path = os.path.join(dirs["_tmp"], f"_tmp_{tid}.wav")
        try:
            extract_audio_from_video(src, audio_path)
        except ImportError as e:
            return {"status": "failed", "path": src, "reason": str(e)}
        _emit(on_progress, "audio_extracting", 0.1)
    else:
        audio_path = src

    _emit(on_progress, "asr_transcribing", 0.15)
    try:
        transcript = _transcribe(audio_path, tid, series_id, src_name,
                                 config, dirs["_tmp"], lock,
                                 on_progress=on_progress)
    except Exception as e:
        if not config.get("save_intermediate"):
            _rm(audio_path if is_video else None)
        return {"status": "failed", "path": src, "reason": str(e)}
    _emit(on_progress, "asr_transcribing", 0.85)

    # source_audio 指向原始源文件（持久存在），而非临时提取的 WAV
    transcript["source_audio"] = src

    missing = _validate_transcript(transcript)
    if missing:
        if not config.get("save_intermediate"):
            _rm(audio_path if is_video else None)
        return {
            "status": "failed", "path": src,
            "reason": f"transcript schema 校验失败，缺失字段: {', '.join(missing)}",
        }

    t_path = os.path.join(dirs["transcripts"], f"{tid}.json")
    utils.atomic_write_json(t_path, transcript)

    quality_warnings = _check_quality(transcript)
    if quality_warnings:
        logger.warning(
            f"产出质量警告 [{os.path.basename(t_path)}]: "
            + "; ".join(quality_warnings)
        )

    segs = transcript.get("segments", [])
    dur = round(segs[-1]["end"] - segs[0]["start"], 1) if segs else 0.0
    return {
        "status": "success",
        "source": src,
        "transcript_id": tid,
        "transcript_path": t_path,
        "segments_count": len(segs),
        "duration_sec": dur,
        "_audio_source": audio_path,
        "_audio_is_temp": is_video,
    }


def ingest_single_file(upload_path, series_id, series_dirs, config, seq,
                       asr_lock=None, on_progress=None):
    """处理单个视频/音频文件，返回结果 dict。"""
    if not os.path.exists(upload_path):
        return {"status": "failed", "path": upload_path,
                "reason": f"文件不存在: {upload_path}"}

    ft = _classify(upload_path)
    if ft == "unknown":
        return {"status": "failed", "path": upload_path,
                "reason": f"不支持的文件类型: {os.path.splitext(upload_path)[1]}"}

    tid = generate_transcript_id(series_id, seq)
    logger.info(f"[seq={seq}] {os.path.basename(upload_path)} type={ft} -> {tid}")

    try:
        return _process_media(upload_path, tid, series_id, series_dirs,
                              config, asr_lock, on_progress,
                              is_video=(ft == "video"))
    except Exception as e:
        logger.error(f"处理异常 [{upload_path}]: {e}")
        return {"status": "failed", "path": upload_path, "reason": str(e)}


_DEFAULT_CONFIG = {
    # ── 功能开关 ──
    "language": "zh",
    "enable_refine": True,
    "enable_summary": True,
    "segment_max_duration_sec": 1800,
    "max_file_size_bytes": 2 * 1024 * 1024 * 1024,  # 2GB
    "save_intermediate": False,
    "force_reprocess": False,
    # ── ASR（保守默认，硬件参数由 CLI/hardware_config 注入） ──
    "asr_backend": "funasr",
    "asr_model": "medium",
    "asr_device": "auto",
    "asr_compute_type": "auto",
    "asr_cpu_threads": 4,
    "asr_cache_dir": None,
    # ── 各引擎并发数，0=自动 ──
    "whisper_workers": 0,
    "funasr_workers": 0,
    "cloud_workers": 0,
    # ── LLM 精修 ──
    "refine_api_url": None,
    "refine_api_key": None,
    "refine_model": "deepseek-v4-pro",
    "refine_timeout": 180,
    "refine_enable_thinking": False,
    "refine_max_output_tokens": 32768,
    # ── 章节摘要（未设置时复用 refine 配置） ──
    "summary_api_url": None,
    "summary_api_key": None,
    "summary_model": "deepseek-v4-pro",
    "summary_timeout": 180,
    "summary_enable_thinking": False,
    "summary_max_output_tokens": 32768,
    # ── 后处理并发 ──
    "postprocess_workers": 0,       # 0=自动（等于待处理文件数）
}


def ingest_series_material(series_id, upload_paths, output_dir,
                           config=None, on_progress=None):
    """系列视频摄取主入口（同步）。

    流程：
    1. 生成 upload_id，创建 data/series/{series_id}/ 目录结构
    2. 逐文件 ASR（GPU 串行），完成后立刻提交精修+摘要到后台线程池
    3. 等待后处理收尾，汇总 stats 与 warnings

    Returns:
        {"status": "success|partial|error", "series_id": "...",
         "transcripts": [...], "stats": {...}, "warnings": [...]}
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    if not upload_paths:
        return {
            "status": "error", "message": "upload_paths 为空",
            "series_id": series_id, "transcripts": [],
            "stats": {}, "warnings": ["upload_paths 为空"],
        }

    # 文件大小前置检查 — Fail-Fast at boundary
    max_size = cfg.get("max_file_size_bytes", 0)
    oversized = []
    for p in upload_paths:
        try:
            fsize = os.path.getsize(p)
        except OSError:
            continue
        if max_size > 0 and fsize > max_size:
            oversized.append((p, fsize))
    oversized_warnings = []
    if oversized:
        details = "; ".join(
            f"{os.path.basename(p)} ({s / (1024 ** 3):.1f}GB)" for p, s in oversized
        )
        logger.warning(f"拒绝 {len(oversized)} 个超大文件 (>{max_size / (1024 ** 3):.0f}GB): {details}")
        oversized_warnings.append(
            f"拒绝 {len(oversized)} 个超大文件 (>{max_size / (1024 ** 3):.0f}GB): {details}"
        )
        oversized_paths = {o[0] for o in oversized}
        upload_paths = [p for p in upload_paths if p not in oversized_paths]
        if not upload_paths:
            return {
                "status": "error",
                "message": f"所有文件均超过大小上限 ({max_size / (1024 ** 3):.0f}GB)",
                "series_id": series_id, "transcripts": [],
                "stats": {}, "warnings": oversized_warnings,
            }

    upload_id = generate_upload_id()
    series_dirs = {
        "root": output_dir,
        "transcripts": os.path.join(output_dir, "transcripts"),
        "_logs": os.path.join(output_dir, "_logs"),
        "_tmp": os.path.join(output_dir, "_tmp"),
    }
    for d in series_dirs.values():
        os.makedirs(d, exist_ok=True)

    _file_handler = logging.FileHandler(
        os.path.join(series_dirs["_logs"], f"ingest_{upload_id}.log"),
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(_file_handler)

    logger.info(f"开始摄取: series_id={series_id}, upload_id={upload_id}, "
                f"files={len(upload_paths)}")

    _emit(on_progress, "file_validating", 0.0)

    index = _load_ingest_index(series_dirs["transcripts"])
    index.setdefault("series_id", series_id)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    postprocess_pool = None
    postprocess_futures: dict = {}
    if cfg.get("enable_refine") or cfg.get("enable_summary"):
        workers = cfg.get("postprocess_workers") or max(len(upload_paths), 1)
        postprocess_pool = ThreadPoolExecutor(max_workers=workers)

    total = len(upload_paths)
    transcripts, failed = [], []

    def _make_file_progress(file_index: int):
        """固定闭包变量，避免并发下循环变量延迟绑定。"""
        f_base = file_index / total
        f_window = 1.0 / total

        def _cb(stage, local_progress=None):
            abs_p = (
                round(f_base + local_progress * f_window, 4)
                if local_progress is not None else None
            )
            _emit(on_progress, stage, abs_p, file_index=file_index)

        return _cb

    # ── 第一遍：幂等检查 ──
    pending: list[tuple[int, str]] = []
    for i, src in enumerate(upload_paths):
        fp = _file_fingerprint(src)
        if fp and not cfg.get("force_reprocess", False):
            cached = index.get("entries", {}).get(fp)
            if cached:
                cached["skipped"] = True
                cached["source"] = src
                transcripts.append(cached)
                logger.info(f"[seq={i + 1}] {os.path.basename(src)} 已存在，跳过 (幂等)")
                continue
        pending.append((i, src))

    # ── 第二遍：并发 ASR（GPU 引擎默认串行） ──
    if pending:
        backend = cfg.get("asr_backend", "auto")
        if backend == "cloud":
            asr_workers = cfg.get("cloud_workers") or cfg.get("asr_workers") or 4
        elif backend in ("whisper", "funasr"):
            asr_workers = cfg.get(f"{backend}_workers") or cfg.get("asr_workers") or 1
        else:
            asr_workers = cfg.get("asr_workers") or 1
        try:
            from ..asr_pipeline.whisper_runner import _set_concurrency
            _set_concurrency(asr_workers)
        except Exception:
            pass
        logger.info(f"并发 ASR: {len(pending)} 文件, {asr_workers} workers")

        with ThreadPoolExecutor(max_workers=asr_workers) as asr_pool:
            asr_futures: dict = {}
            for file_idx, src in pending:
                cb = _make_file_progress(file_idx)
                fut = asr_pool.submit(
                    ingest_single_file, src, series_id, series_dirs,
                    cfg, file_idx + 1, on_progress=cb,
                )
                asr_futures[fut] = (file_idx, src)

            for fut in as_completed(asr_futures):
                file_idx, src = asr_futures[fut]
                try:
                    item = fut.result()
                except Exception as e:
                    logger.error(f"处理异常 [{src}]: {e}")
                    failed.append({"status": "failed", "path": src, "reason": str(e)})
                    continue

                if item.get("status") != "success":
                    logger.error(f"处理失败 [{src}]: {item.get('reason', '未知错误')}")
                    failed.append(item)
                    continue

                item["_file_index"] = file_idx
                transcripts.append(item)

                # ASR 完成 → 立刻提交精修+摘要到后台线程池
                refine_path = item.get("transcript_path")
                refine_id = item.get("transcript_id", "")
                if postprocess_pool and refine_path:
                    fut_post = postprocess_pool.submit(
                        _postprocess, refine_path, refine_id, cfg
                    )
                    postprocess_futures[fut_post] = item
                    _emit(on_progress, "refine_started", None,
                          transcript_id=refine_id, file_index=file_idx)

                fp = _file_fingerprint(src)
                if fp:
                    index.setdefault("entries", {})[fp] = {
                        k: v for k, v in item.items()
                        if not k.startswith("_") and k != "source"
                    }

    # ── 等待所有后处理完成 ──
    if postprocess_pool and postprocess_futures:
        _emit(on_progress, "text_cleaning", 0.90)
        done = 0
        for fut in as_completed(postprocess_futures):
            item = postprocess_futures[fut]
            tid = item.get("transcript_id", "")
            try:
                post = fut.result()
            except Exception as e:
                logger.warning(f"后处理异常 [{tid}]: {e}")
                post = {"refined": False, "summarized": False,
                        "chapters_count": 0, "errors": [str(e)]}
            item["postprocess"] = post
            item["refined"] = post.get("refined", False)
            item["summarized"] = post.get("summarized", False)
            item["chapters_count"] = post.get("chapters_count", 0)

            done += 1
            _emit(on_progress, "text_cleaning",
                  round(0.90 + done / len(postprocess_futures) * 0.08, 4),
                  transcript_id=tid, file_index=item.get("_file_index"))

        # 重新读取文件，更新段落/时长/章节统计
        for item in postprocess_futures.values():
            path = item.get("transcript_path")
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                segs = d.get("segments", [])
                item["segments_count"] = len(segs)
                item["duration_sec"] = (
                    round(segs[-1]["end"] - segs[0]["start"], 1) if segs else 0
                )
                item["chapters_count"] = len(d.get("chapters", []))
                item["summarized"] = bool(d.get("overview") or d.get("chapters"))
            except Exception as e:
                logger.warning(f"后处理后重读失败 [{item.get('transcript_id')}]: {e}")

        _emit(on_progress, "text_cleaning", 0.98)
        postprocess_pool.shutdown(wait=False)

    # ── 清理临时音频 ──
    if not cfg.get("save_intermediate"):
        for item in transcripts:
            src_audio = item.pop("_audio_source", None)
            is_temp = item.pop("_audio_is_temp", False)
            if is_temp and src_audio:
                _rm(src_audio)

    _save_ingest_index(series_dirs["transcripts"], index)

    if not failed:
        overall, msg = "success", "全部处理成功"
    elif not transcripts:
        overall, msg = "error", f"全部 {len(failed)} 个文件处理失败"
    else:
        overall, msg = "partial", f"部分成功: {len(transcripts)}/{len(upload_paths)}"

    dur = sum(t.get("duration_sec", 0) for t in transcripts)
    segs_count = sum(t.get("segments_count", 0) for t in transcripts)
    chapters_count = sum(t.get("chapters_count", 0) for t in transcripts)

    cer_vals = []
    for item in transcripts:
        path = item.get("transcript_path")
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    aq = json.load(fh).get("asr_quality", {})
                if "estimated_cer" in aq:
                    cer_vals.append(aq["estimated_cer"])
            except Exception:
                logger.debug(f"读取 transcript quality 失败: {path}")

    _file_handler.flush()
    _file_handler.close()
    logging.getLogger().removeHandler(_file_handler)

    warnings = list(oversized_warnings)
    for f in failed:
        warnings.append(f"{f['path']}: {f.get('reason', 'unknown')}")

    _emit(on_progress, "done", 1.0)

    return {
        "status": overall,
        "message": msg,
        "series_id": series_id,
        "transcripts": [t["transcript_path"] for t in transcripts],
        "stats": {
            "total_duration_sec": round(dur, 1),
            "segments_count": segs_count,
            "avg_segment_duration_sec": round(dur / segs_count, 1) if segs_count else 0,
            "estimated_cer": round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else 0,
            "chapters_count": chapters_count,
            "refined_files": sum(1 for t in transcripts if t.get("refined")),
            "summarized_files": sum(1 for t in transcripts if t.get("summarized")),
        },
        "warnings": warnings,
    }
