"""
Gradio helper for VoiceQC.
Pipeline: 音频上传/录音 → Qwen3-ASR → sentiment_analyzer → 质检报告展示
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
from scipy.io.wavfile import write as wav_write

from sentiment_analyzer import analyze

# ── 音频工具 ──────────────────────────────────────────────────────────────────

def _normalize_audio(wav) -> np.ndarray:
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    else:
        y = x.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def _audio_to_wav_file(audio: Any) -> Optional[str]:
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        sr, wav_data = (a0, a1) if isinstance(a0, int) else (a1, a0)
        wav_f32 = _normalize_audio(wav_data)
        wav_i16 = (wav_f32 * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav_write(tmp.name, sr, wav_i16)
        return tmp.name
    if isinstance(audio, str) and Path(audio).exists():
        return audio
    return None


# ── 示例文本（测试用，模拟 ASR 输出） ─────────────────────────────────────────

EXAMPLE_TEXTS = [
    # 强投诉场景
    "您好，我是客服，请问有什么可以帮您？我的快递等了整整一周还没到，这什么物流！非常抱歉给您带来不便，我立刻为您查询。你们每次都这么说，根本没用，我要投诉你们！您的反馈我们高度重视，已为您提交加急处理，24小时内专员联系您。好吧，希望这次能真正解决。",
    # 退款场景
    "你好请问是客服吗？是的，有什么需要帮助的？我买的手机壳收到就破损了，质量太差，要退款。非常抱歉，请提供一下订单号，我为您申请退款。订单号是1234567890。好的，已为您提交退款申请，预计三个工作日到账。谢谢，这次处理还挺快的，满意。",
    # 支付问题场景
    "你好，我的账单被多扣了一百块，这是怎么回事？您好，请稍等，我为您核查一下。等了好久了，你们到底什么时候处理？核查结果显示确实多扣了，我们将在两个工作日内退还到您账户。好的，那我等你们退款，态度还行。",
]


# ── 主界面构建函数 ────────────────────────────────────────────────────────────

def make_demo(asr_model):
    """
    构建 VoiceQC Gradio 界面。

    Args:
        asr_model: OVQwen3ASRModel 实例（来自 lab2）
    """

    def run_pipeline(audio, lang, manual_text, use_manual,
                     progress=gr.Progress(track_tqdm=True)):
        """核心流程：ASR → 情绪分析 → 报告生成。"""
        t_start = time.time()

        # Step 1: 获取转录文本
        if use_manual and manual_text.strip():
            transcript = manual_text.strip()
            asr_time = 0.0
            detected_lang = '手动输入'
        else:
            if audio is None:
                err = '⚠️ 请上传录音文件或录制音频'
                return err, '', '', '', err
            wav_path = _audio_to_wav_file(audio)
            if wav_path is None:
                err = '⚠️ 音频格式不支持，请上传 WAV/MP3 文件'
                return err, '', '', '', err

            t0 = time.time()
            language = None if lang == 'Auto' else lang
            results  = asr_model.transcribe(audio=wav_path, language=language)
            asr_time = time.time() - t0
            transcript    = results[0].text if results else ''
            detected_lang = getattr(results[0], 'language', lang) if results else lang

        if not transcript:
            return '转录结果为空，请检查音频内容', '', '', '', ''

        # Step 2: 情绪分析
        report = analyze(transcript)

        total_time = time.time() - t_start

        # 情绪统计摘要（用于单行显示框）
        total_sent = len(report.sentences)
        emotion_summary = (
            f'😠 愤怒 {report.angry_count}句  '
            f'😰 焦虑 {report.anxious_count}句  '
            f'😊 满意 {report.satisfied_count}句  '
            f'😐 中性 {report.neutral_count}句  |  '
            f'综合满意度: {report.satisfaction_label}'
        )

        metrics = (
            f'语言: {detected_lang}  |  '
            f'ASR: {asr_time:.1f}s  |  '
            f'总计: {total_time:.1f}s  |  '
            f'共 {total_sent} 句'
        )

        return (
            transcript,
            report.annotated_text,
            emotion_summary,
            report.summary_md,
            metrics,
        )

    def save_report_file(report_md: str):
        """将 Markdown 报告保存为临时文件供下载。"""
        if not report_md or not report_md.strip():
            return None
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix='.md', mode='w', encoding='utf-8'
        )
        tmp.write(report_md)
        tmp.flush()
        return tmp.name

    # ── UI 布局 ───────────────────────────────────────────────────────────────

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont('Source Sans Pro'), 'Arial', 'sans-serif']
    )

    css = """
    .metric-text { font-family: monospace; font-size: 0.85em; }
    .annotated   { font-family: monospace; white-space: pre; }
    """

    with gr.Blocks(theme=theme, css=css, title='VoiceQC 客服语音质检助手') as demo:

        gr.Markdown("""
# 🎙️ VoiceQC：客服语音智能质检助手
**Intel AI PC 端侧推理 | Qwen3-ASR + OpenVINO | 转录 · 情绪标记 · 质检报告**

上传客服录音，AI 自动完成：**语音转录 → 逐句情绪标注 → 问题识别 → 质检报告**。
全程本地运行，通话内容不离开设备，保护客户隐私。
""")

        with gr.Row():

            # 左列：输入
            with gr.Column(scale=1):
                gr.Markdown('#### 📤 输入')
                audio_in = gr.Audio(
                    label='上传客服录音 / 麦克风录音',
                    type='numpy',
                    sources=['upload', 'microphone'],
                )
                lang_dd = gr.Dropdown(
                    label='语言（留 Auto 自动检测）',
                    choices=['Auto', 'Chinese', 'English'],
                    value='Chinese',
                )
                with gr.Accordion('💬 或直接粘贴转录文本（调试用）', open=False):
                    manual_text = gr.Textbox(
                        label='直接输入客服对话文本',
                        lines=6,
                        placeholder='客服：您好，请问有什么可以帮您？\n客户：我的快递等了一周还没到……',
                    )
                    use_manual = gr.Checkbox(label='使用文字输入（跳过 ASR）', value=False)

                analyze_btn = gr.Button('🔍 开始质检分析', variant='primary', size='lg')
                metrics_box = gr.Textbox(
                    label='推理指标', interactive=False, elem_classes='metric-text'
                )

                gr.Markdown('#### 💡 示例对话（点击后勾选"使用文字输入"）')
                gr.Examples(
                    examples=[[t] for t in EXAMPLE_TEXTS],
                    inputs=[manual_text],
                    label=None,
                )

            # 右列：输出
            with gr.Column(scale=1):
                gr.Markdown('#### 📊 分析结果')
                transcript_box = gr.Textbox(
                    label='ASR 转录原文', lines=5, interactive=False
                )
                emotion_box = gr.Textbox(
                    label='情绪统计', lines=2, interactive=False
                )
                annotated_box = gr.Textbox(
                    label='逐句情绪标注', lines=10, interactive=False,
                    elem_classes='annotated',
                )

        # 完整报告（全宽）
        gr.Markdown('---')
        gr.Markdown('#### 📋 完整质检报告')
        report_display = gr.Markdown()

        with gr.Row():
            download_btn  = gr.Button('⬇️ 下载报告（Markdown）')
            download_file = gr.File(label='报告文件', visible=False)

        # ── 事件绑定 ──────────────────────────────────────────────────────────

        analyze_btn.click(
            run_pipeline,
            inputs=[audio_in, lang_dd, manual_text, use_manual],
            outputs=[transcript_box, annotated_box, emotion_box, report_display, metrics_box],
        )

        download_btn.click(
            save_report_file,
            inputs=[report_display],
            outputs=[download_file],
        ).then(lambda: gr.update(visible=True), outputs=[download_file])

    return demo
