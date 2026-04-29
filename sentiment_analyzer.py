"""
Sentiment Analyzer: 客服通话文本情绪分析器
核心创新点：在 Qwen3-ASR 转录输出基础上，进行逐句情绪标注与质检报告生成。

分析维度：
  - 逐句情绪分类（愤怒 / 焦虑 / 满意 / 轻微不满 / 中性）
  - 问题类别检测（物流 / 质量 / 退款 / 态度 / 账号 / 支付 / 商品）
  - 解决状态识别
  - 客服/客户说话人启发式区分
  - 综合满意度评分（-1.0 ~ 1.0）
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── 情绪关键词词典 ────────────────────────────────────────────────────────────

ANGRY_KEYWORDS = [
    # 投诉举报
    '投诉', '举报', '曝光', '315', '消费者协会', '媒体',
    # 欺诈
    '骗', '欺骗', '虚假', '假冒', '货不对板',
    # 强烈不满
    '不满意', '非常不满', '太差了', '很差', '差劲', '垃圾服务',
    '什么态度', '什么服务', '这叫服务', '太过分', '无法接受', '忍无可忍',
    # 退款索赔
    '退款', '退货', '赔偿', '索赔', '要求退', '必须退',
    # 失职
    '一直没', '还没解决', '根本没用', '不负责', '敷衍',
    '等了多久', '这么久了', '到底什么时候',
    # 情绪词
    '气死', '生气', '愤怒', '崩溃', '受够了',
    # 威胁
    '找你们领导', '投诉你们', '告你们',
]

ANXIOUS_KEYWORDS = [
    '很急', '非常急', '紧急', '急死了', '着急',
    '等了好久', '等了很久', '等了好几天', '等了一周', '等了半个月',
    '什么时候', '还要多久', '多长时间', '几天能到', '何时',
    '今天必须', '马上要用', '明天要用', '急需',
    '迟迟', '一直没收到', '怎么还没', '催一下',
]

SATISFIED_KEYWORDS = [
    '谢谢', '感谢', '非常感谢', '太感谢了', '谢谢你', '谢谢您',
    '很满意', '挺好的', '不错', '很好', '非常好', '满意', '挺满意',
    '解决了', '处理好了', '问题解决', '好多了', '放心了',
    '您辛苦了', '辛苦了', '态度很好', '服务很好', '服务不错',
    '明白了', '了解了', '清楚了', '好的好的', '可以的',
    '理解', '支持', '继续购买', '下次还来',
]

NEGATIVE_WEAK = [
    '不太好', '有点问题', '不是很满意', '一般般', '还行吧',
    '有些慢', '等了一会', '稍微有点', '略有', '不够好',
]

# ── 问题分类关键词 ────────────────────────────────────────────────────────────

ISSUE_CATEGORIES = {
    '物流/配送问题': ['快递', '物流', '配送', '发货', '没收到', '包裹', '运输', '快递员', '揽收', '派送'],
    '商品质量问题': ['质量', '坏了', '破损', '损坏', '有问题', '不能用', '故障', '瑕疵', '缺陷', '次品'],
    '退款/退货问题': ['退款', '退钱', '退货', '换货', '申请退', '退了吗', '退款进度'],
    '客服态度问题': ['态度', '不理人', '爱答不理', '敷衍', '没有回应', '不专业', '语气'],
    '账号/登录问题': ['账号', '登录', '密码', '账户', '无法登录', '被盗', '注册'],
    '支付/费用问题': ['支付', '付款', '扣款', '多扣', '没到账', '转账', '账单', '费用'],
    '商品描述问题': ['描述不符', '和图片', '假货', '仿品', '以次充好', '名不副实'],
}

# ── 客服常用开场语（用于区分说话人） ─────────────────────────────────────────

AGENT_MARKERS = [
    '您好', '你好，', '您好，', '感谢您致电', '请稍等', '为您查询',
    '为您处理', '非常抱歉', '对不起打扰', '请问您',
    '感谢您的耐心等待', '已为您', '已经为您',
]

RESOLUTION_KEYWORDS = [
    '已为您', '已经处理', '已解决', '处理完成', '已退款', '退款成功',
    '已安排', '已核实', '已提交', '会有人联系您', '为您补发',
    '问题已', '已帮您', '成功处理', '24小时内', '专员联系',
]

# ── 数据类 ────────────────────────────────────────────────────────────────────

@dataclass
class SentenceResult:
    text: str
    emotion: str           # angry / anxious / satisfied / negative_weak / neutral
    score: float           # -1.0（极负面）~ 1.0（极正面）
    keywords_hit: List[str] = field(default_factory=list)
    is_agent: bool = False

@dataclass
class AnalysisReport:
    sentences: List[SentenceResult]
    overall_score: float
    satisfaction_label: str
    issues_detected: List[str]
    resolution_detected: bool
    angry_count: int
    anxious_count: int
    satisfied_count: int
    neutral_count: int
    annotated_text: str
    summary_md: str

# ── 情绪 Emoji 映射 ───────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    'angry':         '😠 愤怒',
    'anxious':       '😰 焦虑',
    'satisfied':     '😊 满意',
    'negative_weak': '😕 轻微不满',
    'neutral':       '😐 中性',
}

# ── 核心分析函数 ──────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """按中文标点切分句子。"""
    parts = re.split(r'[。！？!?\n]+', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]


def _is_agent(sentence: str) -> bool:
    """启发式判断该句是否为客服说话人。"""
    return any(marker in sentence for marker in AGENT_MARKERS)


def analyze_sentence(sentence: str) -> SentenceResult:
    hits_angry     = [kw for kw in ANGRY_KEYWORDS     if kw in sentence]
    hits_anxious   = [kw for kw in ANXIOUS_KEYWORDS   if kw in sentence]
    hits_satisfied = [kw for kw in SATISFIED_KEYWORDS if kw in sentence]
    hits_neg_weak  = [kw for kw in NEGATIVE_WEAK      if kw in sentence]

    # 加权评分
    score = (
        len(hits_satisfied) * 0.40
        - len(hits_angry)   * 0.55
        - len(hits_anxious) * 0.30
        - len(hits_neg_weak)* 0.20
    )
    score = max(-1.0, min(1.0, score))

    # 情绪分类（优先级：愤怒 > 焦虑 > 轻微不满 > 满意 > 中性）
    if hits_angry:
        emotion = 'angry'
    elif hits_anxious and not hits_satisfied:
        emotion = 'anxious'
    elif hits_neg_weak and not hits_satisfied:
        emotion = 'negative_weak'
    elif hits_satisfied:
        emotion = 'satisfied'
    else:
        emotion = 'neutral'

    return SentenceResult(
        text=sentence,
        emotion=emotion,
        score=score,
        keywords_hit=hits_angry + hits_anxious + hits_satisfied + hits_neg_weak,
        is_agent=_is_agent(sentence),
    )


def _satisfaction_label(score: float) -> str:
    if score >= 0.5:    return '非常满意 ⭐⭐⭐⭐⭐'
    if score >= 0.15:   return '满意 ⭐⭐⭐⭐'
    if score >= -0.15:  return '中性 ⭐⭐⭐'
    if score >= -0.5:   return '不满 ⭐⭐'
    return '非常不满 ⭐'


def _build_annotated_text(sentences: List[SentenceResult]) -> str:
    lines = []
    for s in sentences:
        role   = '【客服】' if s.is_agent else '【客户】'
        label  = EMOTION_EMOJI[s.emotion]
        kw_str = f'  (命中: {", ".join(s.keywords_hit[:3])})' if s.keywords_hit else ''
        lines.append(f'{role} {s.text}  →  {label}{kw_str}')
    return '\n'.join(lines)


def _build_report_md(report: 'AnalysisReport') -> str:
    total = len(report.sentences)
    issues_str = ('、'.join(report.issues_detected)
                  if report.issues_detected else '未检测到明确问题类别')
    resolution_str = ('✅ 通话中已出现解决方案'
                      if report.resolution_detected else '❌ 未检测到明确解决结果')

    # 高风险语句
    high_risk = [s.text for s in report.sentences if s.emotion == 'angry']
    high_risk_block = ''
    if high_risk:
        items = '\n'.join(f'- {t}' for t in high_risk[:4])
        high_risk_block = f'\n\n### ⚠️ 高风险语句（需人工复核）\n{items}'

    # 建议
    suggestions = []
    if report.angry_count >= 2:
        suggestions.append('客户情绪激动，建议升级处理并安排回访。')
    if not report.resolution_detected:
        suggestions.append('问题未在本次通话中解决，建议跟进。')
    if report.overall_score < -0.3:
        suggestions.append('整体满意度偏低，建议加强客服安抚技巧培训。')
    if not suggestions:
        suggestions.append('本次通话整体表现良好，继续保持。')
    suggestions_block = '\n'.join(f'- {s}' for s in suggestions)

    return f"""# 📋 客服通话质检报告

## 总体评分
| 指标 | 结果 |
|------|------|
| **客户满意度** | {report.satisfaction_label} |
| **情绪综合分值** | {report.overall_score:.2f}（-1极差 ~ +1极好）|
| **解决状态** | {resolution_str} |
| **检测到的问题** | {issues_str} |

## 情绪分布（共 {total} 句）
| 情绪 | 句数 | 占比 |
|------|------|------|
| 😠 愤怒 | {report.angry_count} | {report.angry_count/max(total,1)*100:.0f}% |
| 😰 焦虑 | {report.anxious_count} | {report.anxious_count/max(total,1)*100:.0f}% |
| 😊 满意 | {report.satisfied_count} | {report.satisfied_count/max(total,1)*100:.0f}% |
| 😐 中性 | {report.neutral_count} | {report.neutral_count/max(total,1)*100:.0f}% |
{high_risk_block}

## 质检建议
{suggestions_block}

## 逐句情绪标注
```
{report.annotated_text}
```
"""


def analyze(transcript: str) -> AnalysisReport:
    """
    主入口：对客服通话转录文本进行全面情绪分析。

    Args:
        transcript: Qwen3-ASR 转录输出的文本

    Returns:
        AnalysisReport（包含逐句标注 + 质检报告 Markdown）
    """
    sentences  = _split_sentences(transcript)
    if not sentences:
        sentences = [transcript]

    results    = [analyze_sentence(s) for s in sentences]
    total      = len(results)

    overall    = sum(r.score for r in results) / total if total else 0.0
    overall    = max(-1.0, min(1.0, overall))

    angry_n    = sum(1 for r in results if r.emotion == 'angry')
    anxious_n  = sum(1 for r in results if r.emotion == 'anxious')
    satisfied_n= sum(1 for r in results if r.emotion == 'satisfied')
    neutral_n  = sum(1 for r in results if r.emotion in ('neutral', 'negative_weak'))

    issues     = [cat for cat, kws in ISSUE_CATEGORIES.items()
                  if any(kw in transcript for kw in kws)]
    resolution = any(kw in transcript for kw in RESOLUTION_KEYWORDS)
    annotated  = _build_annotated_text(results)

    report = AnalysisReport(
        sentences=results,
        overall_score=overall,
        satisfaction_label=_satisfaction_label(overall),
        issues_detected=issues,
        resolution_detected=resolution,
        angry_count=angry_n,
        anxious_count=anxious_n,
        satisfied_count=satisfied_n,
        neutral_count=neutral_n,
        annotated_text=annotated,
        summary_md='',
    )
    report.summary_md = _build_report_md(report)
    return report
