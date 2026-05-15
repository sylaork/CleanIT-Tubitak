# -*- coding: utf-8 -*-
"""
CleanIT — TÜBİTAK 2209-A
Türkçe konuşma transkripti · Temizleme · Özet · Anahtar Noktalar · Aksiyon Maddeleri
Akıllı LLM yönlendirme: bağlama göre GPT-4o / GPT-4o-mini seçimi
"""

from __future__ import annotations

import io, json, os, re, shutil, subprocess, threading, uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_file

# ── Yapılandırma ──────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
).strip()
OPENAI_KEY_OK = bool(OPENAI_API_KEY) and not OPENAI_API_KEY.startswith("BURAYA_")

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Whisper ───────────────────────────────────────────────────────────────────

import whisper

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
print(f"Whisper yükleniyor ({WHISPER_MODEL_NAME})…")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Whisper hazır.")
except Exception as _e:
    print(f"  {WHISPER_MODEL_NAME} yüklenemedi, small deneniyor… ({_e})")
    whisper_model = whisper.load_model("small")

# ── ffmpeg yardımcıları ───────────────────────────────────────────────────────

HAS_FFMPEG  = bool(shutil.which("ffmpeg"))
HAS_FFPROBE = bool(shutil.which("ffprobe"))

def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout
    except Exception as e:
        return 999, str(e)

def _duration(path: str) -> Optional[float]:
    if not HAS_FFPROBE:
        return None
    code, out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=nw=1:nk=1", path])
    try:
        return float(out.strip()) if code == 0 else None
    except Exception:
        return None

def _stream_duration(path: str) -> float:
    code, out = _run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                      "-show_entries", "stream=duration",
                      "-of", "default=nw=1:nk=1", path])
    try:
        v = float(out.strip())
        return v if v > 0 else 0.0
    except Exception:
        return _duration(path) or 0.0

def _valid(path: str, min_sec: float = 0.3) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) < 500:
        return False
    dur = _duration(path)
    return dur >= min_sec if dur is not None else True

def normalize_wav(input_path: str, out_dir: str) -> str:
    if not HAS_FFMPEG:
        return input_path
    out = os.path.join(out_dir, "norm.wav")
    # 1. loudnorm
    _run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn",
          "-af", "loudnorm=I=-16:LRA=11:TP=-1.5", out])
    if _valid(out): return out
    # 2. sade dönüşüm
    _run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn",
          "-acodec", "pcm_s16le", out])
    if _valid(out): return out
    return input_path

def split_chunks(wav_path: str, out_dir: str, chunk_sec: int = 300) -> List[str]:
    if not HAS_FFMPEG:
        return [wav_path]
    dur = _duration(wav_path)
    if dur is None or dur <= chunk_sec + 5:
        return [wav_path]
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    code, _ = _run(["ffmpeg", "-y", "-i", wav_path, "-f", "segment",
                    "-segment_time", str(chunk_sec), "-c", "copy",
                    os.path.join(chunk_dir, "chunk_%03d.wav")])
    if code != 0:
        return [wav_path]
    chunks = sorted(
        f for f in (os.path.join(chunk_dir, n) for n in os.listdir(chunk_dir)
                    if n.endswith(".wav"))
        if _valid(f)
    )
    return chunks or [wav_path]

# ── Transkripsiyon ────────────────────────────────────────────────────────────

def _transcribe_chunk(path: str) -> Tuple[str, list]:
    params = dict(language="tr", fp16=False, condition_on_previous_text=True,
                  no_speech_threshold=0.5, logprob_threshold=-1.0,
                  compression_ratio_threshold=2.4,
                  initial_prompt="Bu bir Türkçe konuşma metnidir.")
    try:
        r = whisper_model.transcribe(path, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), **params)
        return (r.get("text") or "").strip(), r.get("segments") or []
    except Exception as e1:
        print(f"  Whisper hata ({e1}), minimal mod…")
    try:
        r = whisper_model.transcribe(path, temperature=0.0, language="tr", fp16=False,
                                     condition_on_previous_text=False)
        return (r.get("text") or "").strip(), r.get("segments") or []
    except Exception as e2:
        print(f"  Minimal mod da başarısız ({e2})")
        return "", []

def transcribe(input_path: str) -> Tuple[str, list]:
    work = os.path.join(UPLOAD_DIR, f"work_{uuid.uuid4().hex[:8]}")
    os.makedirs(work, exist_ok=True)
    try:
        norm = normalize_wav(input_path, work)
        if not _valid(norm):
            raise RuntimeError("Ses dosyası işlenemedi veya boş. "
                               "Desteklenen formatlar: MP3, WAV, M4A, OGG, WEBM.")
        parts, segs = [], []
        for i, ch in enumerate(split_chunks(norm, work)):
            dur = _stream_duration(ch)
            print(f"Parça {i+1}: {dur:.1f}s")
            if dur < 1.0:
                print("  Çok kısa, atlandı.")
                continue
            txt, seg = _transcribe_chunk(ch)
            if txt: parts.append(txt)
            segs.extend(seg)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    full = "\n".join(parts).strip()
    if not full:
        raise RuntimeError("Transkript üretilemedi. Ses dosyasında konuşma bulunamadı.")
    return full, segs

# ── Metin temizleme ───────────────────────────────────────────────────────────

_FILLERS = [
    "ıı","eee","aaa","mmm","hmm","üh","ühh","şey","yani","hani","işte",
    "böyle","öyle","falan","filan","ya","yaa","neyse","tamam","peki","tabii","tabi",
    "mesela","örneğin","şimdi","sonra","önce","ne bileyim","nasıl desem",
]

def clean_text(text: str) -> str:
    if not text: return ""
    t = text
    for f in _FILLERS:
        t = re.sub(rf"\b{re.escape(f)}\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    parts = re.split(r"([.!?]+)", t)
    sents = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i].strip()
        p = parts[i + 1] if i + 1 < len(parts) else "."
        if not s: continue
        sents.append((s[0].upper() + s[1:] if len(s) > 1 else s.upper()) + p)
    t = " ".join(sents)

    final = [s.strip() for s in re.split(r"[.!?]+", t)
             if len(s.strip()) >= 18 and len(s.strip().split()) >= 4]
    return ". ".join(final) + ("." if final else "")

# ── Bağlam tespiti ────────────────────────────────────────────────────────────

_CTX_KW = {
    "teknik":   (["algoritma","yazılım","kod","veri","sistem","api","deploy","debug","model"],
                 ["class","commit","bug","repo","cloud","frontend","backend"]),
    "akademik": (["araştırma","tez","makale","metodoloji","analiz","bulgu","hipotez","deney"],
                 ["kaynak","referans","konferans","jüri","danışman"]),
    "hukuki":   (["kanun","yasa","mahkeme","dava","hukuk","sözleşme","karar","yargı"],
                 ["dilekçe","tanık","delil","savcı","hakim","duruşma"]),
    "tibbi":    (["hasta","tedavi","hastalık","doktor","ilaç","tanı","sağlık","kalp","hastane"],
                 ["semptom","reçete","röntgen","tahlil","enjeksiyon"]),
    "is":       (["toplantı","proje","müşteri","strateji","hedef","bütçe","rapor","deadline"],
                 ["kpi","roi","pazar","marka","yönetici","performans"]),
    "egitim":   (["ders","öğrenci","öğretmen","sınav","ödev","konu","cevap","müfredat"],
                 ["okul","sınıf","kurs","mezuniyet"]),
}

def detect_context(text: str) -> Tuple[str, Dict[str, int]]:
    t = (text or "").lower()
    scores = {}
    for ctx, (primary, secondary) in _CTX_KW.items():
        s  = sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) * 3 for w in primary)
        s += sum(len(re.findall(rf"\b{re.escape(w)}\b", t))     for w in secondary)
        scores[ctx] = s
    if not scores or max(scores.values()) == 0:
        return "genel", scores
    best = max(scores, key=scores.get)
    return (best if scores[best] >= 2 else "genel"), scores

# ── Model seçimi (Akıllı LLM Yönlendirme) ────────────────────────────────────

_DEFAULT_SYSTEM = ("Sen profesyonel bir içerik analiz asistanısın. "
                   "Türkçe konuşma transkriptlerinden özet, anahtar noktalar "
                   "ve aksiyon maddeleri üretirsin.")

_MODEL_MAP = {
    "teknik":   ("gpt-4o",      "GPT-4o · Teknik",
                 "Teknik analiz için gpt-4o seçildi.",
                 "Sen teknik konularda uzman bir analiz asistanısın. "
                 "Yazılım ve sistem içeriklerini net ve doğru özetle."),
    "akademik": ("gpt-4o",      "GPT-4o · Akademik",
                 "Akademik içerik için gpt-4o seçildi.",
                 "Sen akademik içerikleri analiz eden bir asistanısın. "
                 "Bilgiyi düzenli ve maddeli sun."),
    "hukuki":   ("gpt-4o",      "GPT-4o · Hukuki",
                 "Hukuki metinler için gpt-4o seçildi.",
                 "Sen hukuki içerikleri özetleyen bir asistanısın. "
                 "Kesin hüküm verme; metinden çıkarım yap."),
    "tibbi":    ("gpt-4o",      "GPT-4o · Tıbbi",
                 "Tıbbi doğruluk için gpt-4o seçildi.",
                 "Sen tıbbi konuşmaları özetleyen bir asistanısın. "
                 "Teşhis koyma. Son cümle: 'Bu tıbbi tavsiye değildir.'"),
    "is":       ("gpt-4o-mini", "GPT-4o mini · İş",
                 "Toplantı notları için gpt-4o-mini seçildi.",
                 "Sen toplantı notlarını analiz eden bir asistanısın. "
                 "Somut kararlar ve görev atamalarını öne çıkar."),
    "egitim":   ("gpt-4o-mini", "GPT-4o mini · Eğitim",
                 "Eğitim içeriği için gpt-4o-mini seçildi.",
                 "Sen eğitim içeriklerini analiz eden bir asistanısın. "
                 "Anahtar kavramları ve öğrenme noktalarını vurgula."),
    "genel":    ("gpt-4o-mini", "GPT-4o mini · Genel",
                 "Genel içerik için gpt-4o-mini seçildi.",
                 _DEFAULT_SYSTEM),
}

def select_model(context: str, scores: Dict[str, int]) -> Dict[str, Any]:
    mid, label, reason, system = _MODEL_MAP.get(context, _MODEL_MAP["genel"])
    return {"model_id": mid, "label": label, "reason": reason,
            "system": system, "confidence_scores": scores}

# ── Yerel analiz (API yokken) ─────────────────────────────────────────────────

_STOP = set("bir ve veya bu şu o ile için gibi kadar daha çok az en ancak ama "
            "de da ki mi mu mü mı".split())

_CONNECTOR = re.compile(
    r"^(çünkü|yani|ama|fakat|ancak|aslında|zaten|tabii|tabi|mesela|örneğin|"
    r"hani|işte|böyle|öyle|ve\s|veya\s|dediğim|biliyorsunuz|öte\s+yandan|"
    r"bununla|bu\s+nedenle|bu\s+yüzden|ayrıca)\b", re.IGNORECASE)

_INFO = re.compile(
    r"\d+|yüzde|tarih|karar|sonuç|hedef|önemli|kritik|temel|belirlendi|"
    r"planlandı|kararlaştırıldı|gerekli|zorunlu|öncelikli", re.IGNORECASE)

_ACTION_VERBS = re.compile(
    r"^(hazırla|gönder|ilet|araştır|incele|kontrol|takip|tamamla|başlat|"
    r"uygula|düzenle|güncelle|paylaş|değerlendir|planla|organize|başvur|"
    r"bildir|raporla|çalış|öğren|izle|oku|dene|yap|al|katıl|sun|geliştir|"
    r"indir|kur|kaydet|çöz|düzelt|ekle|oluştur|tasarla|yaz|hesapla|seç|"
    r"belirle|test|revize|tekrarla|pratiğ)\w*\b", re.IGNORECASE)

_MUST_DO = re.compile(
    r"\b\w+(malı|meli)(sın|sin|yız|yiz|ım|im|)?\b"
    r"|\b(gerekiyor|lazım|şart|yapılmalı|edilmeli|olmalı)\b", re.IGNORECASE)

_FINDING = re.compile(
    r"\b(belirlendi|tespit\s+edildi|görüldü|saptandı|bulundu|anlaşıldı|"
    r"bilinmektedir|belirtilmiştir|açıklandı|ifade\s+edildi)\b", re.IGNORECASE)

_BAD_START = re.compile(
    r"^(çünkü|yani|ama|fakat|ancak|aslında|öte|bununla|bu\s+nedenle|"
    r"bu\s+yüzden|dolayısıyla|zaten|dediğim|biliyorsunuz|tabii|tabi|"
    r"mesela|örneğin|hani|işte|böyle|öyle|şey\s|ve\s|veya\s)", re.IGNORECASE)

def _ctx_boost(context: str) -> List[str]:
    return {
        "teknik":   ["sistem","api","veri","model","hata","algoritma","kod","performans"],
        "akademik": ["araştırma","bulgu","analiz","sonuç","yöntem","hipotez","deney"],
        "tibbi":    ["tedavi","tanı","hasta","ilaç","risk","doktor","belirti"],
        "is":       ["hedef","proje","müşteri","bütçe","deadline","strateji","karar"],
        "egitim":   ["konu","ders","beceri","kavram","uygulama","ödev","tutorial"],
    }.get(context, [])

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))

def local_summary(text: str, context: str, n: int = 5) -> str:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) >= 5]
    if not sents: return "Özetlenecek yeterli metin bulunamadı."
    freq  = Counter(w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOP)
    boost = _ctx_boost(context)
    scored = []
    for i, s in enumerate(sents):
        w  = re.findall(r"\b\w+\b", s.lower())
        sc = sum(freq.get(x, 0) for x in w if x not in _STOP)
        if any(b in s.lower() for b in boost): sc += 6
        if re.search(r"\d+", s): sc += 2
        if 12 <= len(w) <= 28:   sc += 3
        scored.append((i, s, sc))
    scored.sort(key=lambda x: x[2], reverse=True)
    top = sorted(scored[:min(n, len(scored))], key=lambda x: x[0])
    return "\n\n".join(t[1] for t in top)

def local_key_points(text: str, context: str, n: int = 5) -> List[str]:
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
             and 6 <= len(s.split()) <= 40 and not _CONNECTOR.search(s.strip())]
    if not sents: return []
    freq  = Counter(w for w in re.findall(r"\b\w+\b", text.lower())
                    if w not in _STOP and len(w) > 2)
    boost = _ctx_boost(context)
    scored = []
    for s in sents:
        w  = re.findall(r"\b\w+\b", s.lower())
        sc = sum(freq.get(x, 0) for x in w if x not in _STOP and len(x) > 2)
        if _INFO.search(s):              sc += 6
        if any(b in s.lower() for b in boost): sc += 4
        if 10 <= len(w) <= 25:           sc += 3
        scored.append((s, sc))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = []
    for s, _ in scored:
        if len(picked) >= n: break
        if any(_jaccard(s, p) > 0.5 for p in picked): continue
        picked.append(s if s.endswith(".") else s + ".")
    return picked

def local_actions(text: str, context: str, n: int = 6) -> List[str]:
    sents = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    boost = _ctx_boost(context)
    candidates = []
    for s in sents:
        sl, wc = s.lower(), len(s.split())
        if wc < 3 or wc > 25: continue
        if s.endswith("?"): continue
        if _BAD_START.search(sl): continue
        if _FINDING.search(sl): continue
        sc = 0
        if _ACTION_VERBS.search(sl): sc += 10
        if _MUST_DO.search(sl):      sc += 6
        if sc == 0: continue
        if re.search(r"\d+", sl): sc += 2
        if 4 <= wc <= 15:         sc += 3
        if any(b in sl for b in boost): sc += 3
        candidates.append((s, sc))
    candidates.sort(key=lambda x: x[1], reverse=True)
    picked = []
    for s, _ in candidates:
        if len(picked) >= n: break
        if any(_jaccard(s, p) > 0.6 for p in picked): continue
        txt = (s[0].upper() + s[1:]) if s else s
        picked.append(txt if txt.endswith(".") else txt + ".")
    return picked

# ── OpenAI API ────────────────────────────────────────────────────────────────

def _openai(model_id: str, system: str, messages: List[Dict],
            temperature: float = 0.1, max_tokens: int = 1600) -> str:
    import requests
    if not OPENAI_KEY_OK:
        raise RuntimeError("OpenAI API key eksik.")
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": model_id, "temperature": temperature, "max_tokens": max_tokens,
              "messages": [{"role": "system", "content": system}] + messages},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def _sample_text(text: str, max_chars: int = 7500) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return text[:max_chars] + "\n[…kısaltıldı]", True
    avg   = sum(len(s) for s in sents) / len(sents)
    keep  = max(10, int(max_chars / max(avg, 1)))
    if keep >= len(sents):
        return text[:max_chars] + "\n[…kısaltıldı]", True
    step  = len(sents) / keep
    idx   = sorted({0} | {int(i * step) for i in range(1, keep - 1)} | {len(sents) - 1})
    result = " ".join(sents[i] for i in idx if i < len(sents))
    return (result if len(result) <= max_chars else result[:max_chars] + "\n[…kısaltıldı]"), True

def _parse_json(raw: str) -> Optional[Dict]:
    for attempt in (raw,
                    (re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL) or type("", (), {"group": lambda *_: None})()).group(1),
                    (re.search(r"\{.*\}", raw, re.DOTALL) or type("", (), {"group": lambda *_: None})()).group(0)):
        if not attempt: continue
        try: return json.loads(attempt)
        except Exception: pass
    return None

def llm_analyze(model_id: str, system: str, text: str, summary_len: str) -> Dict[str, Any]:
    hint     = {"short": "3-4 cümle", "medium": "5-7 cümle", "long": "8-12 cümle"}.get(summary_len, "5-7 cümle")
    llm_text, was_cut = _sample_text(text)
    cut_note = "\n\nNOT: Metin uzun olduğundan temsili bölümler alındı." if was_cut else ""

    prompt = f"""Aşağıdaki Türkçe konuşma transkriptini analiz et.{cut_note}

SADECE aşağıdaki JSON formatında yanıt ver:

{{
  "summary": "<{hint} uzunluğunda, konuşmanın özünü aktaran paragraf>",
  "key_points": [
    "<Konuşmada söylenen önemli bir bilgi veya karar — tek cümle>",
    "<...4-6 madde>"
  ],
  "action_items": [
    "<Emir kipiyle başlayan somut görev — örnek: 'Tutorial videoları izle.'>",
    "<...3-6 madde>"
  ]
}}

action_items KURALLARI:
- Emir kipiyle başlar: İzle, Çalış, Hazırla, Oku, Araştır, Dene, Uygula, Tekrarla…
- Somut ve yapılabilir olmalı
- YASAK: "Çünkü…", "Zaten…" ile başlayan açıklama/gözlem cümleleri

key_points KURALLARI:
- Konuşmada gerçekten söylenen bilgi veya karar
- Bağlaç cümleleriyle (Çünkü, Yani, Ama) başlamaz

METİN:
\"\"\"{llm_text}\"\"\"
""".strip()

    raw  = _openai(model_id, system, [{"role": "user", "content": prompt}])
    data = _parse_json(raw)
    if data is None:
        return {"summary": raw.strip(), "key_points": [], "action_items": []}

    summary      = str(data.get("summary", "")).strip()
    key_points   = [str(x).strip() for x in (data.get("key_points") or []) if str(x).strip()]
    action_items = [str(x).strip() for x in (data.get("action_items") or []) if str(x).strip()]

    clean_actions = [a for a in action_items
                     if not a.endswith("?") and not _FINDING.search(a.lower())
                     and not _BAD_START.search(a.lower())]

    return {"summary": summary, "key_points": key_points, "action_items": clean_actions}

# ── Chatbot ───────────────────────────────────────────────────────────────────

def chat_with_llm(model_id: str, system: str, context_text: str,
                  history: List[Dict], question: str) -> str:
    msgs = [{"role": h["role"], "content": h["content"]}
            for h in (history or [])[-10:]
            if h.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": f"{context_text}\n\nSORU: {question}"})
    return _openai(model_id, system, msgs, temperature=0.3, max_tokens=900)

def fallback_chat(question: str, transcript: str, context: str, summary: str) -> str:
    q = question.lower()
    if any(k in q for k in ["özet","kısaca","genel","ne anlat","ne bahset"]):
        return summary or local_summary(transcript, context, 4)
    stop = set("ne nasıl neden kim hangi kaç bir ve veya bu şu o ile için gibi mi".split())
    qw   = [w for w in re.findall(r"\b\w+\b", q) if len(w) >= 3 and w not in stop]
    sents = [s.strip() for s in re.split(r"[.!?]+", transcript) if len(s.strip()) > 12]
    if not sents or not qw:
        return local_summary(transcript, context, 4)
    ranked = sorted(sents, reverse=True,
                    key=lambda s: sum(4 for w in qw if re.search(rf"\b{re.escape(w)}\b", s.lower())))
    top = [s for s in ranked[:3] if any(re.search(rf"\b{re.escape(w)}\b", s.lower()) for w in qw)]
    result = ". ".join(top).strip()
    return (result + ".") if result and not result.endswith(".") else result or local_summary(transcript, context, 4)

# ── Asenkron iş kuyruğu ───────────────────────────────────────────────────────

JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

def _job_set(job_id: str, **kw) -> None:
    with _LOCK:
        JOBS[job_id].update(kw)

def _run_job(job_id: str, audio_path: str, settings: Dict[str, Any]) -> None:
    try:
        _job_set(job_id, status="running")
        result = pipeline(audio_path, settings, job_id)
        if "error" in result:
            _job_set(job_id, status="error", error=result["error"])
        else:
            _job_set(job_id, status="done", result=result)
    except Exception as e:
        _job_set(job_id, status="error", error=str(e))
    finally:
        try: os.remove(audio_path)
        except Exception: pass

# ── Ana pipeline ──────────────────────────────────────────────────────────────

def pipeline(audio_path: str, settings: Dict[str, Any],
             job_id: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "raw_transcript": "", "filtered_text": "", "context": "",
        "context_scores": {}, "selected_model": {}, "summary": "",
        "key_points": [], "action_items": [], "warnings": [],
    }
    try:
        if job_id: _job_set(job_id, step="asr",     step_label="Ses tanıma (Whisper)…")
        transcript, segments = transcribe(audio_path)
        out["raw_transcript"] = transcript
        out["segments"]       = segments

        if job_id: _job_set(job_id, step="filter",  step_label="Metin temizleme…")
        filtered = clean_text(transcript)
        out["filtered_text"] = filtered

        if job_id: _job_set(job_id, step="context", step_label="Bağlam tespiti…")
        ctx, scores = detect_context(filtered or transcript)
        out["context"]        = ctx
        out["context_scores"] = scores

        if job_id: _job_set(job_id, step="model",   step_label="Model seçimi…")
        model_info = select_model(ctx, scores)
        out["selected_model"] = model_info

        if job_id: _job_set(job_id, step="smart",   step_label="Özet & aksiyon üretimi…")
        work = filtered or transcript
        cnt  = {"short": 3, "medium": 5, "long": 8}.get(settings.get("summary_length", "medium"), 5)

        try:
            if not OPENAI_KEY_OK:
                raise RuntimeError("API key yok.")
            analysis = llm_analyze(model_info["model_id"], model_info["system"],
                                   work, settings.get("summary_length", "medium"))
            out["used_llm"] = True
        except Exception as e:
            out["warnings"].append(f"ChatGPT kullanılamadı, yerel analiz devrede: {e}")
            analysis = {
                "summary":      local_summary(work, ctx, cnt),
                "key_points":   local_key_points(work, ctx, 5),
                "action_items": local_actions(work, ctx, 6),
            }
            out["used_llm"] = False

        out["summary"]      = analysis.get("summary", "")
        out["key_points"]   = analysis.get("key_points", []) or []
        out["action_items"] = analysis.get("action_items", []) or []
        return out

    except Exception as e:
        out["error"] = str(e)
        return out

# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index_v8.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "whisper": WHISPER_MODEL_NAME,
                    "ffmpeg": HAS_FFMPEG, "openai": OPENAI_KEY_OK,
                    "openai_key_ok": OPENAI_KEY_OK,
                    "time": datetime.now().isoformat()})

@app.route("/api/process", methods=["POST"])
def api_process():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "Ses dosyası bulunamadı."}), 400
        f        = request.files["audio"]
        settings = json.loads(request.form.get("settings", "{}"))
        ext      = os.path.splitext(f.filename or "")[1] or \
                   (".webm" if "webm" in (f.mimetype or "") else ".wav")
        path = os.path.join(UPLOAD_DIR, f"audio_{uuid.uuid4().hex}{ext}")
        f.save(path)

        job_id = uuid.uuid4().hex
        with _LOCK:
            JOBS[job_id] = {"status": "queued", "step": "", "step_label": "",
                            "result": None, "error": None}
        threading.Thread(target=_run_job, args=(job_id, path, settings), daemon=True).start()
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "İş bulunamadı."}), 404
    resp = {"status": job["status"], "step": job.get("step", ""),
            "step_label": job.get("step_label", "")}
    if job["status"] == "done":  resp["result"] = job["result"]
    if job["status"] == "error": resp["error"]  = job.get("error", "Hata")
    return jsonify(resp)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.get_json(force=True, silent=True) or {}
    msg     = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    ctx      = data.get("context", "genel")
    trans    = data.get("transcript", "")
    summary  = data.get("summary", "")
    kp       = data.get("key_points") or []
    actions  = data.get("action_items") or []
    history  = data.get("history") or []
    minfo    = data.get("selected_model") or select_model(ctx, {})

    parts = []
    if summary:  parts.append(f"ÖZET:\n{summary}")
    if kp:       parts.append("ANAHTAR NOKTALAR:\n" + "\n".join(f"- {k}" for k in kp[:6]))
    if actions:  parts.append("AKSİYONLAR:\n" + "\n".join(f"- {a}" for a in actions[:8]))
    if trans:    parts.append(f"TRANSKRİPT (kısaltılmış):\n{trans[:4000]}")
    ctx_text = "\n\n".join(parts)

    try:
        reply = chat_with_llm(minfo.get("model_id", "gpt-4o-mini"),
                              minfo.get("system", _DEFAULT_SYSTEM),
                              ctx_text, history, msg)
        return jsonify({"response": reply, "used": {"mode": "llm"}})
    except Exception as e:
        return jsonify({"response": fallback_chat(msg, trans, ctx, summary),
                        "used": {"mode": "fallback", "reason": str(e)}})

@app.route("/api/download-report", methods=["POST"])
def api_download():
    try:
        data  = request.get_json(force=True) or {}
        model = data.get("selected_model") or {}
        lines = [
            "=" * 64, "  CleanIT — Konuşma Analiz Raporu", "=" * 64,
            f"  Tarih  : {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            f"  Bağlam : {(data.get('context') or '').upper()}",
            f"  Model  : {model.get('label', '')}",
            "", "─" * 64, "  HAM TRANSKRİPT", "─" * 64,
            data.get("raw_transcript", ""),
            "", "─" * 64, "  TEMİZLENMİŞ METİN", "─" * 64,
            data.get("filtered_text", ""),
            "", "─" * 64, "  ÖZET", "─" * 64,
            data.get("summary", ""),
            "", "─" * 64, "  ANAHTAR NOKTALAR", "─" * 64,
        ]
        for k in (data.get("key_points") or []):
            lines.append(f"  • {k}")
        lines += ["", "─" * 64, "  AKSİYON MADDELERİ", "─" * 64]
        for i, a in enumerate(data.get("action_items") or [], 1):
            lines.append(f"  {i}. {a}")
        lines += ["", "=" * 64]

        buf = io.BytesIO("\n".join(lines).encode("utf-8"))
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f"cleanit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                         mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Başlatma ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  CleanIT  |  http://localhost:3030")
    print(f"  Whisper  : {WHISPER_MODEL_NAME}")
    print(f"  OpenAI   : {'bağlı' if OPENAI_KEY_OK else 'yerel mod (API key yok)'}")
    print(f"{'='*50}\n")
    app.run(debug=True, host="0.0.0.0", port=3030)
