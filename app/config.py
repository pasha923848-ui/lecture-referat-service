import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directory where uploaded videos and extracted audio are stored temporarily.
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", BASE_DIR / "storage"))
UPLOADS_DIR = STORAGE_DIR / "uploads"
AUDIO_DIR = STORAGE_DIR / "audio"

# faster-whisper model settings. "base" runs comfortably on CPU with no GPU
# required; bump to "small"/"medium"/"large-v3" for better accuracy if you
# have the hardware for it. Everything runs fully offline after the model
# weights are downloaded once.
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE") or None  # None = auto-detect

# Beam search width during decoding — faster-whisper's own default is 5.
# Lecture transcripts don't need that much search depth to be readable;
# dropping to 1 (greedy decoding) is a large, free CPU speedup with only a
# minor accuracy cost, which matters a lot for hour-long recordings on a
# laptop CPU with no GPU.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))

# Skips silent/non-speech stretches instead of decoding them — free
# speedup on real lecture recordings, which are full of pauses, and
# doesn't touch actual speech quality since nothing there is discarded.
WHISPER_VAD_FILTER = os.environ.get("WHISPER_VAD_FILTER", "1") not in ("0", "false", "False")

# Feeding each segment's decode back in as context for the next one (the
# faster-whisper default) costs extra compute for a benefit — smoother
# sentence-to-sentence flow — that matters less for a study transcript than
# for, say, subtitles.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = os.environ.get(
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT", "0"
) not in ("0", "false", "False")

# faster-whisper retries a segment at progressively higher temperatures
# (0.0, 0.2, 0.4, ...) when its first attempt looks low-confidence — each
# retry is a full extra decode pass. A single fixed temperature of 0
# (deterministic, no retries) trades a little robustness on unclear audio
# for a real speedup; set e.g. "0.0,0.2,0.4,0.6,0.8,1.0" to restore the
# original fallback ladder.
WHISPER_TEMPERATURE = [float(t) for t in os.environ.get("WHISPER_TEMPERATURE", "0.0").split(",")]

# Threads faster-whisper's CTranslate2 backend may use per transcription.
# The model instance is shared by every worker thread (see
# transcription.get_model), and each concurrent transcription otherwise
# defaults to grabbing every available core — with MAX_WORKERS transcribing
# at once, that's MAX_WORKERS-way oversubscription of the same cores.
# Dividing the host's CPU count across the worker slots keeps them from
# fighting each other; 0 means "let CTranslate2 pick its own default".
WHISPER_CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "0"))

# Number of transcription jobs that may run at the same time. The model
# itself is loaded once and shared by every worker thread (see
# transcription.get_model), so this mostly trades CPU time between
# concurrent users rather than multiplying memory use.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))

# Total jobs allowed to be queued/running at once (including MAX_WORKERS
# already running). Beyond this, new uploads are rejected with 503 instead
# of piling up uploaded files and pending work without bound — this is what
# actually keeps a many-users-at-once scenario within a fixed RAM budget.
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "6"))

# Finished jobs (done/error) are dropped from memory after this many seconds
# so long-running multi-user deployments don't slowly leak RAM by keeping
# every past transcription result around forever.
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(60 * 60)))

# Size of each chunk read while streaming an upload to disk. Uploads are
# never buffered fully in memory, so there is no hard limit on file size.
UPLOAD_CHUNK_SIZE = int(os.environ.get("UPLOAD_CHUNK_SIZE", str(8 * 1024 * 1024)))


# --- Automatic ingestion pipeline (materials from a teacher) ---
#
# Where fully organized, renamed lesson folders end up.
MATERIALS_DIR = Path(os.environ.get("MATERIALS_DIR", BASE_DIR / "materials"))

# Where already-processed source files are tracked, so re-syncing doesn't
# reprocess the same file twice.
DB_PATH = Path(os.environ.get("DB_PATH", STORAGE_DIR / "materials.db"))

# How often the background sync loop polls every configured source.
SYNC_INTERVAL_SECONDS = int(os.environ.get("SYNC_INTERVAL_SECONDS", "300"))

# Simplest possible source: a local folder (e.g. a Google Drive / Yandex.Disk
# desktop sync client already mirrors the shared folder here on disk).
LOCAL_WATCH_DIR = os.environ.get("LOCAL_WATCH_DIR")  # e.g. "./watch"

# Yandex.Disk source: a static OAuth token with disk:read scope, see
# https://yandex.ru/dev/disk/poligon/ ("Отладка запросов" issues one).
YANDEX_DISK_TOKEN = os.environ.get("YANDEX_DISK_TOKEN")
YANDEX_DISK_REMOTE_PATH = os.environ.get("YANDEX_DISK_REMOTE_PATH", "/")

# Google Drive source (fixed, admin-configured folder): a service-account
# JSON key file; share the teacher's folder with that service account's
# email so it can see the files.
GOOGLE_DRIVE_CREDENTIALS_FILE = os.environ.get("GOOGLE_DRIVE_CREDENTIALS_FILE")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")

# Google Drive source (student-pasted links): a plain API key is enough to
# read any folder shared "Anyone with the link can view" — set up once by
# whoever runs the service (Cloud Console -> enable Drive API -> API key),
# then students paste their own teacher's folder link through the web UI.
GOOGLE_DRIVE_API_KEY = os.environ.get("GOOGLE_DRIVE_API_KEY")


# --- Local LLM (writes реферат text) ---
#
# A small instruct model in GGUF format, run via llama-cpp-python on CPU —
# same "bake it into the Docker image at build time" approach as Whisper,
# so no network is needed at runtime either. Qwen2.5-3B is a reasonable
# default for an 8GB-RAM box (~2GB at Q4_K_M); bump to a 7B model via env
# vars if you have more RAM and want better writing quality.
LLM_MODEL_REPO = os.environ.get("LLM_MODEL_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF")
LLM_MODEL_FILE = os.environ.get("LLM_MODEL_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf")
# Where the Dockerfile copies the downloaded .gguf weights at build time —
# loaded directly by app.reference.llm at runtime (not via
# Llama.from_pretrained(repo_id=..., filename=...), which always calls the
# HF Hub API to resolve the filename pattern and hard-fails under
# HF_HUB_OFFLINE=1 even when the file is already cached locally).
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", str(BASE_DIR / "models" / "llm.gguf"))
LLM_CONTEXT_SIZE = int(os.environ.get("LLM_CONTEXT_SIZE", "4096"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1400"))
LLM_THREADS = int(os.environ.get("LLM_THREADS", str(os.cpu_count() or 4)))

# --- Реферат title page. Set these in .env (see .env.example) — the
# defaults are placeholders, not any real university or teacher. ---
def _env(key: str, default: str) -> str:
    return os.environ.get(key) or default


REFERENCE_UNIVERSITY_HEADER = _env(
    "REFERENCE_UNIVERSITY_HEADER",
    "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ<br/>"
    "федеральное государственное образовательное учреждение высшего образования<br/>"
    "«НАЗВАНИЕ УНИВЕРСИТЕТА»",
)
REFERENCE_DEPARTMENT = _env("REFERENCE_DEPARTMENT", "КАФЕДРА № __")
REFERENCE_TEACHER_POSITION = _env("REFERENCE_TEACHER_POSITION", "должность, уч. степень")
REFERENCE_TEACHER_NAME = _env("REFERENCE_TEACHER_NAME", "Фамилия И.О.")
REFERENCE_DISCIPLINE = _env("REFERENCE_DISCIPLINE", "Введение в информационные технологии")
REFERENCE_CITY = _env("REFERENCE_CITY", "Город")
REFERENCE_CITY_YEAR = _env("REFERENCE_CITY_YEAR", f"{REFERENCE_CITY}, {date.today().year}")
STUDENT_NAME = _env("STUDENT_NAME", "")  # usually set in the web UI («Мои данные»)
STUDENT_GROUP = _env("STUDENT_GROUP", "")

# Where generated рефераты are written, so they can be listed/downloaded.
REFERENCES_DIR = Path(os.environ.get("REFERENCES_DIR", STORAGE_DIR / "references"))

# Finished lectures ("Следующая лекция"): videos, transcripts and рефераты
# per lecture. docker-compose mounts it from ./archive on the host, so the
# files are visible outside the container too.
ARCHIVE_DIR = Path(os.environ.get("ARCHIVE_DIR", STORAGE_DIR / "archive"))

for directory in (STORAGE_DIR, UPLOADS_DIR, AUDIO_DIR, MATERIALS_DIR, REFERENCES_DIR, ARCHIVE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
