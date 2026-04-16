"""
STEP 1 — Data Extraction
=========================
Pull raw student engagement data from the TaalMaster Neon PostgreSQL database.

WHY THIS MATTERS:
  ML models don't read your database directly. We need to pull the right tables
  and hand them to the feature engineering step. Think of this as "gathering
  ingredients before cooking."

WHAT WE PULL:
  - users + subscriptions  → who is each student, are they paying?
  - study_streaks          → which days did they log in?
  - quiz_attempts          → how many quizzes, what scores?
  - word_progress          → how many words learned/mastered?
  - writing_submissions    → how many writings, what scores?
  - audio_progress         → how many listening exercises done?
  - content totals         → how much content exists (for coverage %)
"""
import pandas as pd
from sqlalchemy import create_engine, text
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)


def get_students() -> pd.DataFrame:
    """All student accounts with subscription info."""
    query = text("""
        SELECT
            u.id AS user_id,
            u.name,
            u.email,
            u.created_at AS account_created,
            u.trial_ends_at,
            s.status AS subscription_status,
            s.stripe_price_id,
            s.current_period_end AS subscription_expires,
            s.cancel_at_period_end
        FROM users u
        LEFT JOIN subscriptions s ON s.user_id = u.id
        WHERE u.role = 'student'
        ORDER BY u.id
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_study_streaks() -> pd.DataFrame:
    query = text("""
        SELECT user_id, activity_date
        FROM study_streaks
        ORDER BY user_id, activity_date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_quiz_attempts() -> pd.DataFrame:
    query = text("""
        SELECT user_id, section_id, mode, score, total_questions, completed_at
        FROM quiz_attempts
        ORDER BY user_id, completed_at
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_word_progress() -> pd.DataFrame:
    query = text("""
        SELECT user_id, vocabulary_id, status, attempts, correct_attempts, last_seen
        FROM word_progress
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_writing_submissions() -> pd.DataFrame:
    query = text("""
        SELECT user_id, prompt_id, score, submitted_at
        FROM writing_submissions
        ORDER BY user_id, submitted_at
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_audio_progress() -> pd.DataFrame:
    query = text("""
        SELECT user_id, audio_id, listened, listened_at
        FROM audio_progress
        WHERE listened = true
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_total_content_counts() -> dict:
    """How much content exists on the platform (for calculating coverage %)."""
    queries = {
        "total_vocabulary": "SELECT COUNT(*) AS cnt FROM vocabulary",
        "total_audio": "SELECT COUNT(*) AS cnt FROM audio_files",
        "total_prompts": "SELECT COUNT(*) AS cnt FROM writing_prompts",
        "total_sections": "SELECT COUNT(*) AS cnt FROM sections",
    }
    counts = {}
    with engine.connect() as conn:
        for key, q in queries.items():
            result = conn.execute(text(q)).fetchone()
            counts[key] = result[0] if result else 0
    return counts


def extract_all() -> dict:
    """Pull everything needed for feature engineering. One call, all data."""
    return {
        "students": get_students(),
        "streaks": get_study_streaks(),
        "quizzes": get_quiz_attempts(),
        "words": get_word_progress(),
        "writing": get_writing_submissions(),
        "audio": get_audio_progress(),
        "content_counts": get_total_content_counts(),
    }
