"""
Seed realistic students into the TaalMaster Neon DB for model training.
Creates a mix of active, at-risk, and dropout student profiles.

Usage:
    python seed_students.py              # seed 2500 students (default)
    python seed_students.py --count 500  # seed fewer for testing
"""
import argparse
import random
from datetime import datetime, timedelta
from config import DATABASE_URL
from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL)


def get_content_ids():
    with engine.connect() as conn:
        sections = [r[0] for r in conn.execute(text("SELECT id FROM sections")).fetchall()]
        vocab = [r[0] for r in conn.execute(text("SELECT id FROM vocabulary")).fetchall()]
        audio = [r[0] for r in conn.execute(text("SELECT id FROM audio_files")).fetchall()]
        prompts = [r[0] for r in conn.execute(text("SELECT id FROM writing_prompts")).fetchall()]
    return sections, vocab, audio, prompts


def seed(n_students: int = 2500):
    rng = random.Random(42)
    now = datetime.utcnow()
    sections, vocab_ids, audio_ids, prompt_ids = get_content_ids()

    if not sections:
        print("ERROR: No sections found. Run db:seed in the TaalMaster repo first.")
        return

    print(f"Content: {len(sections)} sections, {len(vocab_ids)} words, {len(audio_ids)} audio, {len(prompt_ids)} prompts")

    # 35% active, 30% at-risk, 35% dropout
    profiles = (
        ["active"] * int(n_students * 0.35) +
        ["at_risk"] * int(n_students * 0.30) +
        ["dropout"] * (n_students - int(n_students * 0.35) - int(n_students * 0.30))
    )
    rng.shuffle(profiles)

    first_names = [
        "Emma", "Liam", "Sofia", "Noah", "Mila", "Daan", "Julia", "Sem", "Sara", "Luuk",
        "Anna", "Finn", "Eva", "Jesse", "Sanne", "Bram", "Lisa", "Tim", "Fleur", "Tom",
        "Femke", "Max", "Iris", "Lars", "Nina", "Thijs", "Lotte", "Stijn", "Anouk", "Ruben",
        "Isa", "Milan", "Roos", "Jasper", "Noa", "Bas", "Sophie", "Lucas", "Tessa", "Rick",
        "Ahmed", "Fatima", "Omar", "Yara", "Ali", "Layla", "Hassan", "Nour", "Karim", "Amira",
        "Viktor", "Olga", "Dmitri", "Natasha", "Andrei", "Elena", "Stefan", "Maria", "Jan", "Petra",
        "Wei", "Li", "Chen", "Mei", "Zhang", "Yuki", "Kenji", "Sakura", "Ravi", "Priya",
        "Marco", "Giulia", "Pablo", "Carmen", "Luis", "Ana", "Pierre", "Claire", "Hans", "Ingrid",
    ]
    last_names = [
        "de Vries", "Jansen", "de Boer", "van den Berg", "Bakker", "Visser", "Smit", "Meijer",
        "Mulder", "de Groot", "Bos", "Vos", "Peters", "Hendriks", "van Dijk", "Dekker",
        "Brouwer", "de Wit", "Dijkstra", "Postma", "Schmidt", "Mueller", "Fischer", "Weber",
        "Martinez", "Garcia", "Lopez", "Rodriguez", "Petrov", "Ivanov", "Kim", "Singh",
        "Tanaka", "Kumar", "Rossi", "Dubois", "Johansson", "Andersen", "Okafor", "Mensah",
    ]

    with engine.connect() as conn:
        max_id = conn.execute(text("SELECT COALESCE(MAX(id), 0) FROM users")).fetchone()[0]
        print(f"Starting from user ID {max_id + 1}")

        batch_size = 100
        totals = {"streaks": 0, "quizzes": 0, "words": 0, "writings": 0, "audio": 0}

        for batch_start in range(0, n_students, batch_size):
            batch_end = min(batch_start + batch_size, n_students)
            user_ids = []

            for i in range(batch_start, batch_end):
                profile = profiles[i]
                name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
                email = f"student_{batch_start + (i - batch_start) + max_id + 1}@taalmaster-demo.nl"

                age = {"active": rng.randint(10, 120), "at_risk": rng.randint(5, 60), "dropout": rng.randint(14, 90)}[profile]
                created = now - timedelta(days=age)

                result = conn.execute(text("""
                    INSERT INTO users (name, email, password_hash, role, created_at, trial_ends_at)
                    VALUES (:name, :email, :pw, 'student', :created, :trial) RETURNING id
                """), {"name": name, "email": email, "pw": "$2a$10$dummy_hash_for_seeded_students", "created": created, "trial": created + timedelta(days=3)})
                uid = result.fetchone()[0]
                user_ids.append((uid, profile, age, created))

            # Subscriptions
            for uid, profile, age, created in user_ids:
                if profile == "active" and rng.random() < 0.75:
                    sub_id = f"sub_seed_{uid}_{rng.randint(1000, 9999)}"
                    conn.execute(text("""
                        INSERT INTO subscriptions (user_id, stripe_subscription_id, stripe_price_id, status, current_period_start, current_period_end, cancel_at_period_end)
                        VALUES (:uid, :sub_id, :price, :status, :start, :end, false)
                    """), {"uid": uid, "sub_id": sub_id, "price": rng.choice(["price_monthly", "price_6month", "price_yearly"]), "status": rng.choice(["active", "active", "trialing"]), "start": created, "end": now + timedelta(days=rng.randint(5, 60))})
                elif profile == "at_risk" and rng.random() < 0.35:
                    sub_id = f"sub_seed_{uid}_{rng.randint(1000, 9999)}"
                    conn.execute(text("""
                        INSERT INTO subscriptions (user_id, stripe_subscription_id, stripe_price_id, status, current_period_start, current_period_end, cancel_at_period_end)
                        VALUES (:uid, :sub_id, :price, :status, :start, :end, :cancel)
                    """), {"uid": uid, "sub_id": sub_id, "price": rng.choice(["price_monthly", "price_6month"]), "status": rng.choice(["active", "past_due"]), "start": created, "end": now + timedelta(days=rng.randint(-5, 20)), "cancel": rng.random() < 0.5})
                elif profile == "dropout" and rng.random() < 0.15:
                    sub_id = f"sub_seed_{uid}_{rng.randint(1000, 9999)}"
                    conn.execute(text("""
                        INSERT INTO subscriptions (user_id, stripe_subscription_id, stripe_price_id, status, current_period_start, current_period_end, cancel_at_period_end)
                        VALUES (:uid, :sub_id, 'price_monthly', 'canceled', :start, :end, true)
                    """), {"uid": uid, "sub_id": sub_id, "start": created, "end": now - timedelta(days=rng.randint(1, 30))})

            # Study streaks
            for uid, profile, age, created in user_ids:
                n_days = {"active": rng.randint(max(1, int(age * 0.4)), max(2, int(age * 0.85))), "at_risk": rng.randint(max(1, int(age * 0.1)), max(2, int(age * 0.35))), "dropout": rng.randint(1, max(2, int(age * 0.15)))}[profile]
                possible = list(range(age))
                if profile == "active":
                    weights = [0.3 + 0.7 * (d / max(age, 1)) for d in possible]
                elif profile == "at_risk":
                    weights = [0.5 + 0.5 * (d / max(age, 1)) for d in possible]
                else:
                    cutoff = max(1, age - rng.randint(7, min(age, 60)))
                    weights = [1.0 if d < cutoff else 0.05 for d in possible]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                n_days = min(n_days, len(possible))
                chosen_days = set()
                attempts = 0
                while len(chosen_days) < n_days and attempts < n_days * 3:
                    d = rng.choices(possible, weights=weights, k=1)[0]
                    chosen_days.add(d)
                    attempts += 1
                for d in chosen_days:
                    conn.execute(text("INSERT INTO study_streaks (user_id, activity_date) VALUES (:uid, :d) ON CONFLICT DO NOTHING"), {"uid": uid, "d": (created + timedelta(days=d)).date()})
                    totals["streaks"] += 1

            # Quizzes
            for uid, profile, age, created in user_ids:
                n_q, base = {"active": (rng.randint(8, 50), 0.65 + rng.random() * 0.2), "at_risk": (rng.randint(2, 15), 0.4 + rng.random() * 0.2), "dropout": (rng.randint(0, 8), 0.25 + rng.random() * 0.2)}[profile]
                for _ in range(n_q):
                    total_q = rng.randint(5, 20)
                    score = max(0, min(total_q, round(total_q * (base + rng.gauss(0, 0.12)))))
                    conn.execute(text("INSERT INTO quiz_attempts (user_id, section_id, mode, score, total_questions, completed_at) VALUES (:uid, :sec, :mode, :score, :total, :completed)"),
                        {"uid": uid, "sec": rng.choice(sections), "mode": rng.choice(["flashcard", "multiple_choice", "fill_blank", "translation"]), "score": score, "total": total_q, "completed": created + timedelta(days=rng.randint(0, age - 1), hours=rng.randint(8, 22))})
                    totals["quizzes"] += 1

            # Word progress
            for uid, profile, age, created in user_ids:
                n_w, mastery_p = {"active": (rng.randint(20, min(len(vocab_ids), 120)), 0.45), "at_risk": (rng.randint(5, min(len(vocab_ids), 40)), 0.15), "dropout": (rng.randint(0, min(len(vocab_ids), 20)), 0.08)}[profile]
                for vid in rng.sample(vocab_ids, min(n_w, len(vocab_ids))):
                    att = rng.randint(1, 15)
                    corr = rng.randint(0, att)
                    status = "mastered" if (corr / max(att, 1) >= 0.8 and att >= 3 and rng.random() < mastery_p) else "learning"
                    conn.execute(text("INSERT INTO word_progress (user_id, vocabulary_id, status, attempts, correct_attempts, last_seen) VALUES (:uid, :vid, :status, :att, :corr, :seen) ON CONFLICT DO NOTHING"),
                        {"uid": uid, "vid": vid, "status": status, "att": att, "corr": corr, "seen": created + timedelta(days=rng.randint(0, age - 1))})
                    totals["words"] += 1

            # Writing
            if prompt_ids:
                for uid, profile, age, created in user_ids:
                    n_wr, base_ws = {"active": (rng.randint(2, 12), 6.5), "at_risk": (rng.randint(0, 4), 5.0), "dropout": (rng.randint(0, 2), 3.5)}[profile]
                    for _ in range(n_wr):
                        conn.execute(text("INSERT INTO writing_submissions (user_id, prompt_id, original_text, score, submitted_at) VALUES (:uid, :pid, :txt, :score, :sub)"),
                            {"uid": uid, "pid": rng.choice(prompt_ids), "txt": "Seeded submission.", "score": max(1, min(10, round(base_ws + rng.gauss(0, 1.2)))), "sub": created + timedelta(days=rng.randint(0, age - 1))})
                        totals["writings"] += 1

            # Audio
            if audio_ids:
                for uid, profile, age, created in user_ids:
                    n_a = {"active": rng.randint(3, min(len(audio_ids), 20)), "at_risk": rng.randint(0, min(len(audio_ids), 8)), "dropout": rng.randint(0, min(len(audio_ids), 3))}[profile]
                    for aid in rng.sample(audio_ids, min(n_a, len(audio_ids))):
                        conn.execute(text("INSERT INTO audio_progress (user_id, audio_id, listened, listened_at) VALUES (:uid, :aid, true, :at) ON CONFLICT DO NOTHING"),
                            {"uid": uid, "aid": aid, "at": created + timedelta(days=rng.randint(0, age - 1))})
                        totals["audio"] += 1

            conn.commit()
            print(f"  Batch {batch_start+1}-{batch_end}: {batch_end - batch_start} students")

        count = conn.execute(text("SELECT COUNT(*) FROM users WHERE role = 'student'")).fetchone()[0]
        print(f"\nDone! {n_students} students seeded. Total students in DB: {count}")
        for k, v in totals.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2500)
    args = parser.parse_args()
    seed(args.count)
