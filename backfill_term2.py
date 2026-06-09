from app import app, db
from models import CourseFormSubmission, Lesson, CourseReference
from sqlalchemy import extract

def run_target_backfill():
    selected_year = 2026
    selected_term = 2

    term_months = {
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, 8, 9],
        4: [10, 11, 12]
    }

    # ONLY the corrected Sunday CC 14:30 - 15:50 course
    target_course = CourseReference.query.filter_by(
        active=True,
        day_of_week="Sunday",
        group_priv="CC",
        timerange="14:30 - 15:50"
    ).first()

    if not target_course:
        print("No matching CourseReference for Sunday CC 14:30 - 15:50")
        return

    lessons = Lesson.query.filter(
        extract('year', Lesson.lesson_date) == selected_year,
        extract('month', Lesson.lesson_date).in_(term_months[selected_term]),
        Lesson.group_priv == "CC",
        Lesson.time_frame == "14:30 - 15:50"
    ).all()

    created = 0

    for l in lessons:
        rider = l.client

        exists = CourseFormSubmission.query.filter_by(
            rider_name=rider,
            term_year=selected_year,
            term_number=selected_term
        ).first()

        if exists:
            continue

        new_sub = CourseFormSubmission(
            rider_name=rider,
            current_course=target_course.course_code,
            term_year=selected_year,
            term_number=selected_term,
            status="backfill",
            ftor="W",
            frequency="W"
        )

        db.session.add(new_sub)
        created += 1

    db.session.commit()
    print(f"Target backfill complete. Created {created} Sunday CC 14:30 - 15:50 T2 entries.")


if __name__ == "__main__":
    with app.app_context():
        run_target_backfill()
