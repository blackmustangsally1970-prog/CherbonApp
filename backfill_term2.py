from app import app, db
from models import CourseFormSubmission, Lesson, CourseReference

def insert_june7_cc():
    selected_year = 2026
    selected_term = 2

    # The corrected course reference
    target_course = CourseReference.query.filter_by(
        active=True,
        day_of_week="Sunday",
        group_priv="CC",
        timerange="14:30 - 15:50"
    ).first()

    if not target_course:
        print("No matching CourseReference for Sunday CC 14:30 - 15:50")
        return

    # Pull riders ONLY for 7 June 2026
    lessons = Lesson.query.filter(
        Lesson.lesson_date == '2026-06-07',
        Lesson.group_priv == "CC",
        Lesson.time_frame == "14:30 - 15:50"
    ).all()

    created = 0
    skipped = 0

    for l in lessons:
        rider = l.client

        # Check if rider already has a Term 2 entry
        exists = CourseFormSubmission.query.filter_by(
            rider_name=rider,
            term_year=selected_year,
            term_number=selected_term
        ).first()

        if exists:
            skipped += 1
            continue

        # Create new entry
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
    print(f"Done. Created {created}, Skipped {skipped} (already existed).")


if __name__ == "__main__":
    with app.app_context():
        insert_june7_cc()
