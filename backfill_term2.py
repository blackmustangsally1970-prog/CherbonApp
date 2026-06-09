    # backfill_term2.py
    from app import app, db
    from models import CourseFormSubmission, Lesson, CourseReference
    from sqlalchemy import extract

    def run_backfill():
        selected_year = 2026
        selected_term = 2

        term_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12]
        }

        courses = CourseReference.query.filter_by(active=True).all()

        weekday_to_name = {
            6: "Sunday",
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday"
        }

        term2_lessons = Lesson.query.filter(
            extract('year', Lesson.lesson_date) == selected_year,
            extract('month', Lesson.lesson_date).in_(term_months[selected_term])
        ).all()

        created = 0

        for l in term2_lessons:
            rider = l.client
            dow = weekday_to_name[l.lesson_date.weekday()]

            matched = next(
                (c for c in courses
                 if c.day_of_week == dow
                 and c.timerange == l.time_frame
                 and c.group_priv == l.group_priv),
                None
            )

            if not matched:
                continue

            exists = CourseFormSubmission.query.filter_by(
                rider_name=rider,
                term_year=selected_year,
                term_number=selected_term
            ).first()

            if exists:
                continue

            new_sub = CourseFormSubmission(
                rider_name=rider,
                current_course=matched.course_code,
                term_year=selected_year,
                term_number=selected_term,
                status="backfill",
                ftor="W",
                frequency="W"
            )

            db.session.add(new_sub)
            created += 1

        db.session.commit()
        print(f"Backfill complete. Created {created} Term 2 entries.")


    if __name__ == "__main__":
        with app.app_context():
            run_backfill()
