import sys
import os
from datetime import timedelta

# Ensure this directory is on the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, get_active_term
from models import db, CourseReference, CourseFormSubmission
from playwright.sync_api import sync_playwright


def render_rider_html(app, submission, course_ref, first_date, last_date):
    with app.app_context():
        from flask import render_template

        rider_name = submission.rider_name
        course_display_label = course_ref.display_label
        course_day = course_ref.day_of_week
        course_time_range = course_ref.timerange

        base_price = submission.price or 0

        # Payment calculation
        if submission.ftor == "FT":
            if submission.frequency == "W":
                calculated_price = base_price * 10
            elif submission.frequency == "F":
                calculated_price = base_price * 5
            else:
                calculated_price = base_price
        else:
            calculated_price = base_price

        return render_template(
            "rider_pdf_template.html",
            rider_name=rider_name,
            course_display_label=course_display_label,
            course_day=course_day,
            course_time_range=course_time_range,
            first_lesson_date=first_date.strftime("%d/%m/%Y"),
            last_lesson_date=last_date.strftime("%d/%m/%Y"),
            submission=submission,
            calculated_price=calculated_price
        )


def main():
    # --- Validate args ---
    if len(sys.argv) < 2:
        # IMPORTANT: print NOTHING except the error
        print("ERROR: course_code required")
        sys.exit(1)

    course_code = sys.argv[1]

    # --- Load Flask app ---
    app = create_app()

    with app.app_context():
        course_ref = CourseReference.query.filter_by(course_code=course_code).first()
        if not course_ref:
            print(f"ERROR: No course found for {course_code}")
            sys.exit(1)

        term = get_active_term()

        riders = CourseFormSubmission.query.filter(
            CourseFormSubmission.current_course == course_code,
            CourseFormSubmission.term_year == term.year,
            CourseFormSubmission.term_number == term.term_number,
            CourseFormSubmission.status == "approved",
            CourseFormSubmission.cancelled == False
        ).all()

    output_dir = "/home/ec2-user/CherbonApp/static/pdfs"
    os.makedirs(output_dir, exist_ok=True)

    generated_paths = []

    # Weekday mapping
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    # --- Playwright (silent mode) ---
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for submission in riders:

            # REATTACH submission to active DB session
            with app.app_context():
                submission = CourseFormSubmission.query.get(submission.id)

            # Determine weekday
            course_day_name = course_ref.day_of_week
            course_weekday = weekday_map.get(course_day_name, 0)

            term_weekday = term.start_date.weekday()
            offset_days = (course_weekday - term_weekday) % 7

            first_date = term.start_date + timedelta(days=offset_days)

            # Fortnightly W2 riders start one week later
            if submission.frequency == "F" and submission.start_week == "W2":
                first_date = first_date + timedelta(weeks=1)

            # Last lesson date
            if submission.frequency == "W":
                last_date = first_date + timedelta(weeks=9)
            elif submission.frequency == "F":
                last_date = first_date + timedelta(weeks=8)
            else:
                last_date = first_date

            html = render_rider_html(app, submission, course_ref, first_date, last_date)
            page.set_content(html)

            filename = f"{course_code}_{submission.id}.pdf"
            full_path = os.path.join(output_dir, filename)

            page.pdf(path=full_path, format="A4")
            generated_paths.append(full_path)

        browser.close()

    # --- STRICT OUTPUT FORMAT ---
    print("OK")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
