import sys
import os
from datetime import timedelta

# Ensure this directory is on the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, get_active_term
from models import db, CourseReference, CourseFormSubmission, Term
from playwright.sync_api import sync_playwright


def render_rider_html(app, submission, course_ref, first_date, last_date):
    with app.app_context():
        from flask import render_template

        # Rider name
        rider_name = submission.rider_name

        # Course details
        course_display_label = course_ref.display_label
        course_day = course_ref.day_of_week
        course_time_range = course_ref.timerange

        # Base price
        base_price = submission.price or 0

        # Payment calculation
        # ftor = W (weekly payment) or FT (full term payment)
        # frequency = W (weekly lessons) or F (fortnightly lessons)
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
    if len(sys.argv) < 2:
        print("ERROR: course_code required", file=sys.stderr)
        sys.exit(1)

    course_code = sys.argv[1]

    # Load your REAL Flask app
    app = create_app()

    with app.app_context():
        course_ref = CourseReference.query.filter_by(course_code=course_code).first()
        if not course_ref:
            print(f"ERROR: No course found for {course_code}", file=sys.stderr)
            sys.exit(1)

        # Use the app's real term logic
        term = get_active_term()

        # Only approved riders
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

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for submission in riders:

            # Compute correct first lesson date
            first_date = term.start_date

            # Fortnightly riders starting in Week 2
            if submission.frequency == "F" and submission.start_week == "W2":
                first_date = term.start_date + timedelta(days=7)

            # Compute correct last lesson date
            if submission.frequency == "W":
                # Weekly = 10 lessons → +9 weeks
                last_date = first_date + timedelta(weeks=9)
            elif submission.frequency == "F":
                # Fortnightly = 5 lessons → +8 weeks
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

    print("OK")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
