import sys
import os

# Ensure this directory is on the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, get_active_term
from models import db, CourseReference, CourseFormSubmission, Term
from playwright.sync_api import sync_playwright


def compute_first_last_lesson(term: Term):
    first = term.start_date.strftime("%d/%m/%Y")
    last = term.end_date.strftime("%d/%m/%Y")
    return first, last


def render_rider_html(app, rider, course, first_date, last_date):
    with app.app_context():
        from flask import render_template
        return render_template(
            "rider_pdf_template.html",
            rider=rider,
            course=course,
            first_date=first_date,
            last_date=last_date,
        )


def main():
    if len(sys.argv) < 2:
        print("ERROR: course_code required", file=sys.stderr)
        sys.exit(1)

    course_code = sys.argv[1]

    # Load your REAL Flask app
    app = create_app()

    with app.app_context():
        course = CourseReference.query.filter_by(course_code=course_code).first()
        if not course:
            print(f"ERROR: No course found for {course_code}", file=sys.stderr)
            sys.exit(1)

        # Use the app's real term logic
        term = get_active_term()
        first_date, last_date = compute_first_last_lesson(term)

        riders = CourseFormSubmission.query.filter_by(
            current_course=course_code,
            term_year=term.year,
            term_number=term.number,
            cancelled=False
        ).all()

    output_dir = "/home/ec2-user/CherbonApp/static/pdfs"
    os.makedirs(output_dir, exist_ok=True)

    generated_paths = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for rider in riders:
            html = render_rider_html(app, rider, course, first_date, last_date)
            page.set_content(html)

            filename = f"{course_code}_{rider.id}.pdf"
            full_path = os.path.join(output_dir, filename)

            page.pdf(path=full_path, format="A4")
            generated_paths.append(full_path)

        browser.close()

    print("OK")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
