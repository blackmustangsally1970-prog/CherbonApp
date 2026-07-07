# generate_course_pdfs_cli.py
import sys
from datetime import timedelta
from playwright.sync_api import sync_playwright
from app import db, Course, Client, CourseRider, Term, render_template  # adjust imports

def generate_for_course(course_code):
    course = Course.query.filter_by(course_code=course_code).first_or_404()

    riders = (
        db.session.query(Client)
        .join(CourseRider, CourseRider.client_id == Client.client_id)
        .filter(CourseRider.course_code == course_code)
        .all()
    )

    term = Term.query.filter_by(active=True).first()
    if not term:
        print("ERROR: No active term")
        return 1

    term_start = term.start_date
    weeks = term.weeks

    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    target_weekday = weekday_map[course.day_of_week]
    first_date = term_start + timedelta(days=(target_weekday - term_start.weekday()) % 7)

    if course.frequency == "F" and course.start_week == "W2":
        first_date += timedelta(weeks=1)

    if course.frequency == "W":
        last_date = first_date + timedelta(weeks=weeks - 1)
    else:
        last_date = first_date + timedelta(weeks=weeks - 2)

    generated = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for r in riders:
            html = render_template(
                "rider_pdf_template.html",
                course=course,
                rider=r,
                first_date=first_date.strftime("%d %b %Y"),
                last_date=last_date.strftime("%d %b %Y")
            )

            page.set_content(html)

            rider_pdf_path = f"static/pdfs/{course_code}_{r.client_id}.pdf"
            page.pdf(path=rider_pdf_path)
            generated.append(rider_pdf_path)

        browser.close()

    print("OK")
    for path in generated:
        print(path)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_course_pdfs_cli.py COURSE_CODE")
        sys.exit(1)
    sys.exit(generate_for_course(sys.argv[1]))
