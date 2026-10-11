from flask import current_app
import smtplib
from email.mime.text import MIMEText

def send_email(to_address, subject, body):
    smtp_host = current_app.config.get("SMTP_HOST")
    smtp_user = current_app.config.get("SMTP_USER")
    smtp_pass = current_app.config.get("SMTP_PASS")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_address

    with smtplib.SMTP_SSL(smtp_host, 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, [to_address], msg.as_string())
