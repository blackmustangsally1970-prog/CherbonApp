import requests
from flask import current_app


def send_sms(phone, message):

    username = current_app.config["CLICKSEND_USERNAME"]
    api_key = current_app.config["CLICKSEND_API_KEY"]

    sender = current_app.config["WEDDING_SENDER"]

    payload = {
        "messages": [
            {
                "source": "python",
                "from": sender,
                "body": message,
                "to": phone
            }
        ]
    }

    r = requests.post(
        "https://rest.clicksend.com/v3/sms/send",
        json=payload,
        auth=(username, api_key)
    )

    print("CLICKSEND STATUS:", r.status_code)
    print(r.text)

    return r