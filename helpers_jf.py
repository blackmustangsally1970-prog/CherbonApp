import json

# JotForm Rider field mapping (from your form definition)
RIDER_QIDS = {
    1: {"name": "49", "age": "95", "height": "48", "weight": "50"},
    2: {"name": "77", "age": "96", "height": "64", "weight": "51"},
    3: {"name": "78", "age": "97", "height": "65", "weight": "52"},
    4: {"name": "79", "age": "98", "height": "66", "weight": "53"},
    5: {"name": "80", "age": "99", "height": "67", "weight": "54"},
    6: {"name": "81", "age": "100", "height": "68", "weight": "55"},
    7: {"name": "82", "age": "101", "height": "69", "weight": "56"},
    8: {"name": "83", "age": "102", "height": "70", "weight": "57"},
    9: {"name": "84", "age": "103", "height": "71", "weight": "58"},
    10: {"name": "85", "age": "104", "height": "72", "weight": "59"},
}

TRAIL_FORM_ID = "211290218538049"  # Your JotForm ID


def get_answer(submission, qid):
    answers = submission.get('answers', {})
    field = answers.get(str(qid)) or answers.get(qid)
    if not field:
        return None
    return field.get('answer')


def extract_riders_from_submission(raw_payload):
    submission = raw_payload if isinstance(raw_payload, dict) else json.loads(raw_payload)
    riders = []

    for idx, qids in RIDER_QIDS.items():
        name_answer = get_answer(submission, qids["name"])
        if not name_answer:
            continue

        # Fullname control
        if isinstance(name_answer, dict):
            first = (name_answer.get('first') or '').strip()
            last = (name_answer.get('last') or '').strip()
            full_name = (first + ' ' + last).strip()
        else:
            full_name = str(name_answer).strip()

        if not full_name:
            continue

        age = get_answer(submission, qids["age"])
        height = get_answer(submission, qids["height"])
        weight = get_answer(submission, qids["weight"])

        riders.append({
            "index": idx,
            "name": full_name,
            "age": int(age) if str(age).isdigit() else None,
            "height_cm": int(height) if str(height).isdigit() else None,
            "weight_kg": int(weight) if str(weight).isdigit() else None,
        })

    return riders


def get_main_contact_fields(raw_payload):
    submission = raw_payload if isinstance(raw_payload, dict) else json.loads(raw_payload)
    phone = get_answer(submission, "4")   # Phone Number
    email = get_answer(submission, "5")   # Email
    return {
        "phone": phone,
        "email": email,
    }
