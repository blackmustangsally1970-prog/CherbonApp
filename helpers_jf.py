import json

# JotForm Rider field mapping (from your form definition)
RIDER_QIDS = {
    1:  {"name": "49", "age": "95",  "height": "48", "weight": "50"},
    2:  {"name": "77", "age": "96",  "height": "64", "weight": "51"},
    3:  {"name": "78", "age": "97",  "height": "65", "weight": "52"},
    4:  {"name": "79", "age": "98",  "height": "66", "weight": "53"},
    5:  {"name": "80", "age": "99",  "height": "67", "weight": "54"},
    6:  {"name": "81", "age": "100", "height": "68", "weight": "55"},
    7:  {"name": "82", "age": "101", "height": "69", "weight": "56"},
    8:  {"name": "83", "age": "102", "height": "70", "weight": "57"},
    9:  {"name": "84", "age": "103", "height": "71", "weight": "58"},
    10: {"name": "85", "age": "104", "height": "72", "weight": "59"},
    11: {"name": "86", "age": "105", "height": "73", "weight": "60"},
    12: {"name": "87", "age": "106", "height": "74", "weight": "61"},
    13: {"name": "88", "age": "107", "height": "75", "weight": "62"},
    14: {"name": "89", "age": "108", "height": "76", "weight": "63"},
}


TRAIL_FORM_ID = "211290218538049"  # Your JotForm ID


# -----------------------------------------
# General Enquiry JotForm (single rider)
# -----------------------------------------

GENERAL_ENQUIRY_FORM_ID = "211288469503056"

GENERAL_QIDS = {
    "rider_name_first": "3",      # answers["3"]["answer"]["first"]
    "rider_name_last": "3",       # answers["3"]["answer"]["last"]
    "email_address": "4",         # answers["4"]["answer"]
    "mobile_phone": "5",          # answers["5"]["answer"]["full"]
    "rider_age": "9",             # answers["9"]["answer"]
    "rider_height_cm": "10",      # answers["10"]["answer"]
    "rider_weight_kg": "11",      # answers["11"]["answer"]
    "comments": "15"              # answers["15"]["answer"]
}


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

def parse_general_enquiry_payload(raw_payload):
    """
    Parse a single JotForm submission for the General Enquiry form
    into a clean, flat dict.
    """
    submission = raw_payload if isinstance(raw_payload, dict) else json.loads(raw_payload)
    answers = submission.get("answers", {})

    def _get(qid):
        field = answers.get(str(qid)) or answers.get(qid)
        if not field:
            return None
        return field.get("answer")

    # Rider name (fullname control, same QID for first/last)
    name_field = _get(GENERAL_QIDS["rider_name_first"])
    if isinstance(name_field, dict):
        first = (name_field.get("first") or "").strip()
        last = (name_field.get("last") or "").strip()
        rider_name = (first + " " + last).strip()
    else:
        rider_name = str(name_field).strip() if name_field else None

    # Simple fields
    email = _get(GENERAL_QIDS["email_address"])
    mobile_field = _get(GENERAL_QIDS["mobile_phone"])
    if isinstance(mobile_field, dict):
        mobile = (mobile_field.get("full") or "").strip()
    else:
        mobile = str(mobile_field).strip() if mobile_field else None

    age = _get(GENERAL_QIDS["rider_age"])
    height = _get(GENERAL_QIDS["rider_height_cm"])
    weight = _get(GENERAL_QIDS["rider_weight_kg"])
    comments = _get(GENERAL_QIDS["comments"])

    return {
        "rider_name": rider_name or None,
        "rider_age": int(age) if age and str(age).isdigit() else None,
        "rider_height_cm": int(height) if height and str(height).isdigit() else None,
        "rider_weight_kg": int(weight) if weight and str(weight).isdigit() else None,
        "email_address": email or None,
        "mobile_phone": mobile or None,
        "comments": comments or None,
    }


