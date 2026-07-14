from sqlalchemy import func
from extensions import db
from models import Client



def extract_number(value):
    """
    Extracts digits from a string like '63kg' or '178 cm' and returns an int.
    Returns None if no digits found.
    """
    if not value:
        return None
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def normalize_name(s: str) -> str:
    """
    Normalize a name for matching:
    - strip accents
    - lowercase
    - collapse spaces
    - remove weird unicode spaces
    """
    if not s:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("\xa0", " ")
    return " ".join(s.strip().lower().split())


def parse_jotform_payload(payload, forced_submission_id=None, mode="full"):
    import json
    import re
    from sqlalchemy.util._collections import immutabledict

    if isinstance(payload, immutabledict):
        payload = dict(payload)

    if not isinstance(payload, dict):
        try:
            payload = json.loads(payload)
        except Exception:
            print("ERROR: Could not decode payload:", type(payload))
            return {"riders": [], "disclaimer": None}

    answers = payload.get("answers", {}) or {}

    submission_id = (
        str(forced_submission_id)
        if forced_submission_id
        else str(payload.get("id") or payload.get("submission_id") or "")
    )

    disclaimer_id = str(answers.get("63", {}).get("answer") or "")

    email = ""
    for key, item in answers.items():
        if item.get("type") == "control_email":
            ans = item.get("answer")
            if isinstance(ans, dict):
                email = ans.get("value") or ans.get("text") or ans.get("full") or ""
            else:
                email = ans or ""
            break
    if not email:
        fallback = answers.get("47", {}).get("answer")
        if fallback:
            email = fallback
    email = email or ""

    mobile = ""
    mob_field = answers.get("87", {}).get("answer")
    if isinstance(mob_field, dict):
        mobile = mob_field.get("full") or mob_field.get("text") or ""
    else:
        mobile = mob_field or ""
    mobile = mobile or ""

    guardian = answers.get("86", {}).get("answer") or ""

    invite_token = ""
    for key, item in answers.items():
        if item.get("name") == "i_t":
            ans = item.get("answer")
            if isinstance(ans, dict):
                invite_token = ans.get("text") or ans.get("value") or ans.get("full")
            else:
                invite_token = ans
            break
    invite_token = invite_token or ""

    age_fields = [
        key for key, item in answers.items()
        if item.get("type") in ("control_number", "control_dropdown")
        and "age" in (item.get("text") or "").lower()
    ]
    age_fields = sorted(age_fields, key=lambda x: int(x))

    fullname_fields = []
    for key, item in answers.items():
        if item.get("type") != "control_fullname":
            continue
        label = (item.get("text") or "").lower()
        if "guardian" in label:
            continue
        if "rider" in label:
            fullname_fields.append(key)
    fullname_fields = sorted(fullname_fields, key=lambda x: int(x))

    height_fields = sorted(
        [k for k, v in answers.items() if "height" in (v.get("text") or "").lower()],
        key=lambda x: int(x)
    )
    weight_fields = sorted(
        [k for k, v in answers.items() if "weight" in (v.get("text") or "").lower()],
        key=lambda x: int(x)
    )
    notes_fields = sorted(
        [k for k, v in answers.items() if "notes" in (v.get("text") or "").lower()],
        key=lambda x: int(x)
    )

    riders = []

    for idx, fullname_key in enumerate(fullname_fields):
        item = answers.get(fullname_key)
        if not item:
            continue

        pretty = item.get("prettyFormat")
        if pretty:
            raw_name = pretty
        else:
            ans = item.get("answer") or {}
            raw_name = f"{ans.get('first','')} {ans.get('last','')}".strip()

        name = normalize_name(raw_name)
        if not name:
            continue

        age_key = age_fields[idx] if idx < len(age_fields) else None
        age = answers.get(age_key, {}).get("answer") if age_key else None

        height_key = height_fields[idx] if idx < len(height_fields) else None
        weight_key = weight_fields[idx] if idx < len(weight_fields) else None
        notes_key = notes_fields[idx] if idx < len(notes_fields) else None

        height_val = answers.get(height_key, {}).get("answer") if height_key else None
        weight_val = answers.get(weight_key, {}).get("answer") if weight_key else None
        notes_val = answers.get(notes_key, {}).get("answer") if notes_key else ""

        rider = {
            "name": name,
            "age": age,
            "guardian": guardian,
            "mobile": mobile,
            "email": email,
            "height_cm": extract_number(height_val),
            "weight_kg": extract_number(weight_val),
            "notes": notes_val or "",
            "matches": [],
            "jotform_submission_id": submission_id,
            "invite_token": invite_token,
        }

        rider["rider_index"] = idx + 1

        if mode == "full":
            compact = name.replace(" ", "").replace("-", "")
            like_pattern = f"%{compact}%"

            candidates = db.session.query(
                Client.client_id,
                Client.full_name,
                Client.mobile,
                Client.email_primary,
                Client.jotform_submission_id
            ).filter(
                Client.full_name.isnot(None),
                func.replace(
                    func.replace(func.lower(Client.full_name), " ", ""),
                    "-", ""
                ).like(like_pattern)
            ).all()

            lookup = []
            for c in candidates:
                full = getattr(c, "full_name", "")
                norm = normalize_name(full)
                lookup.append((c, norm))

            matched = None
            for c, norm in lookup:
                if norm == name:
                    matched = c
                    break

            if not matched and email:
                e = email.strip().lower()
                for c, norm in lookup:
                    if (c.email_primary or "").strip().lower() == e:
                        matched = c
                        break

            if not matched and mobile:
                m = re.sub(r"\D", "", mobile)
                for c, norm in lookup:
                    cm = re.sub(r"\D", "", (c.mobile or ""))
                    if cm == m and cm:
                        matched = c
                        break

            if matched:
                rider["matches"].append((
                    matched.client_id,
                    matched.full_name,
                    matched.mobile,
                    matched.email_primary,
                    matched.jotform_submission_id
                ))

        riders.append(rider)

    return {
        "riders": riders,
        "disclaimer": disclaimer_id,
        "submission_id": submission_id,
        "guardian": guardian,
        "email": email,
        "mobile": mobile,
    }

