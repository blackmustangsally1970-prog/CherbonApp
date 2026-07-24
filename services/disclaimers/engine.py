from models import IncomingSubmission, Client
from extensions import db
from flask import url_for

from services.disclaimers.parser import parse_jotform_payload, normalize_name
from services.disclaimers.name_tools import (
    smart_proper_name,
    generate_unique_client_name,
    log_disclaimer_processed
)


import json
import ast
from sqlalchemy.util._collections import immutabledict

def safe_decode_payload(raw):
    if not raw:
        return {}

    # If it's already a dict or immutabledict, just return a plain dict
    if isinstance(raw, (dict, immutabledict)):
        return dict(raw)

    # Try normal JSON
    try:
        return json.loads(raw)
    except:
        pass

    # Try cleaned UTF-8
    try:
        cleaned = raw.encode("utf-8", "ignore").decode("utf-8")
        return json.loads(cleaned)
    except:
        pass

    # Try Python literal
    try:
        return ast.literal_eval(raw)
    except:
        pass

    print("ERROR: Could not decode original payload")
    return {}



def build_conflict_context(submission_row, rider_index):
    """
    Build all data needed to render the conflict page.
    Returns a dict with rider, guardian, mobile, email, disclaimer, matches.
    """

    decoded = safe_decode_payload(submission_row.raw_payload)

    parsed = parse_jotform_payload(
        json.dumps(decoded),
        forced_submission_id=submission_row.id,
        mode="full"
    )

    all_riders = parsed["riders"]

    # Validate index (1-based)
    if rider_index < 1 or rider_index > len(all_riders):
        return None

    rider = all_riders[rider_index - 1]

    guardian   = parsed.get("guardian")
    mobile     = ''.join(filter(str.isdigit, parsed.get("mobile") or ""))
    email      = parsed.get("email")
    disclaimer = parsed.get("disclaimer")

    # Convert matches into Client objects
    match_objects = []
    for m in rider.get("matches", []):
        client_id, full_name, mobile_m, email_m, jot_id = m
        client = Client.query.get(client_id)
        if client:
            match_objects.append(client)

    return {
        "rider": rider,
        "guardian": guardian,
        "mobile": mobile,
        "email": email,
        "disclaimer": disclaimer,
        "matches": match_objects,
        "all_riders": all_riders
    }

def process_conflict_resolution(submission_row, rider_index, choice, client_id):
    import json
    from datetime import datetime
    from sqlalchemy.util._collections import immutabledict

    # 1. Load ORIGINAL payload (full submission)
    original = submission_row.raw_payload

    if isinstance(original, (dict, immutabledict)):
        full_payload = dict(original)
    else:
        full_payload = safe_decode_payload(original)

    # 2. Parse FULL payload for conflict logic
    parsed = parse_jotform_payload(
        json.dumps(full_payload),
        forced_submission_id=submission_row.id,
        mode="full"
    )

    all_riders = parsed["riders"]

    # Safety check
    if rider_index < 1 or rider_index > len(all_riders):
        return None

    rider = all_riders[rider_index - 1]

    # Extract shared fields
    guardian   = (parsed.get("guardian") or "").strip()
    mobile     = ''.join(filter(str.isdigit, parsed.get("mobile") or ""))
    email      = (parsed.get("email") or "").strip()
    disclaimer = int(parsed.get("disclaimer") or 0)

    # Store global disclaimer on submission row
    submission_row.universal_disclaimer = disclaimer

    raw_name = (rider.get("name") or "").strip()
    name = smart_proper_name(raw_name)

    age = int(rider.get("age") or 0)
    height_cm = int(rider.get("height_cm") or 0)
    weight_kg = int(rider.get("weight_kg") or 0)
    notes = (rider.get("notes") or "").strip()

    jotform_id = str(submission_row.form_id)

    existing = Client.query.get(client_id)

    # APPLY USER CHOICE
    if choice == "ignore":
        submission_row.ignored = True

    elif choice in ("use_existing", "overwrite") and existing:
        existing.full_name = name
        existing.guardian_name = guardian
        existing.age = age
        existing.mobile = mobile
        existing.email_primary = email
        existing.height_cm = height_cm
        existing.weight_kg = weight_kg
        existing.notes = notes
        existing.disclaimer = disclaimer
        existing.jotform_submission_id = jotform_id

    elif choice == "new":
        new_client = Client(
            full_name=name,
            guardian_name=guardian,
            age=age,
            mobile=mobile,
            email_primary=email,
            disclaimer=disclaimer,
            height_cm=height_cm,
            weight_kg=weight_kg,
            notes=notes,
            invoice_required=False,
            jotform_submission_id=jotform_id
        )
        db.session.add(new_client)

    elif choice == "new_same_name":
        final_name = generate_unique_client_name(name)
        new_client = Client(
            full_name=final_name,
            guardian_name=guardian,
            age=age,
            mobile=mobile,
            email_primary=email,
            disclaimer=disclaimer,
            height_cm=height_cm,
            weight_kg=weight_kg,
            notes=notes,
            invoice_required=False,
            jotform_submission_id=jotform_id
        )
        db.session.add(new_client)

    # 3. UPDATE CONFLICT STATE IN NEW FIELDS
    submission_row.resolved_riders = submission_row.resolved_riders or {}
    submission_row.cleared_matches = submission_row.cleared_matches or {}

    submission_row.resolved_riders[str(rider_index)] = True
    submission_row.cleared_matches[str(rider_index)] = True

    # 4. OPTIONAL: update payload for debugging/consistency
    full_riders = full_payload.get("riders", [])
    if rider_index - 1 < len(full_riders):
        full_riders[rider_index - 1]["resolved"] = True
        full_riders[rider_index - 1]["matches"] = []

    full_payload["riders"] = full_riders
    submission_row.raw_payload = json.dumps(full_payload)

    # 5. Commit changes
    db.session.commit()

    # 6. Continue pipeline via finalize_notification
    return url_for('finalize_notification', webhook_id=submission_row.id)




def process_all_fastpath():
    """
    Fast-path processor:
    - Finds the next unprocessed, non-ignored submission
    - Hands it to finalize_submission()
    - Respects conflict redirects
    - Returns a URL for the caller to redirect to
    """

    # Find next pending submission
    next_row = (
        IncomingSubmission.query
        .filter_by(processed=False, ignored=False)
        .order_by(IncomingSubmission.received_at.asc())
        .first()
    )

    # Nothing left to process → back to notifications
    if not next_row:
        return url_for('notifications')

    # Let finalize_submission decide:
    # - resolve riders
    # - send to conflict if needed
    # - or return to notifications when done
    redirect_url = finalize_submission(next_row)
    return redirect_url


def finalize_submission(submission_row):
    """
    Main pipeline for processing a single submission.
    Fully patched version:
    - Auto-creates new clients when no matches exist
    - Respects resolved_riders + cleared_matches
    - Prevents parse_jotform_payload from resurrecting matches
    - Kills phantom matches
    - Correct conflict routing
    - Correct processed flag logic
    """

    import json
    from sqlalchemy.util._collections import immutabledict
    from datetime import datetime
    print(">>> USING ENGINE FILE:", __file__)
    print("=== FINALIZE SUBMISSION (AUTO-CREATE VERSION) ===")
    print(f"Submission ID: {submission_row.id}")
    print(f"processed: {submission_row.processed}, ignored: {submission_row.ignored}")
    print(f"resolved_riders (raw): {submission_row.resolved_riders}")
    print(f"cleared_matches (raw): {submission_row.cleared_matches}")

    # Load original payload
    original = submission_row.raw_payload

    if isinstance(original, (dict, immutabledict)):
        full_payload = dict(original)
    else:
        full_payload = safe_decode_payload(original)

    # Parse FULL payload
    parsed = parse_jotform_payload(
        json.dumps(full_payload),
        forced_submission_id=submission_row.id,
        mode="full"
    )

    riders = parsed["riders"]
    total_riders = len(riders)

    resolved_map = submission_row.resolved_riders or {}
    cleared_map = submission_row.cleared_matches or {}

    print(f"Total riders: {total_riders}")

    # ⭐ CRITICAL FIX #1 — Prevent resurrected matches
    for idx, rider in enumerate(riders, start=1):
        if cleared_map.get(str(idx)):
            print(f"Rider {idx}: matches cleared previously → forcing matches=[]")
            rider["matches"] = []

    # ⭐ CRITICAL FIX #2 — Kill phantom matches
    for idx, rider in enumerate(riders, start=1):
        matches = rider.get("matches") or []
        if matches:
            real_matches = []
            for m in matches:
                client_id = m[0]
                if Client.query.get(client_id):
                    real_matches.append(m)

            if not real_matches:
                print(f"Rider {idx}: phantom matches detected → auto-clearing")
                rider["matches"] = []
                cleared_map[str(idx)] = True
                resolved_map[str(idx)] = True

    # ⭐ MAIN RIDER LOOP
    for idx, rider in enumerate(riders, start=1):

        matches = rider.get("matches") or []
        resolved_flag = resolved_map.get(str(idx))
        cleared_flag = cleared_map.get(str(idx))

        print(f"Rider {idx}: name={rider.get('name')}, "
              f"resolved={resolved_flag}, cleared={cleared_flag}, "
              f"matches_count={len(matches)}")

        # Already resolved → skip
        if resolved_flag:
            continue

        # Has matches and not cleared → conflict
        if matches and not cleared_flag:
            print(f"Rider {idx} has unresolved matches → sending to conflict")
            return url_for(
                'resolve_conflict',
                submission_id=submission_row.id,
                rider_index=idx
            )

        # ⭐ AUTO-CREATE NEW CLIENT WHEN NO MATCHES EXIST
        if not matches:
            print(f"Rider {idx}: NO MATCHES → AUTO-CREATING NEW CLIENT")

            new_client = Client(
                full_name=smart_proper_name(rider.get("name") or ""),
                guardian_name=parsed.get("guardian") or "",
                age=int(rider.get("age") or 0),
                mobile=''.join(filter(str.isdigit, parsed.get("mobile") or "")),
                email_primary=parsed.get("email") or "",
                disclaimer=int(parsed.get("disclaimer") or 0),
                height_cm=int(rider.get("height_cm") or 0),
                weight_kg=int(rider.get("weight_kg") or 0),
                notes=rider.get("notes") or "",
                invoice_required=False,
                jotform_submission_id=str(submission_row.form_id)
            )

            db.session.add(new_client)

            resolved_map[str(idx)] = True
            cleared_map[str(idx)] = True
            continue

        # No matches OR matches cleared → mark resolved
        resolved_map[str(idx)] = True

    # Update DB fields
    submission_row.resolved_riders = resolved_map
    submission_row.cleared_matches = cleared_map

    # ⭐ FINAL CHECK — All riders resolved?
    all_resolved = all(
        resolved_map.get(str(i), False)
        for i in range(1, total_riders + 1)
    )

    print(f"All riders resolved? {all_resolved}")

    if all_resolved:
        submission_row.processed = True
        submission_row.processed_at = datetime.utcnow()
        db.session.commit()
        print("Submission marked processed → back to notifications")
        return url_for('notifications')

    db.session.commit()
    print("Not all resolved (unexpected) → back to notifications anyway")
    return url_for('notifications')
