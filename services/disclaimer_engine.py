def build_conflict_context(submission_row, rider_index):
    """
    Build all data needed to render the conflict page.
    Returns a dict with rider, guardian, mobile, email, disclaimer, matches.
    """

    parsed = parse_jotform_payload(
        submission_row.raw_payload,
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
    """
    Handles the POST conflict resolution logic.
    Returns a redirect URL for finalize_notification.
    """

    import json

    parsed = parse_jotform_payload(
        submission_row.raw_payload,
        forced_submission_id=submission_row.id,
        mode="full"
    )

    all_riders = parsed["riders"]

    if rider_index < 1 or rider_index > len(all_riders):
        return None

    rider = all_riders[rider_index - 1]

    guardian   = (parsed.get("guardian") or "").strip()
    mobile     = ''.join(filter(str.isdigit, parsed.get("mobile") or ""))
    email      = (parsed.get("email") or "").strip()
    disclaimer = int(parsed.get("disclaimer") or 0)

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
        pass

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

    # MARK RIDER AS RESOLVED
    parsed["riders"][rider_index - 1]["resolved"] = True
    parsed["riders"][rider_index - 1]["matches"] = []

    submission_row.raw_payload = json.dumps(parsed)
    db.session.commit()

    return url_for('finalize_notification', webhook_id=submission_row.id)


def process_all_fastpath():
    """
    Fast-path processor for submissions with no conflicts.
    Returns a redirect URL.
    """

    from datetime import datetime
    import json

    # Get next unprocessed submission
    next_row = (
        IncomingSubmission.query
        .filter_by(processed=False, ignored=False)
        .first()
    )

    if not next_row:
        return url_for('notifications')

    # Parse payload
    parsed = parse_jotform_payload(
        next_row.raw_payload,
        forced_submission_id=next_row.id,
        mode="full"
    )

    riders = parsed["riders"]
    valid_riders = [r for r in riders if not r.get("incomplete")]

    # Auto-ignore if no valid riders
    if not valid_riders:
        next_row.processed = True
        next_row.ignored = True
        next_row.processed_at = datetime.utcnow()
        db.session.commit()
        return url_for('process_all_pending')

    # Conflict detection
    for idx, rider in enumerate(valid_riders, start=1):
        if rider.get("resolved"):
            continue

        matches = rider.get("matches", [])
        if matches:
            return url_for(
                'resolve_conflict',
                submission_id=next_row.id,
                rider_index=idx
            )

    # No conflicts → fast create
    disclaimer = int(parsed.get("disclaimer") or 0)

    for rider in valid_riders:
        name = smart_proper_name(rider["name"])
        age = int(rider.get("age") or 0)
        guardian = rider.get("guardian")
        mobile = ''.join(filter(str.isdigit, rider.get("mobile") or ""))
        email = rider.get("email")
        jotform_id = rider.get("jotform_submission_id")

        new_client = Client(
            full_name=name,
            age=age,
            guardian_name=guardian,
            mobile=mobile,
            email_primary=email,
            disclaimer=disclaimer,
            height_cm=rider.get("height_cm"),
            weight_kg=rider.get("weight_kg"),
            notes=rider.get("notes"),
            jotform_submission_id=jotform_id
        )
        db.session.add(new_client)

    # Mark processed
    names = [r["name"] for r in valid_riders]
    log_disclaimer_processed(names)

    next_row.processed = True
    next_row.processed_at = datetime.utcnow()
    db.session.commit()

    return url_for('process_all_pending')


def finalize_submission(submission_row):
    """
    Finalizes ALL riders in a submission.
    Returns a redirect URL.
    """

    import json
    from datetime import datetime

    # Parse full payload
    parsed = parse_jotform_payload(
        submission_row.raw_payload,
        forced_submission_id=submission_row.id,
        mode="full"
    )

    all_riders = parsed["riders"]
    valid_riders = [r for r in all_riders if not r.get("incomplete")]

    # If no valid riders → ignore
    if not valid_riders:
        submission_row.processed = True
        submission_row.ignored = True
        submission_row.processed_at = datetime.utcnow()
        db.session.commit()
        return url_for('notifications')

    # Submission-level fields
    guardian   = parsed.get("guardian")
    mobile     = ''.join(filter(str.isdigit, parsed.get("mobile") or ""))
    email      = parsed.get("email")
    disclaimer = int(parsed.get("disclaimer") or 0)

    jotform_id = str(submission_row.form_id)

    # Load clients once
    all_clients = Client.query.all()
    clients_by_id = {c.client_id: c for c in all_clients}

    # ---------------------------------------------------------
    # PROCESS ALL RIDERS
    # ---------------------------------------------------------
    for rider in valid_riders:

        # Skip resolved riders
        if rider.get("resolved"):
            continue

        name = smart_proper_name(rider["name"])
        age = int(rider.get("age") or 0)
        height_cm = int(rider.get("height_cm") or 0)
        weight_kg = int(rider.get("weight_kg") or 0)
        notes = (rider.get("notes") or "").strip()

        matches = rider.get("matches", [])

        # Existing client
        if matches:
            existing_id = matches[0][0]
            client = clients_by_id.get(existing_id)
            if client:
                client.full_name = name
                client.guardian_name = guardian
                client.age = age
                client.mobile = mobile
                client.email_primary = email
                client.height_cm = height_cm
                client.weight_kg = weight_kg
                client.notes = notes
                client.disclaimer = disclaimer
                client.jotform_submission_id = jotform_id
            continue

        # New client
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
            jotform_submission_id=jotform_id
        )
        db.session.add(new_client)

    # ---------------------------------------------------------
    # MARK SUBMISSION AS PROCESSED
    # ---------------------------------------------------------
    submission_row.processed = True
    submission_row.processed_at = datetime.utcnow()
    db.session.commit()

    # ---------------------------------------------------------
    # NEXT SUBMISSION
    # ---------------------------------------------------------
    next_row = IncomingSubmission.query.filter_by(
        processed=False,
        ignored=False
    ).order_by(IncomingSubmission.id.asc()).first()

    if next_row:
        return url_for('finalize_notification', webhook_id=next_row.id)

    return url_for('notifications')


