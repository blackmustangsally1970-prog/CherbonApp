def smart_proper_name(name):
    if not name:
        return ""

    name = name.strip()
    parts = name.split()

    # Dutch particles that stay lowercase
    lowercase_particles = {
        "van", "der", "den", "de", "het", "ten", "ter", "op", "aan", "bij", "uit", "te"
    }

    fixed_parts = []
    for part in parts:
        p = part.lower()

        # Handle O' prefix
        if p.startswith("o'") and len(p) > 2:
            fixed_parts.append("O'" + p[2:].capitalize())
            continue

        # Handle D' prefix
        if p.startswith("d'") and len(p) > 2:
            fixed_parts.append("D'" + p[2:].capitalize())
            continue

        # Handle Mc prefix
        if p.startswith("mc") and len(p) > 2:
            fixed_parts.append("Mc" + p[2:].capitalize())
            continue

        # Handle Mac prefix
        if p.startswith("mac") and len(p) > 3:
            fixed_parts.append("Mac" + p[3:].capitalize())
            continue

        # Handle hyphens (Mary-Anne)
        if "-" in p:
            fixed_parts.append("-".join(s.capitalize() for s in p.split("-")))
            continue

        # Handle Dutch particles (always lowercase)
        if p in lowercase_particles:
            fixed_parts.append(p)
            continue

        # Default
        fixed_parts.append(p.capitalize())

    return " ".join(fixed_parts)


def generate_unique_client_name(base_name):
    cleaned = base_name.strip()

    # Query existing clients with same base prefix
    existing = Client.query.filter(
        Client.full_name.ilike(f"{cleaned}%")
    ).all()

    if not existing:
        return cleaned

    suffix = 2
    while True:
        candidate = f"{cleaned} ({suffix})"
        if not any(c.full_name == candidate for c in existing):
            return candidate
        suffix += 1

def log_disclaimer_processed(names):
    """
    Logs processed disclaimers to the main CherbonApp log folder
    using Brisbane time and a clean, consistent format.
    """

    try:
        # Brisbane timezone
        brisbane_tz = pytz.timezone("Australia/Brisbane")
        now_brisbane = datetime.now(brisbane_tz)
        timestamp = now_brisbane.strftime("%Y-%m-%d %H:%M:%S")

        # Correct log folder (the one your viewer actually reads)
        log_dir = "/var/log/cherbonapp"
        os.makedirs(log_dir, exist_ok=True)

        # Daily rotating file
        log_file = os.path.join(log_dir, f"disclaimers_{now_brisbane.strftime('%Y-%m-%d')}.log")

        # Write entries
        with open(log_file, "a", encoding="utf-8") as f:
            for name in names:
                f.write(f"{timestamp} — Processed disclaimer for: {name}\n")

        print(f"[DisclaimerLog] Logged {len(names)} disclaimers to {log_file}")

    except Exception as e:
        print(f"[DisclaimerLog][ERROR] {e}")
