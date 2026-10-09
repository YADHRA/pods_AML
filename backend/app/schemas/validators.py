from datetime import date


def validate_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "Date must use YYYY-MM-DD format."
        ) from exc

    return parsed.isoformat()