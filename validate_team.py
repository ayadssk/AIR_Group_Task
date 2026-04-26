"""
Run this script to validate your team.toml before submitting:
    python validate_team.py
"""

import sys
import tomllib


def validate(path: str = "team.toml") -> None:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    errors = []

    if not data.get("team_name", "").strip():
        errors.append("team_name is empty.")

    track = data.get("track")
    if track not in range(1, 5):
        errors.append(f"track must be an integer between 1 and 4, got: {track!r}.")

    members = data.get("members", [])
    if len(members) != 5:
        errors.append(f"Expected exactly 5 members, got {len(members)}.")

    for i, m in enumerate(members):
        p = f"members[{i}]"
        if not m.get("name", "").strip():
            errors.append(f"{p}.name is empty.")
        nr = m.get("immatriculation_number", "").strip()
        if not nr:
            errors.append(f"{p}.immatriculation_number is empty.")
        elif not nr.isdigit():
            errors.append(f"{p}.immatriculation_number must contain only digits, got: {nr!r}.")
        if not m.get("github_username", "").strip():
            errors.append(f"{p}.github_username is empty.")

    if errors:
        print("Validation failed — please fix the following errors:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("team.toml is valid.")
    print(f"  Team  : {data['team_name']}")
    print(f"  Track : {data['track']}")
    print(f"  Members:")
    for m in members:
        print(f"    - {m['name']}  [{m['immatriculation_number']}]  @{m['github_username']}")


if __name__ == "__main__":
    validate()
