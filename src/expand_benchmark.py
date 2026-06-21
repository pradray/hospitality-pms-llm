"""Expand benchmark with config advisory and troubleshooting tasks mined from User Guide chunks.

Reads the doc chunks, finds prerequisite/OPERA Control rich sections,
and generates targeted tasks. Appends to existing benchmark.jsonl.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CHUNKS_PATH = PROJECT_ROOT / "output" / "doc_chunks" / "all_doc_chunks.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark.jsonl"


# --- Config advisory tasks grounded in User Guide content ---

CONFIG_TASKS = [
    # Reservations
    {
        "id": "ug-cfg-001", "module": "reservations", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control must be active to allow accompanying guests on a reservation?",
        "expected_answer": "The 'Accompanying Guest' OPERA Control must be active. This allows adding additional guest profiles to a reservation beyond the primary guest.",
    },
    {
        "id": "ug-cfg-002", "module": "reservations", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Controls and prerequisites are needed to enable scheduled room moves for reservations?",
        "expected_answer": "The 'Reservation Sales Screen' OPERA Control must be configured. Room move scheduling requires the reservation to be in InHouse status, and appropriate room availability for the target dates. The Front Desk module must have room assignment capabilities enabled.",
    },
    {
        "id": "ug-cfg-003", "module": "reservations", "category": "config_advisory", "difficulty": "intermediate",
        "question": "How do you configure membership rate rules so that loyalty members automatically get discounted rates?",
        "expected_answer": "The 'Membership Rate Rules' OPERA Control must be active. Rate plans must be associated with membership types and levels. When a reservation is created with a member profile, the system automatically applies the membership rate if the rate plan has membership rules defined.",
    },
    {
        "id": "ug-cfg-004", "module": "reservations", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables sell messages that appear when booking specific dates?",
        "expected_answer": "The 'Activate Sell Message Functionality' OPERA Control must be active. When enabled, date-specific sell messages can be configured to display during the reservation creation process, typically used for promotions or alerts about special events.",
    },
    {
        "id": "ug-cfg-005", "module": "reservations", "category": "config_advisory", "difficulty": "advanced",
        "question": "What configuration controls how award points are redeemed for free night reservations in OPERA Cloud?",
        "expected_answer": "The 'Award Points Redemption' OPERA Control must be active, along with membership type/level configurations that define redemption rules. Rate plans must have guarantee codes linked to award redemption. The membership must have sufficient points balance, and the rate plan must support award stays.",
    },
    {
        "id": "ug-cfg-006", "module": "reservations", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Controls govern how multiple yield market types affect rate availability?",
        "expected_answer": "The 'Multiple Yield Market Type' OPERA Control must be active. This allows different market segments (corporate, leisure, group) to have separate yield thresholds and restrictions. Rate plans can be assigned to specific market codes, and availability is managed per market type.",
    },
    {
        "id": "ug-cfg-007", "module": "reservations", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables promotion coupon codes for reservations?",
        "expected_answer": "The 'Promotion Coupon Codes' OPERA Control must be active, along with the Promotions Module. This allows guests to enter promotional codes during booking to receive discounted rates or added value.",
    },
    {
        "id": "ug-cfg-008", "module": "reservations", "category": "config_advisory", "difficulty": "advanced",
        "question": "What is the complete configuration needed to enable the advance check-in feature for reservations?",
        "expected_answer": "1. The 'Advance Check In' (or 'Advanced Check In') OPERA Control must be active. 2. Pre-registration rules must be configured at the property level. 3. The reservation must be in Due-In status. 4. Room inspection workflows must be set up so clean/inspected rooms are available. 5. Credit card authorization rules should be configured for advance payment capture.",
    },
    {
        "id": "ug-cfg-009", "module": "reservations", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Controls are needed to enable advanced deposit handling for reservations?",
        "expected_answer": "The 'Advanced Deposit Handling' OPERA Control must be active. This enables more sophisticated deposit rules including automatic deposit scheduling, partial deposits, deposit refund rules, and deposit policy enforcement at booking time.",
    },
    {
        "id": "ug-cfg-010", "module": "reservations", "category": "config_advisory", "difficulty": "basic",
        "question": "What configuration enables the reservation trace functionality in OPERA Cloud?",
        "expected_answer": "The 'Reservation Traces' OPERA Control must be active. Traces are automated task reminders linked to reservations that trigger on specific dates (e.g., arrival day, departure day). Trace departments and trace text must be configured.",
    },
    # CRM
    {
        "id": "ug-cfg-011", "module": "crm", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control must be active to manage membership claims in OPERA Cloud?",
        "expected_answer": "The 'Membership Claims' OPERA Control must be active. This enables tracking inquiries about a member's account such as missing points or stays, tier inquiries, information requests, and new card requests.",
    },
    {
        "id": "ug-cfg-012", "module": "crm", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What prerequisites must be met for creating guest or contact profiles in OPERA Cloud?",
        "expected_answer": "1. The 'Contacts' OPERA Control must be active for contact profiles. 2. The 'Multi Language' OPERA Control enables multi-language profile support. 3. Mandatory communication details settings control which fields (email, phone) are required. 4. Address validation settings may require valid address formats.",
    },
    {
        "id": "ug-cfg-013", "module": "crm", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Control enables the incognito/privacy feature for guest profiles?",
        "expected_answer": "The 'Incognito' OPERA Control must be active. When enabled, guests can be marked as incognito, which hides their real name from displays and reports, showing a pseudonym instead. This is used for VIP privacy protection.",
    },
    {
        "id": "ug-cfg-014", "module": "crm", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables automatic issuance of loyalty awards?",
        "expected_answer": "The 'Auto Issue Awards' OPERA Control must be active. When enabled, the Issue Awards process can be run automatically to grant points or awards to members based on their qualifying activity (stays, revenue, etc.).",
    },
    {
        "id": "ug-cfg-015", "module": "crm", "category": "config_advisory", "difficulty": "advanced",
        "question": "What configuration is needed to enable IATA validation for travel agent profiles?",
        "expected_answer": "The 'IATA Validation' OPERA Control must be active. This validates travel agent IATA numbers against an external database during profile creation. The validation service endpoint must be configured, and commission handling should be set up for valid IATA agents.",
    },
    {
        "id": "ug-cfg-016", "module": "crm", "category": "config_advisory", "difficulty": "intermediate",
        "question": "How is the batch profile update feature configured in OPERA Cloud?",
        "expected_answer": "The 'Batch Profile Update' OPERA Control must be active. This allows mass updates to guest profiles such as updating addresses, preferences, or membership details in bulk. Appropriate user permissions must be granted, and update templates should be configured.",
    },
    # Front Office
    {
        "id": "ug-cfg-017", "module": "front_office", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Controls and prerequisites are needed for the pre-registration feature?",
        "expected_answer": "1. The 'Pre-registration External System Trace' OPERA Control must be configured for external system notifications. 2. The 'Pre-registration External System Alert' OPERA Control determines alert behavior. 3. The 'Alerts' OPERA Control must be active. 4. Reservations must be in Due-In status. 5. Pre-check-in rules must be configured at the property level.",
    },
    {
        "id": "ug-cfg-018", "module": "front_office", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables automatic room assignment during check-in?",
        "expected_answer": "The 'Auto Assign Room at Check-In' OPERA Control must be active. When enabled, the system automatically assigns an available inspected room of the correct room type to the reservation during the check-in process.",
    },
    {
        "id": "ug-cfg-019", "module": "front_office", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What configuration is needed to handle back-to-back linked reservations at the front desk?",
        "expected_answer": "The 'Back-to-Back Handling for Linked Reservations' OPERA Control must be active. This enables special handling when a guest has consecutive reservations — the system can automatically manage room retention, folio transfers, and continuous stay processing without requiring a physical checkout and re-checkin.",
    },
    {
        "id": "ug-cfg-020", "module": "front_office", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Control enables text message/SMS handling for guest communications?",
        "expected_answer": "The 'Text Message Handling' OPERA Control must be active. This enables sending text messages to guests for notifications such as room ready alerts, queue updates, and pre-arrival messages. An SMS gateway/provider must be configured for message delivery.",
    },
    {
        "id": "ug-cfg-021", "module": "front_office", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables the discrepant room feature at the front desk?",
        "expected_answer": "The 'Discrepant Room' OPERA Control must be active. This feature flags rooms where the front office status (occupied/vacant) doesn't match the housekeeping status (clean/dirty/occupied), helping identify discrepancies for investigation.",
    },
    {
        "id": "ug-cfg-022", "module": "front_office", "category": "config_advisory", "difficulty": "advanced",
        "question": "What configuration controls the check-in prepayment rules in OPERA Cloud?",
        "expected_answer": "The 'Check In Prepay Rules' OPERA Control must be active. Prepay rules define whether payment must be collected at check-in based on the reservation's rate plan, guarantee type, or market code. Rules can require full prepayment, deposit collection, or credit card authorization before check-in is allowed.",
    },
    {
        "id": "ug-cfg-023", "module": "front_office", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Control enables the connecting rooms feature?",
        "expected_answer": "The 'Connecting Rooms' OPERA Control must be active. This enables room configuration to define which rooms are physically connected (adjacent with a connecting door). During room assignment, the system can prioritize or enforce connecting room assignments for multi-room reservations or family bookings.",
    },
    {
        "id": "ug-cfg-024", "module": "front_office", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables the confidential billing window feature?",
        "expected_answer": "The 'Confidential Billing Window' OPERA Control must be active. This creates a separate billing folio window that is hidden from the guest's view, typically used for comp charges, VIP amenities, or internal charges that the hotel absorbs.",
    },
    # Housekeeping
    {
        "id": "ug-cfg-025", "module": "housekeeping", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control must be active to enable the guest service status feature for housekeeping?",
        "expected_answer": "The 'Guest Service Status' OPERA Control must be active. This enables tracking of guest service preferences and room servicing status, allowing guests to indicate whether they want their room serviced or prefer privacy (Do Not Disturb / Make Up Room).",
    },
    {
        "id": "ug-cfg-026", "module": "housekeeping", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Controls are needed to enable housekeeping credit calculations?",
        "expected_answer": "The 'Housekeeping Credits' OPERA Control must be active. Credits define the cleaning workload for each room type (e.g., a suite gets more credits than a standard room). The 'Guest Age Categories' and 'Turndown' controls may also affect credit calculations for evening service.",
    },
    {
        "id": "ug-cfg-027", "module": "housekeeping", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What OPERA Control enables advanced task sheet generation for housekeeping?",
        "expected_answer": "The 'Advanced Task Sheet' OPERA Control must be active, along with 'Housekeeping Task Scheduling'. Advanced task sheets enable automatic work distribution based on room credits, attendant capacity, floor assignments, and room priority. Task sheets can be auto-generated considering departure/arrival patterns.",
    },
    {
        "id": "ug-cfg-028", "module": "housekeeping", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables the inspected room status in housekeeping?",
        "expected_answer": "The 'Inspected Rooms' (or 'Inspected Status') OPERA Control must be active. This adds an 'Inspected' status to the room condition workflow (Dirty → Clean → Inspected), requiring a supervisor inspection before a room is marked ready for guest assignment.",
    },
    {
        "id": "ug-cfg-029", "module": "housekeeping", "category": "config_advisory", "difficulty": "basic",
        "question": "What OPERA Control enables the pickup room status for housekeeping?",
        "expected_answer": "The 'Pickup Rooms' (or 'Pickup Status') OPERA Control must be active. Pickup is an intermediate cleaning status between Dirty and Clean, indicating that the room needs light tidying (e.g., stay-over rooms) rather than a full cleaning.",
    },
    {
        "id": "ug-cfg-030", "module": "housekeeping", "category": "config_advisory", "difficulty": "intermediate",
        "question": "What configuration is needed for the room rotation feature in housekeeping?",
        "expected_answer": "The 'Room Rotation' OPERA Control must be active. Room rotation ensures even wear across rooms by tracking how recently each room was used and prioritizing assignment of less-recently-used rooms. This is configured at the room type level and affects room assignment order.",
    },
]

# --- Troubleshooting tasks grounded in User Guide content ---

TROUBLESHOOTING_TASKS = [
    {
        "id": "ug-tbl-001", "module": "reservations", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "A hotel has the Promotions Module active but promotion coupon codes are not working during booking. What should be checked?",
        "expected_answer": "1. The 'Promotion Coupon Codes' OPERA Control must be active in addition to the Promotions Module. 2. The promotion must be configured with valid date ranges that include the booking dates. 3. The coupon code must be associated with an active promotion. 4. Rate plans linked to the promotion must be available for the requested dates and room types. 5. The promotion may have usage limits that have been exhausted.",
    },
    {
        "id": "ug-tbl-002", "module": "reservations", "category": "troubleshooting", "difficulty": "basic",
        "question": "Sell messages are not appearing during reservation creation even though they have been configured. What could be wrong?",
        "expected_answer": "1. The 'Activate Sell Message Functionality' OPERA Control may not be active. 2. The sell messages may not be configured for the specific dates being booked. 3. The messages may be configured for a different property than the one being booked. 4. The user's role may not have permission to view sell messages.",
    },
    {
        "id": "ug-tbl-003", "module": "crm", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "A hotel is trying to mark a VIP guest as incognito but the option is not available on the profile. What should be checked?",
        "expected_answer": "1. The 'Incognito' OPERA Control must be active at the property level. 2. The user's role must have permission to set incognito status. 3. The profile must be a guest profile (not a company or agent profile). 4. If using multi-property, the control must be active at the specific property.",
    },
    {
        "id": "ug-tbl-004", "module": "crm", "category": "troubleshooting", "difficulty": "basic",
        "question": "Loyalty points are not being automatically issued to members after checkout. What should be checked?",
        "expected_answer": "1. The 'Auto Issue Awards' OPERA Control must be active. 2. The membership type and level must have award rules configured. 3. The Issue Awards process may need to be run (manually or scheduled). 4. The reservation must have a valid membership profile attached. 5. The stay may not meet minimum qualification criteria (e.g., minimum nights, eligible rate codes).",
    },
    {
        "id": "ug-tbl-005", "module": "front_office", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "The auto room assignment feature is not assigning rooms during check-in. Rooms show as available and inspected. What could be wrong?",
        "expected_answer": "1. The 'Auto Assign Room at Check-In' OPERA Control may not be active. 2. Available rooms may not match the reserved room type. 3. Room preferences on the reservation (floor, features, connecting) may be restricting assignment. 4. The 'Inspected Rooms' control may require rooms to be in Inspected (not just Clean) status. 5. Room rotation settings may affect which rooms are eligible.",
    },
    {
        "id": "ug-tbl-006", "module": "front_office", "category": "troubleshooting", "difficulty": "advanced",
        "question": "A guest has back-to-back reservations but the system is requiring a full checkout and re-checkin between stays, causing billing issues. What configuration is missing?",
        "expected_answer": "1. The 'Back-to-Back Handling for Linked Reservations' OPERA Control must be active. 2. The two reservations must be linked in the system. 3. The reservations must have consecutive dates with no gap. 4. Room retention settings should be configured to keep the same room across both stays. 5. Folio transfer rules should be set up to handle billing continuity.",
    },
    {
        "id": "ug-tbl-007", "module": "front_office", "category": "troubleshooting", "difficulty": "basic",
        "question": "Discrepant rooms are not being flagged even though front office and housekeeping statuses clearly differ. What should be checked?",
        "expected_answer": "1. The 'Discrepant Room' OPERA Control must be active. 2. The room statuses must be updated in real-time — stale data from either system will prevent discrepancy detection. 3. The discrepancy report may need to be manually run or the check may only run during specific processes (e.g., end of day).",
    },
    {
        "id": "ug-tbl-008", "module": "front_office", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "Text message notifications to guests are configured but not being delivered. What should be investigated?",
        "expected_answer": "1. The 'Text Message Handling' OPERA Control must be active. 2. The SMS gateway/provider integration must be configured and operational. 3. The guest profile must have a valid mobile phone number in the correct format. 4. The notification triggers (room ready, queue update) must be configured. 5. The hotel's SMS service quota may be exhausted.",
    },
    {
        "id": "ug-tbl-009", "module": "housekeeping", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "Housekeeping task sheets are being generated but the credit allocations seem wrong — suites are getting the same cleaning time as standard rooms. What configuration should be checked?",
        "expected_answer": "1. The 'Housekeeping Credits' OPERA Control must be active. 2. Room type credit values must be configured — each room type should have appropriate credit weights (e.g., suite = 2 credits, standard = 1 credit). 3. The 'Advanced Task Sheet' control should be active for credit-based distribution. 4. Attendant capacity (max credits per shift) must be configured.",
    },
    {
        "id": "ug-tbl-010", "module": "housekeeping", "category": "troubleshooting", "difficulty": "basic",
        "question": "Rooms cleaned by attendants are showing as 'Clean' but not progressing to 'Inspected' status. What is missing?",
        "expected_answer": "1. The 'Inspected Rooms' or 'Inspected Status' OPERA Control must be active. 2. If the control is active, a supervisor must explicitly inspect and mark rooms as Inspected — the system does not auto-promote from Clean to Inspected. 3. If the control is not active, rooms go directly from Dirty to Clean without requiring inspection.",
    },
    {
        "id": "ug-tbl-011", "module": "housekeeping", "category": "troubleshooting", "difficulty": "basic",
        "question": "The guest service status (Do Not Disturb / Make Up Room) indicators are not available on room status screens. What is missing?",
        "expected_answer": "The 'Guest Service Status' OPERA Control must be active at the property level. This control enables the Do Not Disturb and Make Up Room indicators that guests can set via in-room controls or mobile app, and that housekeeping staff can see on their task sheets and room status screens.",
    },
    {
        "id": "ug-tbl-012", "module": "reservations", "category": "troubleshooting", "difficulty": "advanced",
        "question": "A revenue manager reports that yield controls are not working correctly — restricted rate plans are still available for booking on dates that should be closed. What should be investigated?",
        "expected_answer": "1. The 'Multiple Yield Market Type' OPERA Control settings should be verified. 2. Rate plan restrictions (close-to-arrival, minimum stay, maximum stay) must be correctly configured for the specific dates. 3. Rate category restrictions may not be set up at the room class level (requires 'Rate Category Restrictions' OPERA Control). 4. Yield levels and thresholds must be properly defined. 5. Override permissions — certain user roles may have the ability to override restrictions.",
    },
    {
        "id": "ug-tbl-013", "module": "reservations", "category": "troubleshooting", "difficulty": "intermediate",
        "question": "Reservation traces are configured but not triggering on the expected dates. What could be the issue?",
        "expected_answer": "1. The 'Reservation Traces' OPERA Control must be active. 2. Trace departments must be configured and active. 3. The trace date type (arrival, departure, creation, custom) must be correctly set. 4. The trace may have already been resolved or deleted. 5. The End of Day process may need to run to generate traces for the next day.",
    },
]


def main():
    existing = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            existing.append(json.loads(line))

    seen_ids = {t["id"] for t in existing}

    new_tasks = CONFIG_TASKS + TROUBLESHOOTING_TASKS
    added = 0
    for task in new_tasks:
        if "required_operations" not in task:
            task["required_operations"] = []
        if "evaluation_criteria" not in task:
            task["evaluation_criteria"] = "Must identify the relevant OPERA Control(s) and key configuration steps."
        if "source" not in task:
            task["source"] = "user_guide_derived"

        if task["id"] not in seen_ids:
            existing.append(task)
            seen_ids.add(task["id"])
            added += 1

    with open(BENCHMARK_PATH, "w") as f:
        for t in existing:
            f.write(json.dumps(t) + "\n")

    print(f"Added {added} new tasks (total: {len(existing)})")

    by_cat = {}
    by_mod = {}
    by_diff = {}
    for t in existing:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        by_mod[t["module"]] = by_mod.get(t["module"], 0) + 1
        by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1

    print("\nBy category:")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:25s} {v}")
    print("\nBy module:")
    for k, v in sorted(by_mod.items()):
        print(f"  {k:25s} {v}")
    print("\nBy difficulty:")
    for k, v in sorted(by_diff.items()):
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
