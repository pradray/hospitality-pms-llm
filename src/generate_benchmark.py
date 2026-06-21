"""Generate benchmark tasks from Postman workflows, API specs, and documentation.

Reads the workflow collection for multi-step orchestration tasks,
API specs for single-endpoint tasks, and docs for config/troubleshooting tasks.
Outputs benchmark_seed.jsonl with ~150-200 tasks.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent  # ../Dissertation
POSTMAN_DIR = DATA_ROOT / "data/api-specs/hospitality-api-docs/postman-collections/property"
OUTPUT = PROJECT_ROOT / "data" / "benchmark.jsonl"

# Module mapping for workflow names
WORKFLOW_MODULE = {
    "Check In": "front_office",
    "Check Out": "front_office",
    "Contactless Guest Journey": "multi",
    "Create a Block": "reservations",
    "Create a Standard Rate": "reservations",
    "Create Share Reservations": "reservations",
    "Add an Accompanying Guest to a Reservation": "reservations",
    "Add Tickets to a Reservation": "reservations",
    "Search Availability and Book a Reservation in OPERA Cloud": "reservations",
    "Search Availability by Room Number ": "reservations",
    "Upgrade a Reservation in OPERA Cloud using Property APIs": "reservations",
    "Create a Service Request for a Guest": "front_office",
    "Create a Room Maintenance Request": "housekeeping",
    "Guest Messages for an In House Guest": "reservations",
    "Issue Guest a Digital Key": "front_office",
    "Perform a Room Move": "front_office",
    "Post a charge to a Folio": "front_office",
    "Queue Rooms": "front_office",
    "Schedule a Wake Up Call": "front_office",
    "Property Interfaces - Post Charges Request": "front_office",
    "Property Interfaces - Post Simple Charges": "front_office",
    "Property Interfaces - Guest Inquiry": "reservations",
    "Property Interfaces - Guest Messages": "reservations",
    "Property Interfaces - Locator": "reservations",
    "Property Interfaces - Room Equipment": "housekeeping",
    "Property Interfaces - Wake Up Calls": "front_office",
}


def load_workflows() -> list[dict]:
    path = POSTMAN_DIR / "oracle-hospitality-property-workflows.postman_collection.json"
    with open(path) as f:
        col = json.load(f)
    workflows = col["item"][1]["item"]

    parsed = []
    for wf in workflows:
        steps = []
        for s in wf.get("item", []):
            req = s.get("request", {})
            method = req.get("method", "?")
            url = req.get("url", {})
            if isinstance(url, dict):
                path_parts = url.get("path", [])
                path = "/" + "/".join(path_parts)
            else:
                path = url
            body = req.get("body", {}).get("raw", "")

            steps.append({
                "name": s["name"],
                "method": method,
                "path": path,
                "has_body": bool(body),
            })

        parsed.append({
            "name": wf["name"],
            "module": WORKFLOW_MODULE.get(wf["name"], "multi"),
            "steps": steps,
            "step_count": len(steps),
        })
    return parsed


def generate_workflow_orchestration_tasks(workflows: list[dict]) -> list[dict]:
    """Generate multi-step orchestration tasks from Postman workflows."""
    tasks = []
    task_id = 0

    for wf in workflows:
        if wf["name"].startswith("OSEM") or wf["name"].startswith("Create a Room Class") or wf["name"].startswith("Create a Room Type"):
            continue

        steps_desc = []
        for s in wf["steps"]:
            steps_desc.append(f"{s['method']} {s['path']}")

        unique_paths = []
        seen = set()
        for s in wf["steps"]:
            key = f"{s['method']} {s['path']}"
            if key not in seen:
                unique_paths.append(key)
                seen.add(key)

        difficulty = "basic" if len(unique_paths) <= 3 else "intermediate" if len(unique_paths) <= 6 else "advanced"

        expected = "The required API sequence is:\n" + "\n".join(
            f"{i+1}. {s['name']}: {s['method']} {s['path']}"
            for i, s in enumerate(wf["steps"])
            if f"{s['method']} {s['path']}" in unique_paths
        )

        operations = [s["name"] for s in wf["steps"]]

        task_id += 1
        tasks.append({
            "id": f"wf-orch-{task_id:03d}",
            "module": wf["module"],
            "category": "api_orchestration",
            "difficulty": difficulty,
            "question": f"What is the complete API workflow for: {wf['name']}? List all required API calls in order.",
            "expected_answer": expected,
            "required_operations": operations,
            "evaluation_criteria": f"Must identify the correct sequence of {len(unique_paths)} unique API calls in the right order. Partial credit for getting key steps.",
            "source": "postman_workflow",
        })

    return tasks


def generate_workflow_config_tasks(workflows: list[dict]) -> list[dict]:
    """Generate config advisory tasks based on workflow prerequisites."""
    tasks = []

    config_scenarios = [
        {
            "id": "wf-cfg-001", "module": "reservations",
            "question": "What OPERA Cloud configurations must be in place before a channel manager can search availability and create reservations via API?",
            "expected_answer": "1. Rate plans must be configured with room types and schedules (postRatePlans, postRatePlanSchedules). 2. Room types must be set up and available for sale. 3. Guarantee policies must be configured (getGuaranteePolicies). 4. Payment methods must be enabled (getPaymentMethods via LOV). 5. The external system/channel must be registered with valid OAuth credentials and x-app-key.",
            "difficulty": "advanced",
        },
        {
            "id": "wf-cfg-002", "module": "front_office",
            "question": "What prerequisites must be met in OPERA Cloud before a self-service kiosk can check in guests via API?",
            "expected_answer": "1. Rooms must be in Inspected and Vacant status (hotelRoomStatus=Inspected, hotelRoomFrontOfficeStatus=Vacant). 2. The reservation must be in Due-In status. 3. Pre-registration rules must be configured if using pre-check-in. 4. The kiosk application must be registered with valid OAuth credentials. 5. Room assignment permissions must be granted to the interface.",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-003", "module": "front_office",
            "question": "What is required to enable posting charges to guest folios via a POS interface?",
            "expected_answer": "1. Transaction codes must be configured for each type of charge (getTransactionCodes). 2. The POS system must be registered as an external interface with valid credentials. 3. The reservation must be in InHouse status. 4. Folio windows must be configured if routing charges to specific folios. 5. Posting permissions must be granted to the interface.",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-004", "module": "reservations",
            "question": "What configurations are needed to set up upsell/upgrade rules for reservations?",
            "expected_answer": "1. Room types or room classes must be defined. 2. Upsell rules must be created using postUpsellRule (rsv/config endpoint), specifying from/to room types or classes. 3. Rate codes for upsell pricing must be configured. 4. Membership levels can optionally be linked to upsell eligibility. 5. Rules can be validated using testUpsellRule before activation.",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-005", "module": "reservations",
            "question": "What is required before creating a Block in OPERA Cloud?",
            "expected_answer": "1. Block statuses must be configured (getBlockNewStatuses). 2. Block reservation types must be set up (getBlockReservationTypes). 3. Rate codes for the block must exist (getBlockRateCodes). 4. Block origins and booking types should be configured. 5. A default block code can be generated (getBlockDefaultCode).",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-006", "module": "reservations",
            "question": "What configuration determines which rate plans are available for a specific date range and room type?",
            "expected_answer": "1. Rate plan schedules must be created with date ranges and room type assignments (postRatePlanSchedules). 2. Rate categories must be configured. 3. Room types must be associated with the rate plan. 4. The rate plan must be active and not restricted for the requested dates. 5. Market codes and source codes may further restrict availability.",
            "difficulty": "advanced",
        },
        {
            "id": "wf-cfg-007", "module": "front_office",
            "question": "What configuration is needed to enable digital key issuance for guests?",
            "expected_answer": "1. The digital key vendor system must be registered as an external interface. 2. The guest must have a checked-in reservation with a room assigned. 3. The room key API endpoint (postRoomKeys) must be enabled. 4. The interface must have appropriate permissions for key issuance.",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-008", "module": "front_office",
            "question": "What configuration controls the checkout process and when is a guest allowed to check out?",
            "expected_answer": "1. The reservation must be in InHouse status. 2. The folio balance should typically be $0 (all charges settled with payments). 3. Transaction codes for payments must be configured. 4. Folio generation (postFolios) must be done to create the bill number before checkout. 5. Early checkout or late checkout settings may be configured at the property level.",
            "difficulty": "advanced",
        },
        {
            "id": "wf-cfg-009", "module": "housekeeping",
            "question": "What configuration is needed to manage room maintenance requests via API?",
            "expected_answer": "1. Maintenance codes must be configured for different types of issues (e.g., plumbing, electrical, HVAC). 2. Room IDs must be set up and mapped. 3. The maintenance workflow (create, update, resolve, unresolve, delete) uses the hsk module endpoints. 4. Service request categories should be defined for tracking.",
            "difficulty": "basic",
        },
        {
            "id": "wf-cfg-010", "module": "front_office",
            "question": "What configuration controls wake-up call scheduling in OPERA Cloud?",
            "expected_answer": "1. Wake-up call functionality must be enabled at the property level. 2. The reservation must be in InHouse status. 3. Wake-up calls are managed per reservation using POST/PUT/DELETE on the wakeUpCalls endpoint. 4. The interface system must have permissions to manage wake-up calls.",
            "difficulty": "basic",
        },
        {
            "id": "wf-cfg-011", "module": "front_office",
            "question": "What is the room queue feature and what configuration does it require?",
            "expected_answer": "1. Queue rooms allows placing arriving guests in a queue when their room is not yet ready. 2. Uses PUT /fof/v1/hotels/{hotelId}/queuedReservations/{reservationId} to add to queue. 3. GET queuedReservations to view the queue. 4. DELETE to remove from queue when room becomes available. 5. Queue statistics are available via getQueueReservationsStatistics.",
            "difficulty": "intermediate",
        },
        {
            "id": "wf-cfg-012", "module": "reservations",
            "question": "How are share reservations configured and what are the rules?",
            "expected_answer": "1. Two separate reservations must first be created for the same room and dates. 2. POST /rsv/v1/hotels/{hotelId}/reservations/{reservationId}/shares links them as sharers. 3. Rate splitting can be configured using putShareRateAmount to divide charges. 4. GET shares to view sharing details. 5. DELETE shares to unlink the reservations.",
            "difficulty": "intermediate",
        },
    ]

    for cfg in config_scenarios:
        cfg["category"] = "config_advisory"
        cfg["required_operations"] = []
        cfg["evaluation_criteria"] = "Must identify key configuration prerequisites. Partial credit for covering main points."
        cfg["source"] = "postman_workflow_derived"
        tasks.append(cfg)

    return tasks


def generate_workflow_troubleshooting_tasks() -> list[dict]:
    """Generate troubleshooting tasks based on common workflow failure modes."""
    return [
        {
            "id": "wf-tbl-001", "module": "front_office", "category": "troubleshooting",
            "difficulty": "basic",
            "question": "A kiosk integration calls postCheckIn but receives an error that the reservation cannot be checked in. The reservation status is 'Confirmed'. What is wrong?",
            "expected_answer": "The reservation must be in 'Due-In' status to check in, not 'Confirmed'. A reservation moves to Due-In status on the arrival date. The kiosk should verify the reservation status using getReservation and check that the arrival date matches today's date.",
            "required_operations": [], "evaluation_criteria": "Must identify Due-In status requirement",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-002", "module": "front_office", "category": "troubleshooting",
            "difficulty": "intermediate",
            "question": "A mobile check-in app calls postRoomAssignments but gets an error. The room shows as available in the PMS. What could be wrong?",
            "expected_answer": "1. The room may be available but not Inspected — room assignment typically requires hotelRoomStatus=Inspected. 2. The room may be Occupied (not Vacant) in front office status even if physically empty. 3. The room type may not match the reservation's room type. 4. Another reservation may have been assigned the room between the availability check and the assignment call (race condition). 5. The room may be in Out of Order or Out of Service status.",
            "required_operations": [], "evaluation_criteria": "Must identify Inspected status requirement and at least 2 other causes",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-003", "module": "front_office", "category": "troubleshooting",
            "difficulty": "intermediate",
            "question": "A POS system posts charges successfully using postBillingCharges, but the charges appear on the wrong guest's folio. The room has two guests (share reservation). How should this be handled?",
            "expected_answer": "1. With share reservations, multiple reservations exist for the same room. 2. The POS system must identify the correct reservationId, not just the room number. 3. When looking up by room number, getReservations may return multiple results — the system must let the server/guest select the correct reservation. 4. Routing instructions on each reservation control which folio window receives charges.",
            "required_operations": [], "evaluation_criteria": "Must identify share reservation as the cause and recommend using correct reservationId",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-004", "module": "front_office", "category": "troubleshooting",
            "difficulty": "advanced",
            "question": "A checkout integration calls postCheckOuts but receives an error that the folio has an outstanding balance. The integration has already posted a payment covering all charges. What steps are missing?",
            "expected_answer": "1. After posting payment via postBillingPayments, the integration must generate the folio number using postFolios before checkout. 2. The folio generation step (postFolios) creates the bill number and finalizes the folio. 3. Only after the folio is generated with a $0 balance can postCheckOuts succeed. 4. The integration should verify the folio balance using getFolios before attempting checkout.",
            "required_operations": [], "evaluation_criteria": "Must identify the missing postFolios step between payment and checkout",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-005", "module": "reservations", "category": "troubleshooting",
            "difficulty": "intermediate",
            "question": "A channel manager creates a reservation with postReservation but the reservation does not appear when searching with getReservations. What could be wrong?",
            "expected_answer": "1. The search parameters may not match — check arrival date range, reservation status filter, and hotel ID. 2. The reservation may have been created at a different property than expected. 3. The search may be filtering by a specific status (e.g., Confirmed) while the reservation is in a different status (e.g., Reserved/Tentative). 4. There may be a slight delay in the reservation becoming searchable. 5. The channel manager should capture the reservation ID from the Location header in the postReservation response instead of searching.",
            "required_operations": [], "evaluation_criteria": "Must mention checking search parameters and using the Location header from the POST response",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-006", "module": "reservations", "category": "troubleshooting",
            "difficulty": "basic",
            "question": "An integration tries to split a multi-room reservation using putSplitMultiRoomReservation but gets an error. What should be checked?",
            "expected_answer": "1. The reservation must have 2 or more rooms to be split. 2. The reservation ID must be valid and the reservation must be in an active status (not cancelled or checked out). 3. The request body must include proper payment method copy settings (copyCreditCards, copyOthers). 4. The reservation may already have been split previously.",
            "required_operations": [], "evaluation_criteria": "Must identify that the reservation needs multiple rooms and be in active status",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-007", "module": "reservations", "category": "troubleshooting",
            "difficulty": "advanced",
            "question": "A booking engine calls getHotelAvailability and receives available room types and rates. However, when it tries to create a reservation with postReservation using those exact rate and room type codes, it gets a 'rate not available' error. What could cause this?",
            "expected_answer": "1. Rate availability can change between the search and booking attempt (inventory sold out). 2. The rate plan may have restrictions (minimum stay, closed to arrival, booking window) that weren't checked. 3. The guarantee policy selected may not be valid for the rate plan. 4. Rate plan schedules may not cover the full date range of the reservation. 5. The rate may require specific market/source codes that weren't provided. 6. The rate may have maximum occupancy limits that are exceeded.",
            "required_operations": [], "evaluation_criteria": "Must identify at least 3 causes including rate restrictions and timing",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-008", "module": "front_office", "category": "troubleshooting",
            "difficulty": "intermediate",
            "question": "A room move operation (putRoomMoves) succeeds in the API but the housekeeping system still shows the old room as occupied. What is happening?",
            "expected_answer": "1. The housekeeping system may be caching room status and not refreshing from the API. 2. The old room's housekeeping status needs to be updated to Dirty after the move. 3. The integration should call setRoomCondition or putRoomStatus on the old room to update its status. 4. Business events/streaming may need to be configured to notify the housekeeping system of room moves.",
            "required_operations": [], "evaluation_criteria": "Must identify that housekeeping status update is a separate step from the room move",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-009", "module": "housekeeping", "category": "troubleshooting",
            "difficulty": "basic",
            "question": "A maintenance request is created via the API but cannot be resolved using the resolve endpoint. What should be checked?",
            "expected_answer": "1. The maintenance ID must be valid and correspond to an open/active maintenance request. 2. The maintenance may have already been resolved. 3. The correct endpoint is PUT /hsk/v1/hotels/{hotelId}/maintenances/{maintenanceId}/resolve. 4. The room associated with the maintenance must exist.",
            "required_operations": [], "evaluation_criteria": "Must identify valid maintenance ID and active status as prerequisites",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-010", "module": "multi", "category": "troubleshooting",
            "difficulty": "advanced",
            "question": "A contactless guest journey integration works in the test environment but fails in production. The OAuth token is obtained successfully, but all subsequent API calls return 403 Forbidden. What should be investigated?",
            "expected_answer": "1. The application's scope/permissions in production may differ from test — check the OHIP Developer Portal for granted scopes. 2. The x-app-key may be different between environments. 3. The hotel/property ID may not be authorized for the application in production. 4. The OAuth scope parameter may be incorrect for the production environment. 5. Rate limiting or IP whitelisting may be in effect in production. 6. The application may not be approved/activated in the production environment.",
            "required_operations": [], "evaluation_criteria": "Must identify scope/permission differences between environments as the primary suspect and mention at least 3 other causes",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-011", "module": "front_office", "category": "troubleshooting",
            "difficulty": "intermediate",
            "question": "An integration posts a billing payment to settle a guest's folio, but getFolios still shows an outstanding balance. The payment amount matched the total charges. What is wrong?",
            "expected_answer": "1. New charges may have been posted between reading the balance and posting the payment (minibar, phone, etc.). 2. The payment may have been posted to a different folio window than where the charges reside. 3. Tax calculations may have added additional amounts after the balance was read. 4. The payment transaction code may be incorrect, causing it to be treated as a charge instead of a payment. 5. Currency rounding differences may leave a small residual balance.",
            "required_operations": [], "evaluation_criteria": "Must identify timing of new charges and folio window mismatch as likely causes",
            "source": "postman_workflow_derived",
        },
        {
            "id": "wf-tbl-012", "module": "reservations", "category": "troubleshooting",
            "difficulty": "basic",
            "question": "getAvailableUpsells returns no upsell offers for a reservation that should be eligible. What should be checked?",
            "expected_answer": "1. Upsell rules may not be configured for the reservation's current room type (check upsell rules via getUpsellRules). 2. The reservation status may disqualify it — checked-in, cancelled, no-show, or waitlist reservations cannot be upsold. 3. Shared reservations are not eligible for upsells. 4. Fixed-rate reservations cannot be upsold. 5. No higher room types may be available for the reservation dates.",
            "required_operations": [], "evaluation_criteria": "Must identify status restrictions and configuration requirements",
            "source": "postman_workflow_derived",
        },
    ]


def merge_and_deduplicate(seed_path: Path, new_tasks: list[dict]) -> list[dict]:
    """Merge seed tasks with newly generated tasks, deduplicate by ID."""
    existing = []
    if seed_path.exists():
        with open(seed_path) as f:
            for line in f:
                existing.append(json.loads(line))

    seen_ids = {t["id"] for t in existing}
    merged = list(existing)

    for task in new_tasks:
        if task["id"] not in seen_ids:
            merged.append(task)
            seen_ids.add(task["id"])

    return merged


def print_stats(tasks: list[dict]):
    by_cat = {}
    by_mod = {}
    by_diff = {}
    for t in tasks:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
        by_mod[t["module"]] = by_mod.get(t["module"], 0) + 1
        by_diff[t["difficulty"]] = by_diff.get(t["difficulty"], 0) + 1

    print(f"\nTotal: {len(tasks)} tasks")
    print("\nBy category:")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:25s} {v}")
    print("\nBy module:")
    for k, v in sorted(by_mod.items()):
        print(f"  {k:25s} {v}")
    print("\nBy difficulty:")
    for k, v in sorted(by_diff.items()):
        print(f"  {k:25s} {v}")


def main():
    workflows = load_workflows()
    print(f"Loaded {len(workflows)} Postman workflows")

    orch_tasks = generate_workflow_orchestration_tasks(workflows)
    print(f"Generated {len(orch_tasks)} orchestration tasks from workflows")

    cfg_tasks = generate_workflow_config_tasks(workflows)
    print(f"Generated {len(cfg_tasks)} config advisory tasks")

    tbl_tasks = generate_workflow_troubleshooting_tasks()
    print(f"Generated {len(tbl_tasks)} troubleshooting tasks")

    new_tasks = orch_tasks + cfg_tasks + tbl_tasks

    seed_path = PROJECT_ROOT / "data" / "benchmark_seed.jsonl"
    all_tasks = merge_and_deduplicate(seed_path, new_tasks)

    with open(OUTPUT, "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"\nWritten to {OUTPUT}")
    print_stats(all_tasks)


if __name__ == "__main__":
    main()
