"""Parse Postman collections into chunks for embedding.

Two collection types:
1. Property collection (by module) — each request becomes a chunk with
   method, path, query params, headers, and sample request body.
2. Workflows collection — each workflow becomes a chunk showing the
   full multi-step API sequence with request bodies.

Output: output/postman_chunks/all_postman_chunks.jsonl
"""

import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent
POSTMAN_DIR = DATA_ROOT / "data/api-specs/hospitality-api-docs/postman-collections/property"
OUTPUT_DIR = PROJECT_ROOT / "output" / "postman_chunks"

MAX_BODY_CHARS = 2000

# Map Postman folder names to our 4-module taxonomy
FOLDER_MODULE_MAP = {
    "Reservations (RSV)": "reservations",
    "Reservation Master Data Management  (RSV Config)": "reservations",
    "Rate Plan Management (RTP)": "reservations",
    "Blocks (BLK)": "reservations",
    "Blocks Configuration (BLK Config)": "reservations",
    "Availability (PAR) (Price, Availability, Rate)": "reservations",
    "Channel Configuration (CHL)": "reservations",
    "Profiles  (CRM)": "crm",
    "Profile Configuration (CRM Config)": "crm",
    "Customer Management Service (CMS)": "crm",
    "Activity (ACT)": "crm",
    "Activity Configuration (ACT Config)": "crm",
    "Front Desk Operations (FOF)": "front_office",
    "Front Desk Configuration (FOF Config)": "front_office",
    "Cashiering (CSH)": "front_office",
    "Accounts Receivables (ARS)": "front_office",
    "Housekeeping (HSK)": "housekeeping",
    "Inventory (INV)": "housekeeping",
    "Room Configuration (RM Config)": "housekeeping",
    "Room Rotation (RMR)": "housekeeping",
    "Room Rotation Configuration (RMR Config)": "housekeeping",
    "Event Management (EVM)": "reservations",
    "Event Configuration  (EVM Config)": "reservations",
    "Enterprise Configuration (ENT Config)": "multi",
    "List Of Values (LOV)": "multi",
    "Integration (INT) - Business Events": "multi",
    "Integration Configuration (INT Config)": "multi",
    "Asyncronous APIs": "multi",
}

WORKFLOW_MODULE_MAP = {
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
    "Create a Room Class": "housekeeping",
    "Create a Room Type": "housekeeping",
}


def make_id(prefix: str, text: str) -> str:
    return prefix + hashlib.md5(text.encode()).hexdigest()[:10]


def extract_url(url_obj) -> tuple[str, list[dict]]:
    if isinstance(url_obj, str):
        return url_obj, []
    path = "/" + "/".join(url_obj.get("path", []))
    query = []
    for q in url_obj.get("query", []):
        if q.get("disabled"):
            continue
        query.append({"key": q["key"], "value": q.get("value", "")})
    return path, query


def truncate_body(body_raw: str) -> str:
    if not body_raw:
        return ""
    body_raw = body_raw.strip()
    if len(body_raw) <= MAX_BODY_CHARS:
        return body_raw
    return body_raw[:MAX_BODY_CHARS] + "\n... (truncated)"


def flatten_requests(item, folder_path=""):
    """Recursively extract all requests from nested Postman items."""
    results = []
    name = item.get("name", "")
    current_path = f"{folder_path}/{name}" if folder_path else name

    if "request" in item:
        results.append((current_path, item))
    for sub in item.get("item", []):
        results.extend(flatten_requests(sub, current_path))
    return results


def parse_property_collection() -> list[dict]:
    """Parse the per-module property collection into per-request chunks."""
    path = POSTMAN_DIR / "oracle-hospitality-property.postman_collection.json"
    with open(path) as f:
        col = json.load(f)

    api_modules = col["item"][1]["item"]
    chunks = []

    for module_folder in api_modules:
        folder_name = module_folder["name"]
        module = FOLDER_MODULE_MAP.get(folder_name, "multi")

        requests = flatten_requests(module_folder)
        for req_path, item in requests:
            req = item["request"]
            method = req.get("method", "?")
            url_path, query_params = extract_url(req.get("url", {}))
            body_raw = truncate_body(req.get("body", {}).get("raw", ""))

            text_parts = [f"Postman Example: {item['name']}"]
            text_parts.append(f"Module: {folder_name}")
            text_parts.append(f"{method} {url_path}")

            if query_params:
                params_str = ", ".join(f"{q['key']}={q['value']}" for q in query_params[:10])
                text_parts.append(f"Query parameters: {params_str}")

            if body_raw:
                text_parts.append(f"Sample request body:\n{body_raw}")

            text = "\n".join(text_parts)

            chunks.append({
                "id": make_id("pm-", f"{method}{url_path}{item['name']}"),
                "module": module,
                "doc_type": "postman_example",
                "source_file": "oracle-hospitality-property.postman_collection.json",
                "section": req_path,
                "method": method,
                "path": url_path,
                "text": text,
            })

    return chunks


def parse_workflows_collection() -> list[dict]:
    """Parse the workflows collection into per-workflow chunks."""
    path = POSTMAN_DIR / "oracle-hospitality-property-workflows.postman_collection.json"
    with open(path) as f:
        col = json.load(f)

    workflows = col["item"][1]["item"]
    chunks = []

    for wf in workflows:
        wf_name = wf["name"]
        module = WORKFLOW_MODULE_MAP.get(wf_name, "multi")
        steps = wf.get("item", [])

        text_parts = [f"Workflow: {wf_name}"]
        text_parts.append(f"Steps: {len(steps)}")
        text_parts.append("")

        for i, step in enumerate(steps, 1):
            req = step.get("request", {})
            method = req.get("method", "?")
            url_path, query_params = extract_url(req.get("url", {}))
            body_raw = truncate_body(req.get("body", {}).get("raw", ""))

            text_parts.append(f"Step {i}: {step['name']}")
            text_parts.append(f"  {method} {url_path}")

            if query_params:
                params_str = ", ".join(f"{q['key']}={q['value']}" for q in query_params[:5])
                text_parts.append(f"  Query: {params_str}")

            if body_raw:
                # For workflows, include shorter bodies to keep total chunk size manageable
                short_body = truncate_body(body_raw[:800])
                text_parts.append(f"  Body:\n{short_body}")

            text_parts.append("")

        text = "\n".join(text_parts)

        # If a workflow is very long, split it
        if len(text) > 6000:
            mid = len(steps) // 2
            for part_idx, step_range in enumerate([(0, mid), (mid, len(steps))]):
                part_text_parts = [f"Workflow: {wf_name} (Part {part_idx + 1}/2)"]
                part_text_parts.append(f"Total steps: {len(steps)}")
                part_text_parts.append("")

                for i in range(step_range[0], step_range[1]):
                    step = steps[i]
                    req = step.get("request", {})
                    method = req.get("method", "?")
                    url_path, _ = extract_url(req.get("url", {}))
                    body_raw = req.get("body", {}).get("raw", "")

                    part_text_parts.append(f"Step {i + 1}: {step['name']}")
                    part_text_parts.append(f"  {method} {url_path}")
                    if body_raw:
                        part_text_parts.append(f"  Body:\n{truncate_body(body_raw[:600])}")
                    part_text_parts.append("")

                part_text = "\n".join(part_text_parts)
                chunks.append({
                    "id": make_id("wf-", f"{wf_name}-part{part_idx}"),
                    "module": module,
                    "doc_type": "postman_workflow",
                    "source_file": "oracle-hospitality-property-workflows.postman_collection.json",
                    "section": f"{wf_name} (Part {part_idx + 1}/2)",
                    "method": "",
                    "path": "",
                    "text": part_text,
                })
        else:
            chunks.append({
                "id": make_id("wf-", wf_name),
                "module": module,
                "doc_type": "postman_workflow",
                "source_file": "oracle-hospitality-property-workflows.postman_collection.json",
                "section": wf_name,
                "method": "",
                "path": "",
                "text": text,
            })

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    property_chunks = parse_property_collection()
    print(f"Property collection: {len(property_chunks)} request chunks")

    workflow_chunks = parse_workflows_collection()
    print(f"Workflows collection: {len(workflow_chunks)} workflow chunks")

    all_chunks = property_chunks + workflow_chunks

    # Stats
    by_module = {}
    by_type = {}
    for c in all_chunks:
        by_module[c["module"]] = by_module.get(c["module"], 0) + 1
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1

    output_path = OUTPUT_DIR / "all_postman_chunks.jsonl"
    with open(output_path, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks → {output_path}")
    print("\nBy module:")
    for k, v in sorted(by_module.items()):
        print(f"  {k:20s} {v}")
    print("\nBy type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
