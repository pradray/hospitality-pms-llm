"""Parse scoped OpenAPI (Swagger 2.0) specs into per-endpoint chunk documents."""

import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT.parent  # ../Dissertation
SPEC_DIR = DATA_ROOT / "data/api-specs/hospitality-api-docs/rest-api-specs/property/v1"
OUTPUT_DIR = PROJECT_ROOT / "output/api_chunks"

SCOPED_SPECS = ["rsv", "rsvasync", "rsvcfg", "crm", "crmasync", "crmcfg", "fof", "fofcfg", "hsk"]

MODULE_MAP = {
    "rsv": "reservations", "rsvasync": "reservations", "rsvcfg": "reservations",
    "crm": "crm", "crmasync": "crm", "crmcfg": "crm",
    "fof": "front_office", "fofcfg": "front_office",
    "hsk": "housekeeping",
}

SPEC_TYPE_MAP = {
    "rsv": "sync", "rsvasync": "async", "rsvcfg": "config",
    "crm": "sync", "crmasync": "async", "crmcfg": "config",
    "fof": "sync", "fofcfg": "config",
    "hsk": "sync",
}


def resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a $ref pointer like '#/parameters/authKey' or '#/definitions/Foo'."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        node = node[part]
    return node


def simplify_schema(spec: dict, schema: dict, depth: int = 0) -> dict | str | None:
    """Recursively resolve $ref and simplify a schema to a readable dict, max 3 levels deep."""
    if depth > 3:
        return "..."
    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        try:
            resolved = resolve_ref(spec, schema["$ref"])
        except KeyError:
            return f"${ref_name} (unresolved)"
        return simplify_schema(spec, resolved, depth + 1)

    if "allOf" in schema:
        merged = {}
        for sub in schema["allOf"]:
            result = simplify_schema(spec, sub, depth)
            if isinstance(result, dict):
                merged.update(result)
        return merged

    if schema.get("type") == "array" and "items" in schema:
        return {"type": "array", "items": simplify_schema(spec, schema["items"], depth + 1)}

    if schema.get("type") == "object" or "properties" in schema:
        props = {}
        for name, prop in schema.get("properties", {}).items():
            props[name] = simplify_schema(spec, prop, depth + 1)
        result = {"type": "object"}
        if props:
            result["properties"] = props
        required = schema.get("required")
        if required:
            result["required"] = required
        return result

    simple = {}
    for key in ("type", "format", "enum", "description", "default", "minimum", "maximum", "pattern"):
        if key in schema:
            simple[key] = schema[key]
    return simple if simple else schema.get("type", "unknown")


def extract_parameters(spec: dict, raw_params: list) -> tuple[list[dict], dict | None]:
    """Split params into query/path/header params and body schema."""
    params = []
    body_schema = None

    for p in raw_params:
        if "$ref" in p:
            try:
                p = resolve_ref(spec, p["$ref"])
            except KeyError:
                continue

        if p.get("in") == "body":
            body_schema = simplify_schema(spec, p.get("schema", {}))
            continue

        # Skip auth/infra headers — not useful for domain understanding
        if p.get("in") == "header" and p.get("name") in (
            "authorization", "x-app-key", "x-hotelid", "Accept-Language",
            "x-request-id", "x-originating-application", "externalData",
        ):
            continue

        params.append({
            "name": p.get("name"),
            "in": p.get("in"),
            "type": p.get("type", p.get("schema", {}).get("type", "string")),
            "required": p.get("required", False),
            "description": p.get("description", ""),
        })

    return params, body_schema


def extract_response_schema(spec: dict, responses: dict) -> dict | None:
    """Extract the success response schema (200 or 201)."""
    for code in ("200", "201"):
        resp = responses.get(code, {})
        schema = resp.get("schema")
        if schema:
            return simplify_schema(spec, schema)
    return None


def extract_error_codes(responses: dict) -> list[str]:
    """List non-success HTTP status codes."""
    return sorted(c for c in responses if c not in ("200", "201", "204"))


def strip_html(text: str) -> str:
    """Remove HTML tags from description strings."""
    import re
    return re.sub(r"<[^>]+>", " ", text).strip()


def parse_spec(spec_name: str) -> list[dict]:
    spec_path = SPEC_DIR / f"{spec_name}.json"
    with open(spec_path) as f:
        spec = json.load(f)

    base_path = spec.get("basePath", "")
    module = MODULE_MAP[spec_name]
    spec_type = SPEC_TYPE_MAP[spec_name]
    chunks = []

    for path, methods in spec.get("paths", {}).items():
        for method, endpoint in methods.items():
            if not isinstance(endpoint, dict) or method.startswith("x-"):
                continue

            operation_id = endpoint.get("operationId", "")
            summary = endpoint.get("summary", "")
            description = strip_html(endpoint.get("description", ""))
            tags = endpoint.get("tags", [])

            params, body_schema = extract_parameters(spec, endpoint.get("parameters", []))
            response_schema = extract_response_schema(spec, endpoint.get("responses", {}))
            error_codes = extract_error_codes(endpoint.get("responses", {}))

            full_path = f"{base_path}{path}"
            chunk_id = hashlib.md5(f"{spec_name}:{method}:{path}".encode()).hexdigest()[:12]

            # Build the text representation for embedding
            text_parts = [
                f"{method.upper()} {full_path}",
                f"Operation: {operation_id}",
                f"Summary: {summary}",
            ]
            if description and description != summary:
                text_parts.append(f"Description: {description}")
            if tags:
                text_parts.append(f"Tags: {', '.join(tags)}")
            if params:
                param_lines = [f"  - {p['name']} ({p['in']}, {p['type']}, {'required' if p['required'] else 'optional'}): {p['description']}" for p in params]
                text_parts.append("Parameters:\n" + "\n".join(param_lines))
            if body_schema:
                text_parts.append(f"Request Body: {json.dumps(body_schema, indent=2, default=str)}")
            if response_schema:
                text_parts.append(f"Response Schema: {json.dumps(response_schema, indent=2, default=str)}")
            if error_codes:
                text_parts.append(f"Error Codes: {', '.join(error_codes)}")

            chunk = {
                "id": chunk_id,
                "module": module,
                "spec_name": spec_name,
                "spec_type": spec_type,
                "doc_type": "api_spec",
                "method": method.upper(),
                "path": full_path,
                "operation_id": operation_id,
                "summary": summary,
                "description": description,
                "tags": tags,
                "parameters": params,
                "body_schema": body_schema,
                "response_schema": response_schema,
                "error_codes": error_codes,
                "text": "\n".join(text_parts),
            }
            chunks.append(chunk)

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for spec_name in SCOPED_SPECS:
        chunks = parse_spec(spec_name)
        all_chunks.extend(chunks)
        print(f"{spec_name:12s}  {len(chunks):4d} endpoints")

    # Write combined output
    output_path = OUTPUT_DIR / "all_endpoints.jsonl"
    with open(output_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nTotal: {len(all_chunks)} endpoint chunks → {output_path}")

    # Stats
    by_module = {}
    by_method = {}
    for c in all_chunks:
        by_module[c["module"]] = by_module.get(c["module"], 0) + 1
        by_method[c["method"]] = by_method.get(c["method"], 0) + 1

    print("\nBy module:")
    for m, count in sorted(by_module.items()):
        print(f"  {m:20s} {count:4d}")
    print("\nBy HTTP method:")
    for m, count in sorted(by_method.items()):
        print(f"  {m:8s} {count:4d}")


if __name__ == "__main__":
    main()
