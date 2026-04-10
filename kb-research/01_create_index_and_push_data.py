"""Create an Azure AI Search index and push sample safety-manual documents.

This script demonstrates the "push model" — creating an index schema
(with semantic configuration) and uploading documents directly via
SearchClient.upload_documents(), without using an indexer.

Usage:
    python 01_create_index_and_push_data.py
    python 01_create_index_and_push_data.py --index-name my-index --delete-first
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
)

DEFAULT_INDEX_NAME = "kb-research-demo"


# ── Sample documents ────────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "id": "ppe-001",
        "title": "Personal Protective Equipment (PPE) Requirements",
        "category": "PPE",
        "source_url": "https://contoso-energy.example.com/safety/ppe-requirements",
        "content": (
            "All Contoso Energy employees and contractors must wear the appropriate "
            "personal protective equipment (PPE) when entering operational areas. "
            "At a minimum, hard hats, safety glasses, steel-toed boots, and high-visibility "
            "vests are required in all plant zones designated as 'active operations.'\n\n"
            "Hearing protection rated at NRR 25 dB or higher must be worn in areas where "
            "noise levels exceed 85 dBA. Signage at zone entry points indicates the required "
            "hearing protection level. Employees who fail to wear required hearing protection "
            "are subject to immediate removal from the work area.\n\n"
            "Respiratory protection is mandatory in confined spaces, chemical handling areas, "
            "and any zone where airborne contaminant levels exceed permissible exposure limits "
            "(PELs). Only NIOSH-approved respirators that have been fit-tested within the "
            "last 12 months may be used. Records of fit testing must be maintained by the "
            "site safety coordinator."
        ),
    },
    {
        "id": "evac-002",
        "title": "Emergency Evacuation Procedures",
        "category": "Emergency Response",
        "source_url": "https://contoso-energy.example.com/safety/evacuation",
        "content": (
            "When the site-wide evacuation alarm sounds — three short blasts followed by one "
            "long blast — all personnel must immediately stop work, secure any hazardous "
            "processes if it can be done in under 30 seconds, and proceed to the nearest "
            "designated muster point. Muster points are identified by green assembly-point "
            "signs and are located at least 150 meters from any process unit.\n\n"
            "Each department supervisor or designated safety warden is responsible for "
            "conducting a headcount at the muster point within five minutes of the alarm. "
            "Any unaccounted personnel must be reported to the Incident Commander immediately "
            "via the emergency radio channel (Channel 9).\n\n"
            "Evacuation routes must remain clear and unobstructed at all times. Monthly "
            "inspections of evacuation pathways are conducted by the site safety team. "
            "Blocked or compromised routes must be reported through the Contoso Safety "
            "Management System (SMS) within 24 hours."
        ),
    },
    {
        "id": "csentry-003",
        "title": "Confined Space Entry Procedures",
        "category": "Confined Spaces",
        "source_url": "https://contoso-energy.example.com/safety/confined-space",
        "content": (
            "No employee or contractor may enter a permit-required confined space without "
            "a valid Confined Space Entry Permit signed by the area supervisor and the site "
            "safety officer. The permit must specify atmospheric testing results, required "
            "PPE, communication procedures, and rescue provisions.\n\n"
            "Atmospheric monitoring must be performed before entry and continuously during "
            "occupancy. Acceptable conditions are: oxygen between 19.5%–23.5%, LEL below 10%, "
            "hydrogen sulfide below 10 ppm, and carbon monoxide below 25 ppm. If any reading "
            "falls outside these limits, all entrants must exit immediately and the entry "
            "permit is void.\n\n"
            "A trained attendant must remain at the entry point for the entire duration of "
            "the confined space operation. The attendant must maintain continuous communication "
            "with entrants, monitor atmospheric readings, and be prepared to initiate rescue "
            "procedures without entering the space. Non-entry rescue equipment (tripod, "
            "winch, and retrieval line) must be set up before any entrant crosses the plane "
            "of the opening."
        ),
    },
    {
        "id": "hotwork-004",
        "title": "Hot Work Permit Requirements",
        "category": "Hot Work",
        "source_url": "https://contoso-energy.example.com/safety/hot-work",
        "content": (
            "Hot work — including welding, cutting, brazing, grinding, and any operation "
            "producing sparks or open flame — requires a Hot Work Permit issued by the area "
            "authority. The permit is valid for a single shift only and must be renewed for "
            "each subsequent shift.\n\n"
            "Before hot work begins, a fire watch must inspect the work area within a "
            "10-meter radius to remove or protect combustible materials. Fire-resistant "
            "blankets or shields must be used to protect any combustibles that cannot be "
            "relocated. A dedicated fire watch with a charged extinguisher must remain on "
            "site during the work and for a minimum of 60 minutes after hot work ceases.\n\n"
            "Hot work is prohibited within 15 meters of any storage vessel containing "
            "flammable liquids or gases unless the vessel has been drained, purged, and "
            "gas-tested to below 1% LEL. The area supervisor must verify purge certificates "
            "before signing the Hot Work Permit."
        ),
    },
    {
        "id": "chem-005",
        "title": "Chemical Handling and Storage",
        "category": "Chemical Safety",
        "source_url": "https://contoso-energy.example.com/safety/chemical-handling",
        "content": (
            "All chemicals used at Contoso Energy facilities must have a current Safety Data "
            "Sheet (SDS) on file and accessible within the work area. Employees must review "
            "the SDS for any chemical before first use and whenever handling procedures "
            "change. SDSs are available electronically through the Contoso Safety Portal and "
            "in hardcopy binders located at each chemical storage area.\n\n"
            "Incompatible chemicals must be stored in separate secondary containment areas "
            "with a minimum separation of 3 meters. Acids and bases must never share the "
            "same containment. Oxidizers must be stored away from organic materials and "
            "flammable substances. All chemical containers must be labeled with the GHS- "
            "compliant label including the product name, hazard pictograms, signal word, "
            "and precautionary statements.\n\n"
            "Spill kits appropriate to the chemicals in use must be staged within 15 meters "
            "of any chemical handling or storage area. Employees must be trained in the "
            "correct use of spill kits and understand the spill reporting thresholds defined "
            "in the Contoso Environmental Compliance Manual."
        ),
    },
    {
        "id": "fall-006",
        "title": "Fall Protection Standards",
        "category": "Fall Protection",
        "source_url": "https://contoso-energy.example.com/safety/fall-protection",
        "content": (
            "Fall protection is required for any work performed at a height of 1.8 meters "
            "(6 feet) or more above a lower level. Acceptable fall protection systems include "
            "guardrail systems, personal fall arrest systems (PFAS), or safety net systems. "
            "The method of fall protection must be specified in the task-specific Job Safety "
            "Analysis (JSA) before work begins.\n\n"
            "All personal fall arrest equipment — harnesses, lanyards, self-retracting "
            "lifelines, and anchorage connectors — must be inspected by a competent person "
            "before each use. Equipment showing signs of wear, damage, or deterioration must "
            "be immediately removed from service and tagged 'Do Not Use.' Annual recertification "
            "of all fall arrest equipment is mandatory.\n\n"
            "Workers using personal fall arrest systems must be trained in proper donning, "
            "anchor point selection, free-fall distance calculations, and suspension trauma "
            "risks. Rescue plans must be in place before elevated work begins, with rescue "
            "capability demonstrated to arrive within 10 minutes of a fall event."
        ),
    },
    {
        "id": "elec-007",
        "title": "Electrical Safety and Lockout/Tagout",
        "category": "Electrical Safety",
        "source_url": "https://contoso-energy.example.com/safety/electrical-loto",
        "content": (
            "All electrical work at Contoso Energy must comply with NFPA 70E requirements "
            "for electrical safety in the workplace. Only qualified electrical workers — those "
            "who have completed Contoso's Electrical Safety Qualification Program — may perform "
            "work on or near energized conductors or circuit parts operating at 50 volts or "
            "more.\n\n"
            "Lockout/Tagout (LOTO) procedures must be followed whenever servicing or "
            "maintaining equipment where unexpected energization could cause injury. Each "
            "authorized worker must apply their own individual lock and tag. Group LOTO "
            "procedures require a designated principal authorized employee who coordinates "
            "the application and removal of a group lockbox.\n\n"
            "Arc flash hazard analysis must be performed for all electrical panels, switchgear, "
            "and motor control centers. Equipment must be labeled with the arc flash boundary, "
            "incident energy level, and required PPE category. Workers must wear arc-rated "
            "clothing and face protection appropriate to the incident energy level before "
            "opening any energized panel."
        ),
    },
    {
        "id": "incident-008",
        "title": "Incident Reporting and Investigation",
        "category": "Incident Management",
        "source_url": "https://contoso-energy.example.com/safety/incident-reporting",
        "content": (
            "All workplace incidents — including injuries, near-misses, property damage, and "
            "environmental releases — must be reported to the shift supervisor within 15 "
            "minutes of occurrence. The supervisor must enter the initial report into the "
            "Contoso Safety Management System (SMS) within two hours. Failure to report "
            "incidents in a timely manner is a serious safety violation.\n\n"
            "Incidents classified as 'serious' (lost-time injuries, hospitalizations, "
            "amputations, or environmental releases exceeding reportable quantities) trigger "
            "a formal Root Cause Analysis (RCA) investigation. The RCA team must include the "
            "area supervisor, a safety representative, a maintenance representative, and at "
            "least one front-line worker from the affected crew.\n\n"
            "Investigation findings and corrective actions must be documented in the SMS "
            "within 10 business days of the incident. Corrective actions are tracked to "
            "completion with assigned owners and due dates. The site safety committee reviews "
            "all open corrective actions monthly and escalates overdue items to plant "
            "management."
        ),
    },
    {
        "id": "scaffold-009",
        "title": "Scaffolding Safety Requirements",
        "category": "Fall Protection",
        "source_url": "https://contoso-energy.example.com/safety/scaffolding",
        "content": (
            "All scaffolding at Contoso Energy facilities must be erected, modified, and "
            "dismantled under the supervision of a scaffold-competent person who has completed "
            "the Contoso Scaffold Competent Person Training Program. Scaffold designs for "
            "heights exceeding 38 meters or unusual configurations must be reviewed and "
            "approved by a licensed professional engineer.\n\n"
            "Before any scaffold is placed in service, the competent person must inspect it "
            "and attach a green 'Scaffold Ready' tag. Scaffolds that are incomplete or under "
            "modification must display a red 'Do Not Use' tag. Daily inspections are required "
            "before each shift, after any weather event with winds exceeding 40 km/h, and "
            "after any seismic event.\n\n"
            "All scaffold platforms must be fully planked with no gaps exceeding 2.5 cm "
            "between planks. Guardrails, mid-rails, and toe boards are required on all open "
            "sides and ends of platforms more than 1.8 meters above the ground. Access to "
            "scaffold platforms must be via built-in ladders, stair towers, or equivalent "
            "safe access — climbing on cross-braces is strictly prohibited."
        ),
    },
    {
        "id": "heat-010",
        "title": "Heat Stress Prevention Program",
        "category": "Health & Wellness",
        "source_url": "https://contoso-energy.example.com/safety/heat-stress",
        "content": (
            "Contoso Energy's Heat Stress Prevention Program applies to all outdoor and "
            "indoor work where the Wet Bulb Globe Temperature (WBGT) index exceeds 28°C. "
            "Supervisors must monitor WBGT readings hourly during warm-weather months and "
            "adjust work-rest cycles according to the Contoso Heat Stress Action Level "
            "Chart.\n\n"
            "When the WBGT exceeds 30°C, a mandatory work-rest regimen of 45 minutes work "
            "to 15 minutes rest in a shaded or air-conditioned area is enforced. Workers "
            "must have continuous access to cool drinking water — a minimum of one liter per "
            "worker per hour must be provided. Electrolyte replacement drinks should be "
            "available but are not a substitute for water.\n\n"
            "All supervisors and employees working in heat-exposed areas must complete annual "
            "heat stress awareness training that covers recognition of heat-related illness "
            "symptoms, buddy-system monitoring, acclimatization schedules for new or returning "
            "workers, and emergency first-aid procedures for heat stroke and heat exhaustion."
        ),
    },
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_env() -> None:
    """Load .env from the current directory or one level up."""
    if os.path.isfile(".env"):
        load_dotenv(".env")
    elif os.path.isfile(os.path.join("..", ".env")):
        load_dotenv(os.path.join("..", ".env"))
    else:
        # Try the script's own directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, ".env")
        parent_env = os.path.join(os.path.dirname(script_dir), ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path)
        elif os.path.isfile(parent_env):
            load_dotenv(parent_env)


def _get_credential():
    """Return the appropriate credential based on AZURE_SEARCH_AUTH_TYPE."""
    auth_type = os.environ.get("AZURE_SEARCH_AUTH_TYPE", "token").lower()
    if auth_type == "key":
        api_key = os.environ.get("AZURE_SEARCH_API_KEY", "")
        if not api_key:
            print("ERROR: AZURE_SEARCH_AUTH_TYPE=key but AZURE_SEARCH_API_KEY is empty.")
            sys.exit(1)
        print(f"  Auth type: API key")
        return AzureKeyCredential(api_key)
    else:
        from azure.identity import DefaultAzureCredential

        print(f"  Auth type: DefaultAzureCredential (Entra ID)")
        return DefaultAzureCredential(
            exclude_managed_identity_credential=True,
            exclude_shared_token_cache_credential=True,
        )


# ── Index schema ────────────────────────────────────────────────────────────


def _build_index(index_name: str) -> SearchIndex:
    """Build the SearchIndex definition with fields and semantic config."""
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            retrievable=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            retrievable=True,
        ),
        SearchableField(
            name="category",
            type=SearchFieldDataType.String,
            retrievable=True,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="source_url",
            type=SearchFieldDataType.String,
            retrievable=True,
            filterable=True,
        ),
    ]

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="category")],
        ),
    )

    semantic_search = SemanticSearch(
        default_configuration_name="my-semantic-config",
        configurations=[semantic_config],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=semantic_search,
    )


# ── Main logic ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an Azure AI Search index and push sample safety-manual documents."
    )
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        help=f"Name of the search index (default: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--delete-first",
        action="store_true",
        help="Delete the index if it already exists, then recreate it",
    )
    args = parser.parse_args()

    # ── 1. Load environment ─────────────────────────────────────────────
    _load_env()
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    if not endpoint:
        print("ERROR: AZURE_SEARCH_ENDPOINT is not set. Check your .env file.")
        sys.exit(1)

    print("=" * 60)
    print("Azure AI Search — Index Creation & Document Push")
    print("=" * 60)
    print(f"  Endpoint:   {endpoint}")
    print(f"  Index name: {args.index_name}")
    credential = _get_credential()
    print()

    # ── 2. Create index client ──────────────────────────────────────────
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    # ── 3. Optionally delete existing index ─────────────────────────────
    if args.delete_first:
        try:
            index_client.delete_index(args.index_name)
            print(f"[✓] Deleted existing index '{args.index_name}'")
        except Exception:
            print(f"[–] Index '{args.index_name}' did not exist (nothing to delete)")

    # ── 4. Create the index ─────────────────────────────────────────────
    print(f"\n[→] Creating index '{args.index_name}' ...")
    index_def = _build_index(args.index_name)
    try:
        index_client.create_or_update_index(index_def)
        print(f"[✓] Index '{args.index_name}' is ready")
    except Exception as exc:
        print(f"[✗] Failed to create index: {exc}")
        sys.exit(1)

    # ── 5. Push documents ───────────────────────────────────────────────
    print(f"\n[→] Uploading {len(SAMPLE_DOCUMENTS)} documents ...")
    search_client = SearchClient(
        endpoint=endpoint,
        index_name=args.index_name,
        credential=credential,
    )
    result = search_client.upload_documents(documents=SAMPLE_DOCUMENTS)

    succeeded = sum(1 for r in result if r.succeeded)
    failed = sum(1 for r in result if not r.succeeded)
    print(f"[✓] Upload complete — {succeeded} succeeded, {failed} failed")
    if failed:
        for r in result:
            if not r.succeeded:
                print(f"    FAILED: {r.key} — {r.error_message}")

    # ── 6. Verify with a search query ───────────────────────────────────
    print("\n[→] Waiting a few seconds for indexing to complete ...")
    time.sleep(3)

    print("[→] Running verification search for 'safety' ...")
    results = search_client.search(search_text="safety", top=5)
    hits = list(results)
    print(f"[✓] Verification search returned {len(hits)} result(s):")
    for hit in hits:
        score = hit.get("@search.score", "n/a")
        print(f"    • [{score:.4f}] {hit['title']}  (id={hit['id']})")

    print("\n" + "=" * 60)
    print("Done! Index is ready for queries.")
    print("=" * 60)


if __name__ == "__main__":
    main()
