"""Shared state for the CDSS LangGraph flow (Step 11).

Rules:
- Every node returns ONLY the keys it owns, never the whole state.
  vector_rag and knowledge_graph run in the same step, so if both wrote the
  same key without a reducer LangGraph would raise InvalidUpdateError.
- errors uses operator.add: a node returns {"errors": [item]} and LangGraph
  appends it to the list (never overwrites), even when two nodes add at once.
- evidence_output uses merge_dicts: each Evidence Scanner pass returns only
  the newly scanned diseases, which are added to the existing dict.
"""

import operator
from typing import Annotated, Any, Optional, TypedDict


def merge_dicts(old: Optional[dict], new: Optional[dict]) -> dict:
    """Reducer: add new keys to the existing dict instead of replacing it."""
    merged = dict(old or {})
    merged.update(new or {})
    return merged


class CDSSState(TypedDict, total=False):
    patient_text: str
    disabled_agents: list                               # ablation switches for this run (rag, kg, evidence, optimizer)
    clarifier_output: Optional[dict]                    # EXACTLY the 5 keys, read by later agents
    clarifier_details: Optional[dict]                   # extra Clarifier info (entities, demographics, ...)
    rag_output: Optional[dict]                          # overwritten on each loop
    kg_output: Optional[dict]                           # written once (KG runs once)
    kg_safety_messages: list                            # flat list made by the KG wrapper
    evidence_output: Annotated[dict, merge_dicts]       # disease name -> scanner result
    evidence_groq: Annotated[dict, merge_dicts]         # disease name -> her Groq jobs / fallbacks (real mode)
    fusion_output: Optional[dict]
    rag_confidence: Optional[float]                     # confidence_1.score (C1), 0-1
    rag_top_k: int                                      # 5 -> 10 -> 15
    loop_count: int                                     # max 2
    final_output: Optional[dict]
    errors: Annotated[list, operator.add]               # every node adds, never overwrites


def initial_state(patient_text: str, top_k_start: int = 5, disabled_agents=()) -> CDSSState:
    return {
        "patient_text": patient_text,
        "disabled_agents": sorted(disabled_agents),
        "rag_top_k": top_k_start,
        "loop_count": 0,
        "evidence_output": {},
        "evidence_groq": {},
        "kg_safety_messages": [],
        "errors": [],
    }
