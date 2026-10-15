"""Offline mechanical checks; prose coverage is not a validation of claim grounding."""

import argparse
import json
from pathlib import Path
import re


def audit_assessment(answer):
    text = answer.get("assessment_text") or ""
    options = answer.get("reference", {}).get("options", {})
    labels = list(options)
    scores, seen = {}, set()
    if labels:
        pattern = r"(?:^|[\n.;])\s*(?:[-*]\s*)?(?:Option\s+)?(?:\*\*)?(" + \
                  "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True)) + \
                  r")(?:\*\*)?\s*(?:[:.)—(\-]|\s)"
        for match in re.finditer(pattern, text):
            label = match.group(1)
            seen.add(label)
            score = re.match(r"\s*(?:score\s*[:=]?\s*)?([0-3])(?!\.\d)(?:\s*/\s*3)?(?=\s|[,:.;)—\-]|$)", text[match.end():], re.I)
            if score:
                scores[label] = int(score.group(1))
    consistency = None
    if labels and set(scores) == set(labels) and answer.get("letter") in scores:
        consistency = scores[answer["letter"]] == max(scores.values())
    current = [r for r in answer.get("trace", []) if r.get("phase") in ("answer", "answer_recovery") and r.get("status") == "accepted"]
    refs = set(current[-1].get("evidence_ids", [])) | set(current[-1].get("visual_ids", [])) if current else None
    cited = set(re.findall(r"(?<![\w:])(?:visual:[\w:-]+|frame:\d+|t\d+)(?!\w)", text))
    return {"question_id": answer.get("question_id"), "assessment_present": bool(text),
            "labels_found": sorted(seen), "label_coverage": len(seen) / len(labels) if labels else None,
            "mechanically_complete": set(seen) == set(labels) if labels else None,
            "parsed_scores": scores, "assessment_answer_consistent": consistency,
            "citations_not_in_final_packet": sorted(cited - refs) if refs is not None else None,
            "grounding_review": "manual review required; these checks do not validate claim support"}


def audit_directory(directory):
    return [audit_assessment(json.loads(path.read_text())) for path in sorted((Path(directory) / "answers").glob("*.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = audit_directory(args.results)
    payload = {"questions": len(records), "records": records, "scope": "mechanical audit, not a grounding judge"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
