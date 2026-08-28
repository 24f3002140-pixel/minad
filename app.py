from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

SAFE_MAX = 9007199254740991
INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]


def is_safe_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= SAFE_MAX


def is_positive_safe_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and 0 < x <= SAFE_MAX


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sorted_codes(codes):
    return sorted(set(codes), key=lambda x: x.encode("utf-8"))


def sorted_strings(values):
    return sorted(values, key=lambda x: x.encode("utf-8"))


def choose(data):
    policy = data.get("policy")
    candidates = data.get("candidates")
    if not isinstance(policy, dict) or not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError()

    by_name = {}
    for c in candidates:
        if not isinstance(c, dict):
            raise ValueError()
        name = c.get("name")
        if name not in INTERVENTIONS or name in by_name:
            raise ValueError()
        by_name[name] = c

    if set(by_name) != set(INTERVENTIONS):
        raise ValueError()

    required_policy = ["minQuality", "freshnessRequired", "maxLatencyMs",
                       "maxMemoryMb", "maxLabeledExamples", "maxTotalCost",
                       "horizonRequests"]
    if any(k not in policy for k in required_policy):
        raise ValueError()

    if not finite_number(policy["minQuality"]) or not 0 <= policy["minQuality"] <= 1:
        raise ValueError()
    if not isinstance(policy["freshnessRequired"], bool):
        raise ValueError()
    for k in ["maxLatencyMs", "maxMemoryMb", "maxLabeledExamples", "maxTotalCost"]:
        if not finite_number(policy[k]) or policy[k] < 0:
            raise ValueError()
    if not is_safe_int(policy["horizonRequests"]):
        raise ValueError()

    eligible, total_costs, reason_codes = [], {}, {}

    for name in INTERVENTIONS:
        c = by_name[name]
        required = ["available", "quality", "freshness", "latencyMs", "memoryMb",
                    "labeledExamples", "oneTimeCost", "recurringCost"]
        if any(k not in c for k in required):
            raise ValueError()
        if not isinstance(c["available"], bool):
            raise ValueError()
        if not finite_number(c["quality"]) or not 0 <= c["quality"] <= 1:
            raise ValueError()
        if not isinstance(c["freshness"], bool):
            raise ValueError()
        for k in ["latencyMs", "memoryMb", "labeledExamples", "oneTimeCost", "recurringCost"]:
            if not finite_number(c[k]) or c[k] < 0:
                raise ValueError()
        if not is_safe_int(c["labeledExamples"]):
            raise ValueError()

        total = round(c["oneTimeCost"] + policy["horizonRequests"] * c["recurringCost"], 12)
        total_costs[name] = total
        reasons = []

        if not c["available"]: reasons.append("UNAVAILABLE")
        if c["quality"] < policy["minQuality"]: reasons.append("QUALITY_FLOOR")
        if policy["freshnessRequired"] and not c["freshness"]: reasons.append("FRESHNESS_REQUIRED")
        if c["latencyMs"] > policy["maxLatencyMs"]: reasons.append("LATENCY_LIMIT")
        if c["memoryMb"] > policy["maxMemoryMb"]: reasons.append("MEMORY_LIMIT")
        if c["labeledExamples"] > policy["maxLabeledExamples"]: reasons.append("DATA_LIMIT")
        if total > policy["maxTotalCost"]: reasons.append("COST_LIMIT")

        reason_codes[name] = sorted_codes(reasons)
        if not reasons:
            eligible.append(name)

    return {
        "selected": eligible[0] if eligible else None,
        "eligible": eligible,
        "totalCosts": total_costs,
        "reasonCodes": reason_codes
    }


def repair(data):
    codes = []
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise ValueError()

    token_valid = True
    for t in tokens:
        if not isinstance(t, dict) or not is_safe_int(t.get("id")) \
           or t.get("role") not in ["system", "user", "assistant"] \
           or not isinstance(t.get("padding"), bool) \
           or not isinstance(t.get("text"), str):
            token_valid = False
            break

    if not token_valid:
        labels = [-100] * len(tokens)
        codes.append("INVALID_TOKEN")
    else:
        labels = [
            t["id"] if t["role"] == "assistant" and not t["padding"] else -100
            for t in tokens
        ]

    template_pass = data.get("templateApplications") == 1
    if not template_pass:
        codes.append("CHAT_TEMPLATE_COUNT")

    params = data.get("parameters")
    allowed = data.get("allowedTargets")
    parameter_valid = isinstance(params, list) and isinstance(allowed, list) and len(allowed) > 0

    if parameter_valid:
        parameter_valid = (
            len(set(allowed)) == len(allowed)
            and all(isinstance(x, str) and x for x in allowed)
        )

    names = set()
    trainable = []

    if isinstance(params, list):
        for p in params:
            if not isinstance(p, dict):
                parameter_valid = False
                continue
            name, target, numel = p.get("name"), p.get("target"), p.get("numel")
            if (not isinstance(name, str) or not isinstance(target, str)
                or not is_positive_safe_int(numel) or name in names):
                parameter_valid = False
                continue
            names.add(name)
            if (target in allowed and
                (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"))):
                trainable.append(p)

    if not any(
        isinstance(p, dict) and isinstance(p.get("name"), str)
        and isinstance(p.get("target"), str)
        and p["target"] in allowed
        and (p["name"].endswith(".lora_A.weight") or p["name"].endswith(".lora_B.weight"))
        for p in params if isinstance(params, list)
    ):
        parameter_valid = False

    if not parameter_valid:
        codes.append("INVALID_PARAMETER")

    trainable.sort(key=lambda p: p["name"].encode("utf-8"))
    trainable_names = [p["name"] for p in trainable]
    trainable_count = sum(p["numel"] for p in trainable) if all(
        is_positive_safe_int(p["numel"]) for p in trainable
    ) else 0

    if data.get("inferenceMode") is not False:
        codes.append("INFERENCE_MODE")

    artifact_files = data.get("artifactFiles")
    expected_files = ["adapter_config.json", "adapter_model.safetensors"]
    if (
        not isinstance(artifact_files, list)
        or len(artifact_files) != 2
        or sorted(artifact_files, key=lambda x: x.encode("utf-8")) != expected_files
    ):
        codes.append("ADAPTER_FILE_SET")
        adapter_files = sorted_strings([x for x in artifact_files if isinstance(x, str)]) if isinstance(artifact_files, list) else []
    else:
        adapter_files = expected_files

    checkpoint = data.get("checkpoint")
    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(k in checkpoint for k in ["model", "optimizer", "scheduler", "step", "rng", "dataPosition"])
    )
    if not checkpoint_complete:
        codes.append("INCOMPLETE_CHECKPOINT")

    base_revision = data.get("baseRevision")
    revision_valid = isinstance(base_revision, str) and re.fullmatch(r"[0-9a-f]{40}", base_revision) is not None
    if not revision_valid:
        codes.append("MUTABLE_BASE_REVISION")

    digests_valid = all(
        isinstance(data.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", data.get(k)) is not None
        for k in ["datasetDigest", "codeDigest", "configDigest"]
    )

    expected = data.get("expectedDigests")
    lineage_pass = revision_valid and digests_valid
    if isinstance(expected, dict):
        for k in ["datasetDigest", "codeDigest", "configDigest"]:
            if k in expected and expected[k] != data.get(k):
                lineage_pass = False
    if not lineage_pass:
        codes.append("LINEAGE_MISMATCH")

    train_ids, eval_ids = data.get("trainRowIds"), data.get("evalRowIds")
    eval_isolated = (
        isinstance(train_ids, list) and isinstance(eval_ids, list)
        and len(train_ids) > 0 and len(eval_ids) > 0
        and all(isinstance(x, str) and x for x in train_ids)
        and all(isinstance(x, str) and x for x in eval_ids)
        and len(set(train_ids)) == len(train_ids)
        and len(set(eval_ids)) == len(eval_ids)
        and set(train_ids).isdisjoint(set(eval_ids))
    )
    if not eval_isolated:
        codes.append("EVAL_LEAKAGE")

    evaluation_deterministic = data.get("dropoutActiveDuringEval") is False
    if not evaluation_deterministic:
        codes.append("EVAL_DROPOUT_ACTIVE")

    mb, ga, replicas, expected_batch = (
        data.get("microBatch"), data.get("gradientAccumulation"),
        data.get("replicas"), data.get("expectedEffectiveBatch")
    )
    batch_valid = all(is_positive_safe_int(x) for x in [mb, ga, replicas, expected_batch])
    if batch_valid:
        batch_valid = mb * ga * replicas == expected_batch
    if not batch_valid:
        codes.append("EFFECTIVE_BATCH_MISMATCH")

    uninterrupted, resumed, tolerance = (
        data.get("uninterruptedWeights"),
        data.get("resumedWeights"),
        data.get("resumeTolerance")
    )
    resume_pass = (
        isinstance(uninterrupted, list) and isinstance(resumed, list)
        and len(uninterrupted) > 0 and len(uninterrupted) == len(resumed)
        and finite_number(tolerance) and tolerance >= 0
        and all(finite_number(x) for x in uninterrupted + resumed)
        and all(abs(a - b) <= tolerance for a, b in zip(uninterrupted, resumed))
    )
    if not resume_pass:
        codes.append("RESUME_DIVERGENCE")

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_names,
        "trainableCount": trainable_count,
        "peftConfigPass": parameter_valid,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sorted_codes(codes)
    }


@app.post("/adapt")
async def adapt(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(data, dict) or data.get("operation") not in ["choose", "repair"]:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    try:
        return choose(data) if data["operation"] == "choose" else repair(data)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
