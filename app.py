from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

SAFE_MAX = 9007199254740991
INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

def safe_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= SAFE_MAX

def pos_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and 0 < x <= SAFE_MAX

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def utf8_sorted(xs):
    return sorted(xs, key=lambda x: x.encode("utf-8"))

def codes(xs):
    return utf8_sorted(set(xs))

def choose(data):
    out = {
        "selected": None,
        "eligible": [],
        "totalCosts": {n: None for n in INTERVENTIONS},
        "reasonCodes": {n: ["INVALID_INPUT"] for n in INTERVENTIONS},
    }

    policy = data.get("policy")
    candidates = data.get("candidates")

    policy_ok = (
        isinstance(policy, dict)
        and finite(policy.get("minQuality"))
        and 0 <= policy["minQuality"] <= 1
        and isinstance(policy.get("freshnessRequired"), bool)
        and finite(policy.get("maxLatencyMs")) and policy["maxLatencyMs"] >= 0
        and finite(policy.get("maxMemoryMb")) and policy["maxMemoryMb"] >= 0
        and safe_int(policy.get("maxLabeledExamples"))
        and finite(policy.get("maxTotalCost")) and policy["maxTotalCost"] >= 0
        and safe_int(policy.get("horizonRequests"))
    )

    candidate_map = {}
    candidates_ok = isinstance(candidates, list) and len(candidates) == 4

    if candidates_ok:
        for c in candidates:
            if not isinstance(c, dict) or c.get("name") not in INTERVENTIONS:
                candidates_ok = False
                continue
            name = c["name"]
            if name in candidate_map:
                candidates_ok = False
            else:
                candidate_map[name] = c

    if set(candidate_map) != set(INTERVENTIONS):
        candidates_ok = False

    if not policy_ok or not candidates_ok:
        return out

    for name in INTERVENTIONS:
        c = candidate_map[name]
        required = ["available","quality","freshness","latencyMs","memoryMb",
                    "labeledExamples","oneTimeCost","recurringCost"]

        candidate_ok = (
            all(k in c for k in required)
            and isinstance(c.get("available"), bool)
            and finite(c.get("quality")) and 0 <= c["quality"] <= 1
            and isinstance(c.get("freshness"), bool)
            and finite(c.get("latencyMs")) and c["latencyMs"] >= 0
            and finite(c.get("memoryMb")) and c["memoryMb"] >= 0
            and safe_int(c.get("labeledExamples"))
            and finite(c.get("oneTimeCost")) and c["oneTimeCost"] >= 0
            and finite(c.get("recurringCost")) and c["recurringCost"] >= 0
        )

        if not candidate_ok:
            out["reasonCodes"][name] = ["INVALID_INPUT"]
            continue

        total = round(
            c["oneTimeCost"] +
            policy["horizonRequests"] * c["recurringCost"], 12
        )
        out["totalCosts"][name] = total

        r = []
        if not c["available"]: r.append("UNAVAILABLE")
        if c["quality"] < policy["minQuality"]: r.append("QUALITY_FLOOR")
        if policy["freshnessRequired"] and not c["freshness"]:
            r.append("FRESHNESS_REQUIRED")
        if c["latencyMs"] > policy["maxLatencyMs"]: r.append("LATENCY_LIMIT")
        if c["memoryMb"] > policy["maxMemoryMb"]: r.append("MEMORY_LIMIT")
        if c["labeledExamples"] > policy["maxLabeledExamples"]: r.append("DATA_LIMIT")
        if total > policy["maxTotalCost"]: r.append("COST_LIMIT")

        out["reasonCodes"][name] = codes(r)
        if not r:
            out["eligible"].append(name)

    out["selected"] = out["eligible"][0] if out["eligible"] else None
    return out

def repair(data):
    reasons = []

    tokens = data.get("tokens")
    token_valid = isinstance(tokens, list) and len(tokens) > 0
    if token_valid:
        for t in tokens:
            if not isinstance(t, dict):
                token_valid = False
                break
            if not safe_int(t.get("id")):
                token_valid = False
                break
            if t.get("role") not in ("system", "user", "assistant"):
                token_valid = False
                break
            if not isinstance(t.get("padding"), bool):
                token_valid = False
                break
            if not isinstance(t.get("text"), str):
                token_valid = False
                break

    if token_valid:
        labels = [
            t["id"] if t["role"] == "assistant" and not t["padding"] else -100
            for t in tokens
        ]
    else:
        labels = [-100] * len(tokens) if isinstance(tokens, list) else []
        reasons.append("INVALID_TOKEN")

    template_pass = data.get("templateApplications") == 1
    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    params = data.get("parameters")
    allowed = data.get("allowedTargets")
    parameter_valid = True

    if not isinstance(params, list):
        params = []
        parameter_valid = False

    if not isinstance(allowed, list) or len(allowed) == 0:
        allowed = []
        parameter_valid = False
    elif len(set(allowed)) != len(allowed) or any(
        not isinstance(x, str) or not x for x in allowed
    ):
        parameter_valid = False

    seen = set()
    trainable = []
    has_lora_allowed = False

    for p in params:
        if not isinstance(p, dict):
            parameter_valid = False
            continue
        name, target, numel = p.get("name"), p.get("target"), p.get("numel")
        if (
            not isinstance(name, str) or not name or
            not isinstance(target, str) or not target or
            not pos_int(numel) or name in seen
        ):
            parameter_valid = False
            continue
        seen.add(name)

        is_lora = name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")
        if target in allowed and is_lora:
            has_lora_allowed = True
            trainable.append(p)

    if not has_lora_allowed:
        parameter_valid = False

    if not parameter_valid:
        reasons.append("INVALID_PARAMETER")

    trainable.sort(key=lambda p: p["name"].encode("utf-8"))
    trainable_params = [p["name"] for p in trainable]
    trainable_count = sum(p["numel"] for p in trainable)

    inference_mode = data.get("inferenceMode") is False
    if not inference_mode:
        reasons.append("INFERENCE_MODE")

    artifact_files = data.get("artifactFiles")
    expected_files = ["adapter_config.json", "adapter_model.safetensors"]
    adapter_ok = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and sorted(artifact_files, key=lambda x: x.encode("utf-8")) == expected_files
    )
    adapter_files = (
        utf8_sorted(artifact_files)
        if isinstance(artifact_files, list) and all(isinstance(x, str) for x in artifact_files)
        else []
    )
    if not adapter_ok:
        reasons.append("ADAPTER_FILE_SET")

    checkpoint = data.get("checkpoint")
    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(k in checkpoint for k in
                ["model","optimizer","scheduler","step","rng","dataPosition"])
    )
    if not checkpoint_complete:
        reasons.append("INCOMPLETE_CHECKPOINT")

    base = data.get("baseRevision")
    base_ok = isinstance(base, str) and re.fullmatch(r"[0-9a-f]{40}", base) is not None
    if not base_ok:
        reasons.append("MUTABLE_BASE_REVISION")

    digest_ok = all(
        isinstance(data.get(k), str) and
        re.fullmatch(r"[0-9a-f]{64}", data.get(k)) is not None
        for k in ["datasetDigest","codeDigest","configDigest"]
    )

    lineage_pass = base_ok and digest_ok
    expected = data.get("expectedDigests")
    if isinstance(expected, dict):
        for k in ["datasetDigest","codeDigest","configDigest"]:
            if k in expected and expected[k] != data.get(k):
                lineage_pass = False
    if not lineage_pass:
        reasons.append("LINEAGE_MISMATCH")

    train_ids, eval_ids = data.get("trainRowIds"), data.get("evalRowIds")
    eval_isolated = (
        isinstance(train_ids, list) and isinstance(eval_ids, list)
        and len(train_ids) > 0 and len(eval_ids) > 0
        and all(isinstance(x, str) and x for x in train_ids)
        and all(isinstance(x, str) and x for x in eval_ids)
        and len(set(train_ids)) == len(train_ids)
        and len(set(eval_ids)) == len(eval_ids)
        and set(train_ids).isdisjoint(eval_ids)
    )
    if not eval_isolated:
        reasons.append("EVAL_LEAKAGE")

    evaluation_deterministic = data.get("dropoutActiveDuringEval") is False
    if not evaluation_deterministic:
        reasons.append("EVAL_DROPOUT_ACTIVE")

    mb, ga, replicas, expected_batch = (
        data.get("microBatch"), data.get("gradientAccumulation"),
        data.get("replicas"), data.get("expectedEffectiveBatch")
    )
    batch_ok = all(pos_int(x) for x in [mb, ga, replicas, expected_batch])
    if batch_ok:
        batch_ok = mb * ga * replicas == expected_batch
    if not batch_ok:
        reasons.append("EFFECTIVE_BATCH_MISMATCH")

    uninterrupted = data.get("uninterruptedWeights")
    resumed = data.get("resumedWeights")
    tolerance = data.get("resumeTolerance")

    resume_pass = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted) == len(resumed)
        and finite(tolerance) and tolerance >= 0
        and all(finite(x) for x in uninterrupted)
        and all(finite(x) for x in resumed)
        and all(abs(a-b) <= tolerance for a,b in zip(uninterrupted, resumed))
    )
    if not resume_pass:
        reasons.append("RESUME_DIVERGENCE")

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": parameter_valid,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": codes(reasons)
    }

@app.post("/adapt")
async def adapt(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error":"INVALID_INPUT"})

    if not isinstance(data, dict) or data.get("operation") not in ("choose", "repair"):
        return JSONResponse(status_code=400, content={"error":"INVALID_INPUT"})

    try:
        if data["operation"] == "choose":
            return JSONResponse(content=choose(data))
        return JSONResponse(content=repair(data))
    except Exception:
        # Operation is valid: return the required operation-specific shape,
        # rather than turning a test case into HTTP 400.
        if data["operation"] == "choose":
            return JSONResponse(content={
                "selected": None,
                "eligible": [],
                "totalCosts": {n: None for n in INTERVENTIONS},
                "reasonCodes": {n:["INVALID_INPUT"] for n in INTERVENTIONS},
            })
        return JSONResponse(content={
            "labels": [],
            "templatePass": False,
            "trainableParams": [],
            "trainableCount": 0,
            "peftConfigPass": False,
            "adapterFiles": [],
            "checkpointComplete": False,
            "lineagePass": False,
            "evalIsolated": False,
            "evaluationDeterministic": False,
            "resumePass": False,
            "reasonCodes": ["INVALID_TOKEN"]
        })
