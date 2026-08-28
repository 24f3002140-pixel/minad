from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

SAFE_MAX = 9007199254740991
INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]

FULL_MODEL_FILES = {
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model.bin",
    "model.bin.index.json",
}


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= SAFE_MAX
    )


def positive_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 < x <= SAFE_MAX
    )


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
    )


def utf8_sort(values):
    return sorted(values, key=lambda x: x.encode("utf-8"))


def sort_codes(values):
    return utf8_sort(list(set(values)))


# ============================================================
# CHOOSE
# ============================================================

def choose(data):
    names = INTERVENTIONS

    result = {
        "selected": None,
        "eligible": [],
        "totalCosts": {n: None for n in names},
        "reasonCodes": {n: ["INVALID_INPUT"] for n in names},
    }

    policy = data.get("policy")
    candidates = data.get("candidates")

    policy_valid = (
        isinstance(policy, dict)
        and finite_number(policy.get("minQuality"))
        and 0 <= policy["minQuality"] <= 1
        and isinstance(policy.get("freshnessRequired"), bool)
        and finite_number(policy.get("maxLatencyMs"))
        and policy["maxLatencyMs"] >= 0
        and finite_number(policy.get("maxMemoryMb"))
        and policy["maxMemoryMb"] >= 0
        and safe_int(policy.get("maxLabeledExamples"))
        and finite_number(policy.get("maxTotalCost"))
        and policy["maxTotalCost"] >= 0
        and safe_int(policy.get("horizonRequests"))
    )

    candidate_map = {}
    candidates_valid = isinstance(candidates, list) and len(candidates) == 4

    if candidates_valid:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                candidates_valid = False
                continue

            name = candidate.get("name")

            if name not in names:
                candidates_valid = False
                continue

            if name in candidate_map:
                candidates_valid = False
                continue

            candidate_map[name] = candidate

    if set(candidate_map.keys()) != set(names):
        candidates_valid = False

    if not policy_valid or not candidates_valid:
        return result

    for name in names:
        c = candidate_map[name]

        required = [
            "available",
            "quality",
            "freshness",
            "latencyMs",
            "memoryMb",
            "labeledExamples",
            "oneTimeCost",
            "recurringCost",
        ]

        valid = all(k in c for k in required)

        valid = valid and isinstance(c.get("available"), bool)

        valid = (
            valid
            and finite_number(c.get("quality"))
            and 0 <= c["quality"] <= 1
        )

        valid = valid and isinstance(c.get("freshness"), bool)

        valid = (
            valid
            and finite_number(c.get("latencyMs"))
            and c["latencyMs"] >= 0
        )

        valid = (
            valid
            and finite_number(c.get("memoryMb"))
            and c["memoryMb"] >= 0
        )

        valid = valid and safe_int(c.get("labeledExamples"))

        valid = (
            valid
            and finite_number(c.get("oneTimeCost"))
            and c["oneTimeCost"] >= 0
        )

        valid = (
            valid
            and finite_number(c.get("recurringCost"))
            and c["recurringCost"] >= 0
        )

        if not valid:
            result["reasonCodes"][name] = ["INVALID_INPUT"]
            continue

        total = (
            c["oneTimeCost"]
            + policy["horizonRequests"] * c["recurringCost"]
        )

        total = round(total, 12)

        result["totalCosts"][name] = total

        reasons = []

        if not c["available"]:
            reasons.append("UNAVAILABLE")

        if c["quality"] < policy["minQuality"]:
            reasons.append("QUALITY_FLOOR")

        if policy["freshnessRequired"] and not c["freshness"]:
            reasons.append("FRESHNESS_REQUIRED")

        if c["latencyMs"] > policy["maxLatencyMs"]:
            reasons.append("LATENCY_LIMIT")

        if c["memoryMb"] > policy["maxMemoryMb"]:
            reasons.append("MEMORY_LIMIT")

        if c["labeledExamples"] > policy["maxLabeledExamples"]:
            reasons.append("DATA_LIMIT")

        if total > policy["maxTotalCost"]:
            reasons.append("COST_LIMIT")

        result["reasonCodes"][name] = sort_codes(reasons)

        if not reasons:
            result["eligible"].append(name)

    if result["eligible"]:
        result["selected"] = result["eligible"][0]

    return result


# ============================================================
# PEFT PARAMETER VALIDATION
# ============================================================

def valid_parameter_shape(parameter):
    if "shape" not in parameter:
        return True

    shape = parameter["shape"]
    numel = parameter.get("numel")

    if not isinstance(shape, list) or len(shape) == 0:
        return False

    product = 1

    for dimension in shape:
        if not positive_safe_int(dimension):
            return False

        if product > SAFE_MAX // dimension:
            return False

        product *= dimension

    return product == numel


def repair(data):
    reasons = []

    # --------------------------------------------------------
    # TOKENS
    # --------------------------------------------------------

    tokens = data.get("tokens")

    token_valid = (
        isinstance(tokens, list)
        and len(tokens) > 0
    )

    if token_valid:
        for token in tokens:
            if not isinstance(token, dict):
                token_valid = False
                break

            if not safe_int(token.get("id")):
                token_valid = False
                break

            if token.get("role") not in (
                "system",
                "user",
                "assistant",
            ):
                token_valid = False
                break

            if not isinstance(token.get("padding"), bool):
                token_valid = False
                break

            if not isinstance(token.get("text"), str):
                token_valid = False
                break

    if token_valid:
        labels = [
            token["id"]
            if token["role"] == "assistant"
            and token["padding"] is False
            else -100
            for token in tokens
        ]
    else:
        labels = (
            [-100] * len(tokens)
            if isinstance(tokens, list)
            else []
        )
        reasons.append("INVALID_TOKEN")

    # --------------------------------------------------------
    # CHAT TEMPLATE
    # --------------------------------------------------------

    template_pass = data.get("templateApplications") == 1

    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    parameters = data.get("parameters")
    allowed_targets = data.get("allowedTargets")

    parameter_pass = True

    if not isinstance(parameters, list):
        parameters = []
        parameter_pass = False

    if not isinstance(allowed_targets, list):
        allowed_targets = []
        parameter_pass = False

    if len(allowed_targets) == 0:
        parameter_pass = False

    if len(allowed_targets) != len(set(allowed_targets)):
        parameter_pass = False

    if any(
        not isinstance(target, str) or target == ""
        for target in allowed_targets
    ):
        parameter_pass = False

    seen_names = set()
    trainable = []

    for parameter in parameters:
        if not isinstance(parameter, dict):
            parameter_pass = False
            continue

        name = parameter.get("name")
        target = parameter.get("target")
        numel = parameter.get("numel")

        if not isinstance(name, str) or name == "":
            parameter_pass = False
            continue

        if not isinstance(target, str) or target == "":
            parameter_pass = False
            continue

        if not positive_safe_int(numel):
            parameter_pass = False
            continue

        if name in seen_names:
            parameter_pass = False
            continue

        if not valid_parameter_shape(parameter):
            parameter_pass = False

        seen_names.add(name)

        is_lora_parameter = (
            name.endswith(".lora_A.weight")
            or name.endswith(".lora_B.weight")
        )

        if target in allowed_targets and is_lora_parameter:
            trainable.append(parameter)

    if not any(
        isinstance(p, dict)
        and isinstance(p.get("name"), str)
        and isinstance(p.get("target"), str)
        and p.get("target") in allowed_targets
        and (
            p["name"].endswith(".lora_A.weight")
            or p["name"].endswith(".lora_B.weight")
        )
        for p in parameters
    ):
        parameter_pass = False

    if not parameter_pass:
        reasons.append("INVALID_PARAMETER")

    trainable.sort(
        key=lambda p: p["name"].encode("utf-8")
    )

    trainable_params = [
        p["name"]
        for p in trainable
    ]

    trainable_count = 0

    for p in trainable:
        if trainable_count > SAFE_MAX - p["numel"]:
            trainable_count = 0
            parameter_pass = False

            if "INVALID_PARAMETER" not in reasons:
                reasons.append("INVALID_PARAMETER")

            break

        trainable_count += p["numel"]

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if data.get("inferenceMode") is not False:
        reasons.append("INFERENCE_MODE")

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    artifact_files = data.get("artifactFiles")

    if (
        isinstance(artifact_files, list)
        and all(isinstance(x, str) for x in artifact_files)
    ):
        adapter_files = utf8_sort(artifact_files)

        full_model = any(
            x in FULL_MODEL_FILES
            for x in artifact_files
        )
    else:
        adapter_files = []
        full_model = False

    if full_model:
        reasons.append("FULL_MODEL_ARTIFACT")

    adapter_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and len(set(artifact_files)) == 2
        and all(isinstance(x, str) for x in artifact_files)
        and utf8_sort(artifact_files) == ADAPTER_FILES
    )

    if not adapter_pass:
        reasons.append("ADAPTER_FILE_SET")

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = data.get("checkpoint")

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(
            key in checkpoint
            for key in [
                "model",
                "optimizer",
                "scheduler",
                "step",
                "rng",
                "dataPosition",
            ]
        )
    )

    if not checkpoint_complete:
        reasons.append("INCOMPLETE_CHECKPOINT")

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    base_revision = data.get("baseRevision")

    base_revision_valid = (
        isinstance(base_revision, str)
        and re.fullmatch(
            r"[0-9a-f]{40}",
            base_revision,
        ) is not None
    )

    if not base_revision_valid:
        reasons.append("MUTABLE_BASE_REVISION")

    digest_valid = all(
        isinstance(data.get(key), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            data.get(key),
        ) is not None
        for key in [
            "datasetDigest",
            "codeDigest",
            "configDigest",
        ]
    )

    lineage_pass = (
        base_revision_valid
        and digest_valid
    )

    expected = data.get("expectedDigests")

    if isinstance(expected, dict):
        for key in [
            "datasetDigest",
            "codeDigest",
            "configDigest",
        ]:
            if (
                key in expected
                and expected[key] != data.get(key)
            ):
                lineage_pass = False

    if not lineage_pass:
        reasons.append("LINEAGE_MISMATCH")

    # --------------------------------------------------------
    # EVALUATION ISOLATION
    # --------------------------------------------------------

    train_ids = data.get("trainRowIds")
    eval_ids = data.get("evalRowIds")

    eval_isolated = (
        isinstance(train_ids, list)
        and isinstance(eval_ids, list)
        and len(train_ids) > 0
        and len(eval_ids) > 0
        and all(
            isinstance(x, str) and x != ""
            for x in train_ids
        )
        and all(
            isinstance(x, str) and x != ""
            for x in eval_ids
        )
        and len(train_ids) == len(set(train_ids))
        and len(eval_ids) == len(set(eval_ids))
        and set(train_ids).isdisjoint(set(eval_ids))
    )

    if not eval_isolated:
        reasons.append("EVAL_LEAKAGE")

    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    evaluation_deterministic = (
        data.get("dropoutActiveDuringEval") is False
    )

    if not evaluation_deterministic:
        reasons.append("EVAL_DROPOUT_ACTIVE")

    # --------------------------------------------------------
    # EFFECTIVE BATCH
    # --------------------------------------------------------

    micro_batch = data.get("microBatch")
    gradient_accumulation = data.get(
        "gradientAccumulation"
    )
    replicas = data.get("replicas")
    expected_effective_batch = data.get(
        "expectedEffectiveBatch"
    )

    batch_valid = all(
        positive_safe_int(x)
        for x in [
            micro_batch,
            gradient_accumulation,
            replicas,
            expected_effective_batch,
        ]
    )

    if batch_valid:
        batch_valid = (
            micro_batch
            * gradient_accumulation
            * replicas
            == expected_effective_batch
        )

    if not batch_valid:
        reasons.append("EFFECTIVE_BATCH_MISMATCH")

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    uninterrupted = data.get(
        "uninterruptedWeights"
    )
    resumed = data.get("resumedWeights")
    tolerance = data.get("resumeTolerance")

    resume_pass = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted) == len(resumed)
        and finite_number(tolerance)
        and tolerance >= 0
        and all(
            finite_number(x)
            for x in uninterrupted
        )
        and all(
            finite_number(x)
            for x in resumed
        )
        and all(
            abs(a - b) <= tolerance
            for a, b in zip(
                uninterrupted,
                resumed,
            )
        )
    )

    if not resume_pass:
        reasons.append("RESUME_DIVERGENCE")

    # --------------------------------------------------------
    # EXACT RESPONSE
    # --------------------------------------------------------

    return {
        "labels": labels,
        "templatePass": template_pass,
        "trainableParams": trainable_params,
        "trainableCount": trainable_count,
        "peftConfigPass": parameter_pass,
        "adapterFiles": adapter_files,
        "checkpointComplete": checkpoint_complete,
        "lineagePass": lineage_pass,
        "evalIsolated": eval_isolated,
        "evaluationDeterministic": evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes": sort_codes(reasons),
    }


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/adapt")
async def adapt(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if (
        not isinstance(data, dict)
        or data.get("operation")
        not in ("choose", "repair")
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if data["operation"] == "choose":
        return JSONResponse(
            content=choose(data)
        )

    return JSONResponse(
        content=repair(data)
    )
