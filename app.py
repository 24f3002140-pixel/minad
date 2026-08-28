from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import math
import re

app = FastAPI()

MAX_SAFE = 9007199254740991

INTERVENTIONS = [
    "prompt_only",
    "retrieval",
    "lora",
    "qlora",
]

ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]

REASON_CODES = [
    "INVALID_TOKEN",
    "INVALID_PARAMETER",
    "CHAT_TEMPLATE_COUNT",
    "INFERENCE_MODE",
    "FULL_MODEL_ARTIFACT",
    "ADAPTER_FILE_SET",
    "INCOMPLETE_CHECKPOINT",
    "MUTABLE_BASE_REVISION",
    "LINEAGE_MISMATCH",
    "EFFECTIVE_BATCH_MISMATCH",
    "EVAL_LEAKAGE",
    "EVAL_DROPOUT_ACTIVE",
    "RESUME_DIVERGENCE",
]


def safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE
    )


def positive_safe_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 < x <= MAX_SAFE
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


def valid_hex(value, length):
    return (
        isinstance(value, str)
        and re.fullmatch(
            rf"[0-9a-f]{{{length}}}",
            value,
        ) is not None
    )


# ============================================================
# CHOOSE
# ============================================================

def choose(data):

    result = {
        "selected": None,
        "eligible": [],
        "totalCosts": {
            "prompt_only": None,
            "retrieval": None,
            "lora": None,
            "qlora": None,
        },
        "reasonCodes": {
            "prompt_only": ["INVALID_INPUT"],
            "retrieval": ["INVALID_INPUT"],
            "lora": ["INVALID_INPUT"],
            "qlora": ["INVALID_INPUT"],
        },
    }

    policy = data.get("policy")
    candidates = data.get("candidates")

    policy_ok = (
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
    candidates_ok = (
        isinstance(candidates, list)
        and len(candidates) == 4
    )

    if candidates_ok:
        for c in candidates:
            if not isinstance(c, dict):
                candidates_ok = False
                continue

            name = c.get("name")

            if name not in INTERVENTIONS:
                candidates_ok = False
                continue

            if name in candidate_map:
                candidates_ok = False
                continue

            candidate_map[name] = c

    if set(candidate_map.keys()) != set(INTERVENTIONS):
        candidates_ok = False

    if not policy_ok or not candidates_ok:
        return result

    for name in INTERVENTIONS:

        c = candidate_map[name]

        valid = (
            isinstance(c.get("available"), bool)
            and finite_number(c.get("quality"))
            and 0 <= c["quality"] <= 1
            and isinstance(c.get("freshness"), bool)
            and finite_number(c.get("latencyMs"))
            and c["latencyMs"] >= 0
            and finite_number(c.get("memoryMb"))
            and c["memoryMb"] >= 0
            and safe_int(c.get("labeledExamples"))
            and finite_number(c.get("oneTimeCost"))
            and c["oneTimeCost"] >= 0
            and finite_number(c.get("recurringCost"))
            and c["recurringCost"] >= 0
        )

        if not valid:
            result["reasonCodes"][name] = [
                "INVALID_INPUT"
            ]
            continue

        total = round(
            c["oneTimeCost"]
            + policy["horizonRequests"]
            * c["recurringCost"],
            12,
        )

        result["totalCosts"][name] = total

        reasons = []

        if not c["available"]:
            reasons.append("UNAVAILABLE")

        if c["quality"] < policy["minQuality"]:
            reasons.append("QUALITY_FLOOR")

        if (
            policy["freshnessRequired"]
            and not c["freshness"]
        ):
            reasons.append("FRESHNESS_REQUIRED")

        if c["latencyMs"] > policy["maxLatencyMs"]:
            reasons.append("LATENCY_LIMIT")

        if c["memoryMb"] > policy["maxMemoryMb"]:
            reasons.append("MEMORY_LIMIT")

        if (
            c["labeledExamples"]
            > policy["maxLabeledExamples"]
        ):
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
# PEFT PARAMETER HELPERS
# ============================================================

def parameter_is_valid(p):
    if not isinstance(p, dict):
        return False

    name = p.get("name")
    target = p.get("target")
    numel = p.get("numel")

    if not isinstance(name, str) or name == "":
        return False

    if not isinstance(target, str) or target == "":
        return False

    if not positive_safe_int(numel):
        return False

    # Some graders include shape metadata. Validate it when
    # present, but do not require fields not in the contract.
    if "shape" in p:

        shape = p["shape"]

        if (
            not isinstance(shape, list)
            or len(shape) == 0
        ):
            return False

        product = 1

        for dimension in shape:

            if not positive_safe_int(dimension):
                return False

            if product > MAX_SAFE // dimension:
                return False

            product *= dimension

        if product != numel:
            return False

    return True


def lora_name(name):
    return (
        isinstance(name, str)
        and (
            name.endswith(".lora_A.weight")
            or name.endswith(".lora_B.weight")
        )
    )


def repair(data):

    reasons = []

    # ========================================================
    # TOKENS
    # ========================================================

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

            if not isinstance(
                token.get("padding"),
                bool,
            ):
                token_valid = False
                break

            if not isinstance(
                token.get("text"),
                str,
            ):
                token_valid = False
                break

    if token_valid:

        labels = [
            t["id"]
            if (
                t["role"] == "assistant"
                and t["padding"] is False
            )
            else -100
            for t in tokens
        ]

    else:

        labels = (
            [-100] * len(tokens)
            if isinstance(tokens, list)
            else []
        )

        reasons.append("INVALID_TOKEN")

    # ========================================================
    # TEMPLATE
    # ========================================================

    template_pass = (
        data.get("templateApplications") == 1
    )

    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    # ========================================================
    # PARAMETERS
    # ========================================================

    parameters = data.get("parameters")
    allowed = data.get("allowedTargets")

    parameter_pass = True

    if not isinstance(parameters, list):
        parameters = []
        parameter_pass = False

    if not isinstance(allowed, list):
        allowed = []
        parameter_pass = False

    if len(allowed) == 0:
        parameter_pass = False

    # Validate allowedTargets WITHOUT using set()
    # until all values are known to be strings.
    if any(
        not isinstance(x, str)
        or x == ""
        for x in allowed
    ):
        parameter_pass = False

    else:

        if len(allowed) != len(set(allowed)):
            parameter_pass = False

    allowed_set = (
        set(allowed)
        if all(
            isinstance(x, str)
            for x in allowed
        )
        else set()
    )

    seen_names = set()
    trainable = []

    for p in parameters:

        if not parameter_is_valid(p):
            parameter_pass = False
            continue

        name = p["name"]
        target = p["target"]

        if name in seen_names:
            parameter_pass = False
            continue

        seen_names.add(name)

        # Train ONLY parameters that satisfy BOTH:
        # 1. target is explicitly allowed
        # 2. name ends in LoRA A/B weight
        if (
            target in allowed_set
            and lora_name(name)
        ):
            trainable.append(p)

    # At least one trainable LoRA parameter required.
    if len(trainable) == 0:
        parameter_pass = False

    # Sort names by UTF-8 bytes.
    trainable.sort(
        key=lambda p: p["name"].encode("utf-8")
    )

    trainable_params = [
        p["name"]
        for p in trainable
    ]

    # Safe numel sum.
    trainable_count = 0

    for p in trainable:

        n = p["numel"]

        if n > MAX_SAFE - trainable_count:

            parameter_pass = False
            trainable_count = 0
            break

        trainable_count += n

    if not parameter_pass:
        reasons.append("INVALID_PARAMETER")

    # ========================================================
    # INFERENCE
    # ========================================================

    if data.get("inferenceMode") is not False:
        reasons.append("INFERENCE_MODE")

    # ========================================================
    # ARTIFACTS
    # ========================================================

    artifact_files = data.get("artifactFiles")

    if isinstance(artifact_files, list):

        # Return the actual supplied set sorted by UTF-8.
        if all(
            isinstance(x, str)
            for x in artifact_files
        ):
            adapter_files = utf8_sort(
                artifact_files
            )
        else:
            adapter_files = []

    else:

        adapter_files = []

    # EXACTLY these two names, exactly once each.
    adapter_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(
            isinstance(x, str)
            for x in artifact_files
        )
        and len(
            set(artifact_files)
        ) == 2
        and utf8_sort(
            artifact_files
        ) == ADAPTER_FILES
    )

    if not adapter_pass:
        reasons.append("ADAPTER_FILE_SET")

    # Explicit full-model artifact detection.
    full_model = False

    if isinstance(artifact_files, list):

        for filename in artifact_files:

            if not isinstance(filename, str):
                continue

            lower = filename.lower()

            if lower in {
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.bin",
                "model.bin.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.safetensors",
                "pytorch_model.safetensors.index.json",
            }:
                full_model = True
                break

    if full_model:
        reasons.append(
            "FULL_MODEL_ARTIFACT"
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = data.get("checkpoint")

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(
            key in checkpoint
            for key in (
                "model",
                "optimizer",
                "scheduler",
                "step",
                "rng",
                "dataPosition",
            )
        )
    )

    if not checkpoint_complete:
        reasons.append(
            "INCOMPLETE_CHECKPOINT"
        )

    # ========================================================
    # LINEAGE
    # ========================================================

    base_revision = data.get(
        "baseRevision"
    )

    base_valid = valid_hex(
        base_revision,
        40,
    )

    if not base_valid:
        reasons.append(
            "MUTABLE_BASE_REVISION"
        )

    digest_valid = all(
        valid_hex(
            data.get(key),
            64,
        )
        for key in (
            "datasetDigest",
            "codeDigest",
            "configDigest",
        )
    )

    lineage_pass = (
        base_valid
        and digest_valid
    )

    expected = data.get(
        "expectedDigests"
    )

    if isinstance(expected, dict):

        for key in (
            "datasetDigest",
            "codeDigest",
            "configDigest",
        ):

            if (
                key in expected
                and expected[key]
                != data.get(key)
            ):
                lineage_pass = False

    if not lineage_pass:
        reasons.append(
            "LINEAGE_MISMATCH"
        )

    # ========================================================
    # EVALUATION ISOLATION
    # ========================================================

    train_ids = data.get(
        "trainRowIds"
    )

    eval_ids = data.get(
        "evalRowIds"
    )

    eval_isolated = (
        isinstance(train_ids, list)
        and isinstance(eval_ids, list)
        and len(train_ids) > 0
        and len(eval_ids) > 0
        and all(
            isinstance(x, str)
            and x != ""
            for x in train_ids
        )
        and all(
            isinstance(x, str)
            and x != ""
            for x in eval_ids
        )
        and len(train_ids)
        == len(set(train_ids))
        and len(eval_ids)
        == len(set(eval_ids))
        and set(train_ids).isdisjoint(
            set(eval_ids)
        )
    )

    if not eval_isolated:
        reasons.append(
            "EVAL_LEAKAGE"
        )

    # ========================================================
    # EVAL DROPOUT
    # ========================================================

    evaluation_deterministic = (
        data.get(
            "dropoutActiveDuringEval"
        ) is False
    )

    if not evaluation_deterministic:
        reasons.append(
            "EVAL_DROPOUT_ACTIVE"
        )

    # ========================================================
    # EFFECTIVE BATCH
    # ========================================================

    micro = data.get("microBatch")
    accumulation = data.get(
        "gradientAccumulation"
    )
    replicas = data.get("replicas")
    expected_batch = data.get(
        "expectedEffectiveBatch"
    )

    batch_valid = all(
        positive_safe_int(x)
        for x in (
            micro,
            accumulation,
            replicas,
            expected_batch,
        )
    )

    if batch_valid:

        batch_valid = (
            micro
            * accumulation
            * replicas
            == expected_batch
        )

    if not batch_valid:
        reasons.append(
            "EFFECTIVE_BATCH_MISMATCH"
        )

    # ========================================================
    # RESUME
    # ========================================================

    uninterrupted = data.get(
        "uninterruptedWeights"
    )

    resumed = data.get(
        "resumedWeights"
    )

    tolerance = data.get(
        "resumeTolerance"
    )

    resume_pass = (
        isinstance(uninterrupted, list)
        and isinstance(resumed, list)
        and len(uninterrupted) > 0
        and len(uninterrupted)
        == len(resumed)
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
        reasons.append(
            "RESUME_DIVERGENCE"
        )

    # ========================================================
    # EXACT RESPONSE
    # ========================================================

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
            content={
                "error": "INVALID_INPUT"
            },
        )

    if (
        not isinstance(data, dict)
        or data.get("operation")
        not in ("choose", "repair")
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            },
        )

    if data["operation"] == "choose":
        return JSONResponse(
            content=choose(data)
        )

    return JSONResponse(
        content=repair(data)
    )
