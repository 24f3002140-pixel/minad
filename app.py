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


def utf8_sort(items):
    return sorted(items, key=lambda x: x.encode("utf-8"))


def unique_sorted_codes(items):
    return utf8_sort(list(set(items)))


# ============================================================
# CHOOSE
# ============================================================

def choose(data):

    policy = data.get("policy")
    candidates = data.get("candidates")

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

    candidates_valid = (
        isinstance(candidates, list)
        and len(candidates) == 4
    )

    if candidates_valid:

        for c in candidates:

            if not isinstance(c, dict):
                candidates_valid = False
                continue

            name = c.get("name")

            if name not in INTERVENTIONS:
                candidates_valid = False
                continue

            if name in candidate_map:
                candidates_valid = False
                continue

            candidate_map[name] = c

    if set(candidate_map.keys()) != set(INTERVENTIONS):
        candidates_valid = False

    if not policy_valid or not candidates_valid:
        return result

    for name in INTERVENTIONS:

        c = candidate_map[name]

        fields = [
            "available",
            "quality",
            "freshness",
            "latencyMs",
            "memoryMb",
            "labeledExamples",
            "oneTimeCost",
            "recurringCost",
        ]

        valid = all(field in c for field in fields)

        if valid:
            valid = isinstance(c["available"], bool)

        if valid:
            valid = (
                finite_number(c["quality"])
                and 0 <= c["quality"] <= 1
            )

        if valid:
            valid = isinstance(c["freshness"], bool)

        if valid:
            valid = (
                finite_number(c["latencyMs"])
                and c["latencyMs"] >= 0
            )

        if valid:
            valid = (
                finite_number(c["memoryMb"])
                and c["memoryMb"] >= 0
            )

        if valid:
            valid = safe_int(c["labeledExamples"])

        if valid:
            valid = (
                finite_number(c["oneTimeCost"])
                and c["oneTimeCost"] >= 0
            )

        if valid:
            valid = (
                finite_number(c["recurringCost"])
                and c["recurringCost"] >= 0
            )

        if not valid:
            result["reasonCodes"][name] = ["INVALID_INPUT"]
            continue

        total = (
            c["oneTimeCost"]
            + policy["horizonRequests"] * c["recurringCost"]
        )

        result["totalCosts"][name] = round(total, 12)

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

        if (
            result["totalCosts"][name]
            > policy["maxTotalCost"]
        ):
            reasons.append("COST_LIMIT")

        result["reasonCodes"][name] = unique_sorted_codes(
            reasons
        )

        if not reasons:
            result["eligible"].append(name)

    if result["eligible"]:
        result["selected"] = result["eligible"][0]

    return result


# ============================================================
# REPAIR
# ============================================================

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

        labels = []

        for token in tokens:

            if (
                token["role"] == "assistant"
                and token["padding"] is False
            ):
                labels.append(token["id"])
            else:
                labels.append(-100)

    else:

        labels = (
            [-100] * len(tokens)
            if isinstance(tokens, list)
            else []
        )

        reasons.append("INVALID_TOKEN")

    # --------------------------------------------------------
    # TEMPLATE
    # --------------------------------------------------------

    template_pass = (
        data.get("templateApplications") == 1
    )

    if not template_pass:
        reasons.append("CHAT_TEMPLATE_COUNT")

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    parameters = data.get("parameters")
    allowed_targets = data.get("allowedTargets")

    parameter_valid = True

    if not isinstance(parameters, list):
        parameter_valid = False
        parameters = []

    if (
        not isinstance(allowed_targets, list)
        or len(allowed_targets) == 0
    ):
        parameter_valid = False
        allowed_targets = []

    else:

        if len(allowed_targets) != len(
            set(allowed_targets)
        ):
            parameter_valid = False

        if any(
            not isinstance(x, str)
            for x in allowed_targets
        ):
            parameter_valid = False

    seen_names = set()
    trainable = []

    for parameter in parameters:

        if not isinstance(parameter, dict):
            parameter_valid = False
            continue

        name = parameter.get("name")
        target = parameter.get("target")
        numel = parameter.get("numel")

        if not isinstance(name, str):
            parameter_valid = False
            continue

        if name == "":
            parameter_valid = False
            continue

        if not isinstance(target, str):
            parameter_valid = False
            continue

        if target == "":
            parameter_valid = False
            continue

        if not positive_safe_int(numel):
            parameter_valid = False
            continue

        if name in seen_names:
            parameter_valid = False
            continue

        seen_names.add(name)

        is_lora = (
            name.endswith(".lora_A.weight")
            or name.endswith(".lora_B.weight")
        )

        if (
            target in allowed_targets
            and is_lora
        ):
            trainable.append(parameter)

    # At least one valid LoRA parameter must
    # use an allowed target.
    if len(trainable) == 0:
        parameter_valid = False

    if not parameter_valid:
        reasons.append("INVALID_PARAMETER")

    # UTF-8 byte sorting
    trainable.sort(
        key=lambda p: p["name"].encode("utf-8")
    )

    trainable_params = [
        p["name"]
        for p in trainable
    ]

    # Safe sum
    trainable_count = 0

    for p in trainable:

        n = p["numel"]

        if (
            trainable_count
            > MAX_SAFE - n
        ):
            parameter_valid = False
            trainable_count = 0

            if "INVALID_PARAMETER" not in reasons:
                reasons.append("INVALID_PARAMETER")

            break

        trainable_count += n

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if data.get("inferenceMode") is not False:
        reasons.append("INFERENCE_MODE")

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    artifact_files = data.get("artifactFiles")

    if isinstance(artifact_files, list):

        adapter_files = [
            x for x in artifact_files
            if isinstance(x, str)
        ]

        adapter_files = utf8_sort(
            adapter_files
        )

    else:

        adapter_files = []

    # Exact set, exactly once each.
    adapter_pass = (
        isinstance(artifact_files, list)
        and len(artifact_files) == 2
        and all(
            isinstance(x, str)
            for x in artifact_files
        )
        and len(set(artifact_files)) == 2
        and utf8_sort(
            artifact_files
        ) == ADAPTER_FILES
    )

    if not adapter_pass:
        reasons.append("ADAPTER_FILE_SET")

    # A full-model artifact is specifically
    # a model artifact, rather than an adapter.
    full_model_artifact = False

    if isinstance(artifact_files, list):

        for filename in artifact_files:

            if not isinstance(filename, str):
                continue

            lower = filename.lower()

            if (
                lower.endswith(".bin")
                or lower.endswith(".safetensors")
                and lower not in {
                    "adapter_model.safetensors"
                }
            ):
                if (
                    lower.startswith("pytorch_model")
                    or lower.startswith("model.")
                    or lower == "model.safetensors"
                ):
                    full_model_artifact = True

    if full_model_artifact:
        reasons.append("FULL_MODEL_ARTIFACT")

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

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
        reasons.append("INCOMPLETE_CHECKPOINT")

    # --------------------------------------------------------
    # LINEAGE
    # --------------------------------------------------------

    base_revision = data.get(
        "baseRevision"
    )

    base_valid = (
        isinstance(base_revision, str)
        and re.fullmatch(
            r"[0-9a-f]{40}",
            base_revision,
        ) is not None
    )

    if not base_valid:
        reasons.append(
            "MUTABLE_BASE_REVISION"
        )

    digest_valid = True

    for key in (
        "datasetDigest",
        "codeDigest",
        "configDigest",
    ):

        value = data.get(key)

        if (
            not isinstance(value, str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                value,
            ) is None
        ):
            digest_valid = False

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

    # --------------------------------------------------------
    # EVAL ISOLATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EVAL DROPOUT
    # --------------------------------------------------------

    evaluation_deterministic = (
        data.get(
            "dropoutActiveDuringEval"
        ) is False
    )

    if not evaluation_deterministic:
        reasons.append(
            "EVAL_DROPOUT_ACTIVE"
        )

    # --------------------------------------------------------
    # EFFECTIVE BATCH
    # --------------------------------------------------------

    micro_batch = data.get(
        "microBatch"
    )

    gradient_accumulation = data.get(
        "gradientAccumulation"
    )

    replicas = data.get(
        "replicas"
    )

    expected_batch = data.get(
        "expectedEffectiveBatch"
    )

    batch_valid = all(
        positive_safe_int(x)
        for x in (
            micro_batch,
            gradient_accumulation,
            replicas,
            expected_batch,
        )
    )

    if batch_valid:

        batch_valid = (
            micro_batch
            * gradient_accumulation
            * replicas
            == expected_batch
        )

    if not batch_valid:
        reasons.append(
            "EFFECTIVE_BATCH_MISMATCH"
        )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # REQUIRED EXACT RESPONSE
    # --------------------------------------------------------

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
        "reasonCodes": unique_sorted_codes(
            reasons
        ),
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
        not in (
            "choose",
            "repair",
        )
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
