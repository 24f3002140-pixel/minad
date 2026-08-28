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

ADAPTER_SET = {
    "adapter_config.json",
    "adapter_model.safetensors",
}

FULL_MODEL_ARTIFACTS = {
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model.safetensors",
    "pytorch_model.safetensors.index.json",
    "model.bin",
    "model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
}


# ============================================================
# BASIC VALIDATORS
# ============================================================

def safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_SAFE
    )


def positive_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_SAFE
    )


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def utf8_sorted(values):
    return sorted(
        values,
        key=lambda x: x.encode("utf-8")
    )


def sorted_unique_codes(codes):
    return utf8_sorted(list(set(codes)))


def valid_hex(value, length):
    return (
        isinstance(value, str)
        and re.fullmatch(
            "[0-9a-f]{" + str(length) + "}",
            value
        ) is not None
    )


# ============================================================
# CHOOSE OPERATION
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

    # --------------------------------------------------------
    # Validate policy
    # --------------------------------------------------------

    policy_valid = (
        isinstance(policy, dict)
        and finite_number(policy.get("minQuality"))
        and 0 <= policy["minQuality"] <= 1
        and isinstance(
            policy.get("freshnessRequired"),
            bool
        )
        and finite_number(
            policy.get("maxLatencyMs")
        )
        and policy["maxLatencyMs"] >= 0
        and finite_number(
            policy.get("maxMemoryMb")
        )
        and policy["maxMemoryMb"] >= 0
        and safe_int(
            policy.get("maxLabeledExamples")
        )
        and finite_number(
            policy.get("maxTotalCost")
        )
        and policy["maxTotalCost"] >= 0
        and safe_int(
            policy.get("horizonRequests")
        )
    )

    # --------------------------------------------------------
    # Validate exactly four candidates
    # --------------------------------------------------------

    candidates_valid = (
        isinstance(candidates, list)
        and len(candidates) == 4
    )

    candidate_map = {}

    if candidates_valid:

        for candidate in candidates:

            if not isinstance(candidate, dict):
                candidates_valid = False
                continue

            name = candidate.get("name")

            if name not in INTERVENTIONS:
                candidates_valid = False
                continue

            if name in candidate_map:
                candidates_valid = False
                continue

            candidate_map[name] = candidate

    if set(candidate_map.keys()) != set(INTERVENTIONS):
        candidates_valid = False

    if not policy_valid or not candidates_valid:
        return result

    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    for name in INTERVENTIONS:

        c = candidate_map[name]

        valid = True

        if not isinstance(
            c.get("available"),
            bool
        ):
            valid = False

        if not (
            finite_number(c.get("quality"))
            and 0 <= c["quality"] <= 1
        ):
            valid = False

        if not isinstance(
            c.get("freshness"),
            bool
        ):
            valid = False

        if not (
            finite_number(c.get("latencyMs"))
            and c["latencyMs"] >= 0
        ):
            valid = False

        if not (
            finite_number(c.get("memoryMb"))
            and c["memoryMb"] >= 0
        ):
            valid = False

        if not safe_int(
            c.get("labeledExamples")
        ):
            valid = False

        if not (
            finite_number(c.get("oneTimeCost"))
            and c["oneTimeCost"] >= 0
        ):
            valid = False

        if not (
            finite_number(c.get("recurringCost"))
            and c["recurringCost"] >= 0
        ):
            valid = False

        if not valid:
            result["reasonCodes"][name] = [
                "INVALID_INPUT"
            ]
            continue

        total = (
            c["oneTimeCost"]
            + policy["horizonRequests"]
            * c["recurringCost"]
        )

        total = round(total, 12)

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

        if (
            c["latencyMs"]
            > policy["maxLatencyMs"]
        ):
            reasons.append("LATENCY_LIMIT")

        if (
            c["memoryMb"]
            > policy["maxMemoryMb"]
        ):
            reasons.append("MEMORY_LIMIT")

        if (
            c["labeledExamples"]
            > policy["maxLabeledExamples"]
        ):
            reasons.append("DATA_LIMIT")

        if total > policy["maxTotalCost"]:
            reasons.append("COST_LIMIT")

        result["reasonCodes"][name] = (
            sorted_unique_codes(reasons)
        )

        if not reasons:
            result["eligible"].append(name)

    # Published priority is already prompt_only -> retrieval
    # -> lora -> qlora.
    if result["eligible"]:
        result["selected"] = result["eligible"][0]

    return result


# ============================================================
# REPAIR OPERATION
# ============================================================

def repair(data):

    reasons = []

    # ========================================================
    # TOKENIZATION / LOSS MASK
    # ========================================================

    tokens = data.get("tokens")

    tokens_valid = (
        isinstance(tokens, list)
        and len(tokens) > 0
    )

    if tokens_valid:

        for token in tokens:

            if not isinstance(token, dict):
                tokens_valid = False
                break

            if not safe_int(
                token.get("id")
            ):
                tokens_valid = False
                break

            if token.get("role") not in (
                "system",
                "user",
                "assistant",
            ):
                tokens_valid = False
                break

            if not isinstance(
                token.get("padding"),
                bool
            ):
                tokens_valid = False
                break

            if not isinstance(
                token.get("text"),
                str
            ):
                tokens_valid = False
                break

    if tokens_valid:

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

        if isinstance(tokens, list):
            labels = [-100] * len(tokens)
        else:
            labels = []

        reasons.append("INVALID_TOKEN")

    # ========================================================
    # CHAT TEMPLATE
    # ========================================================

    template_pass = (
        data.get("templateApplications") == 1
    )

    if not template_pass:
        reasons.append(
            "CHAT_TEMPLATE_COUNT"
        )

    # ========================================================
    # PEFT PARAMETERS
    # ========================================================

    parameters = data.get("parameters")
    allowed_targets = data.get(
        "allowedTargets"
    )

    parameter_pass = True

    if not isinstance(parameters, list):
        parameters = []
        parameter_pass = False

    if not isinstance(
        allowed_targets,
        list
    ):
        allowed_targets = []
        parameter_pass = False

    # allowedTargets must be non-empty,
    # unique strings.
    if len(allowed_targets) == 0:
        parameter_pass = False

    if any(
        not isinstance(x, str)
        for x in allowed_targets
    ):
        parameter_pass = False

    if all(
        isinstance(x, str)
        for x in allowed_targets
    ):
        if len(allowed_targets) != len(
            set(allowed_targets)
        ):
            parameter_pass = False

    allowed_set = {
        x for x in allowed_targets
        if isinstance(x, str)
    }

    seen_names = set()
    trainable = []

    for parameter in parameters:

        if not isinstance(
            parameter,
            dict
        ):
            parameter_pass = False
            continue

        name = parameter.get("name")
        target = parameter.get("target")
        numel = parameter.get("numel")

        # Only the fields specified by the contract
        # are used for validity.
        if not isinstance(name, str):
            parameter_pass = False
            continue

        if name == "":
            parameter_pass = False
            continue

        if not isinstance(target, str):
            parameter_pass = False
            continue

        if target == "":
            parameter_pass = False
            continue

        if not positive_safe_int(numel):
            parameter_pass = False
            continue

        # Names must be unique.
        if name in seen_names:
            parameter_pass = False
            continue

        seen_names.add(name)

        # Train ONLY:
        #   allowed target
        #   AND LoRA A/B parameter suffix.
        if (
            target in allowed_set
            and (
                name.endswith(
                    ".lora_A.weight"
                )
                or name.endswith(
                    ".lora_B.weight"
                )
            )
        ):
            trainable.append({
                "name": name,
                "numel": numel,
            })

    # At least one eligible LoRA parameter.
    if len(trainable) == 0:
        parameter_pass = False

    # UTF-8 byte ordering.
    trainable.sort(
        key=lambda p:
        p["name"].encode("utf-8")
    )

    trainable_params = [
        p["name"]
        for p in trainable
    ]

    # Safe integer accumulation.
    trainable_count = 0

    for parameter in trainable:

        n = parameter["numel"]

        if n > MAX_SAFE - trainable_count:

            parameter_pass = False
            trainable_count = 0
            break

        trainable_count += n

    if not parameter_pass:
        reasons.append(
            "INVALID_PARAMETER"
        )

    # ========================================================
    # INFERENCE MODE
    # ========================================================

    if data.get(
        "inferenceMode"
    ) is not False:

        reasons.append(
            "INFERENCE_MODE"
        )

    # ========================================================
    # ADAPTER ARTIFACTS
    # ========================================================

    artifact_files = data.get(
        "artifactFiles"
    )

    if (
        isinstance(artifact_files, list)
        and all(
            isinstance(x, str)
            for x in artifact_files
        )
    ):

        adapter_files = utf8_sorted(
            artifact_files
        )

    else:

        adapter_files = []

    # Exact set:
    # adapter_config.json
    # adapter_model.safetensors
    #
    # Exactly once each.
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
        and set(artifact_files)
        == ADAPTER_SET
    )

    if not adapter_pass:
        reasons.append(
            "ADAPTER_FILE_SET"
        )

    # Full-model artifacts.
    full_model = False

    if isinstance(
        artifact_files,
        list
    ):

        for filename in artifact_files:

            if (
                isinstance(filename, str)
                and filename
                in FULL_MODEL_ARTIFACTS
            ):
                full_model = True
                break

    if full_model:
        reasons.append(
            "FULL_MODEL_ARTIFACT"
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = data.get(
        "checkpoint"
    )

    required_checkpoint_fields = (
        "model",
        "optimizer",
        "scheduler",
        "step",
        "rng",
        "dataPosition",
    )

    checkpoint_complete = (
        isinstance(checkpoint, dict)
        and all(
            key in checkpoint
            for key in required_checkpoint_fields
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
        40
    )

    if not base_valid:
        reasons.append(
            "MUTABLE_BASE_REVISION"
        )

    digest_keys = (
        "datasetDigest",
        "codeDigest",
        "configDigest",
    )

    digest_valid = all(
        valid_hex(
            data.get(key),
            64
        )
        for key in digest_keys
    )

    lineage_pass = (
        base_valid
        and digest_valid
    )

    expected = data.get(
        "expectedDigests"
    )

    if isinstance(expected, dict):

        for key in digest_keys:

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
    # EVALUATION DROPOUT
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

    micro_batch = data.get(
        "microBatch"
    )

    gradient_accumulation = data.get(
        "gradientAccumulation"
    )

    replicas = data.get(
        "replicas"
    )

    expected_effective_batch = data.get(
        "expectedEffectiveBatch"
    )

    batch_valid = all(
        positive_safe_int(x)
        for x in (
            micro_batch,
            gradient_accumulation,
            replicas,
            expected_effective_batch,
        )
    )

    if batch_valid:

        batch_valid = (
            micro_batch
            * gradient_accumulation
            * replicas
            == expected_effective_batch
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
                resumed
            )
        )
    )

    if not resume_pass:
        reasons.append(
            "RESUME_DIVERGENCE"
        )

    # ========================================================
    # EXACT REQUIRED RESPONSE
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
        "evaluationDeterministic":
            evaluation_deterministic,
        "resumePass": resume_pass,
        "reasonCodes":
            sorted_unique_codes(reasons),
    }


# ============================================================
# POST /adapt
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
            }
        )

    if (
        not isinstance(data, dict)
        or data.get("operation")
        not in (
            "choose",
            "repair"
        )
    ):

        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_INPUT"
            }
        )

    if data["operation"] == "choose":
        return JSONResponse(
            content=choose(data)
        )

    return JSONResponse(
        content=repair(data)
    )
