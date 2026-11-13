import json
from types import SimpleNamespace

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from evaluate import BASE_MODEL_ID, BASE_REVISION, convert_bfcl_schema_to_json_schema, parse_and_score_response

TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)
PARSER = Qwen3CoderToolParser(TOKENIZER)


def score(text, truncated, tools, ground_truth, category):
    request = ChatCompletionRequest.model_validate(
        {
            "model": BASE_MODEL_ID,
            "messages": [{"role": "user", "content": ""}],
            "tools": [{"type": "function", "function": convert_bfcl_schema_to_json_schema(t)} for t in tools],
        }
    )
    response = SimpleNamespace(text=text, finish_reason="length" if truncated else "stop")
    example = {"function": tools, "ground_truth": ground_truth, "category": category}
    try:
        calls, result = parse_and_score_response(example, response, request, PARSER)
    except Exception:
        return None, False
    malformed = result.get("error") == "Пустой, обрезанный или некорректно размеченный ответ."
    return (None if malformed else calls), bool(result["valid"])


def same(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) < 1e-6
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def pair_score(pred, gold):
    (pred_name, pred_args), = pred.items()
    (gold_name, gold_args), = gold.items()
    if pred_name != gold_name:
        return 0.0
    keys = set(pred_args) | set(gold_args)
    if not keys:
        return 1.0
    hits = sum(k in pred_args and k in gold_args and same(pred_args[k], gold_args[k]) for k in keys)
    return 0.5 + 0.5 * hits / len(keys)


def partial_credit(pred_calls, gold_calls):
    if not gold_calls:
        return float(not pred_calls)
    if not pred_calls:
        return 0.0
    remaining = list(gold_calls)
    total = 0.0
    for pred in pred_calls:
        if not remaining:
            break
        best = max(remaining, key=lambda g: pair_score(pred, g))
        total += pair_score(pred, best)
        remaining.remove(best)
    return total / max(len(pred_calls), len(gold_calls))


def rows(completion_ids, tools, ground_truth, category, calls):
    for ids, t, gt, cat, gold in zip(completion_ids, tools, ground_truth, category, calls):
        text = TOKENIZER.decode(ids, skip_special_tokens=False)
        truncated = TOKENIZER.eos_token_id not in ids and TOKENIZER.convert_tokens_to_ids("<|im_end|>") not in ids
        text = text.split("<|im_end|>")[0].split(TOKENIZER.eos_token)[0]
        yield score(text, truncated, json.loads(t), json.loads(gt), cat), json.loads(gold)


def checker_reward(completion_ids, tools, ground_truth, category, calls, **kwargs):
    return [float(valid) for (_, valid), _ in rows(completion_ids, tools, ground_truth, category, calls)]


def partial_reward(completion_ids, tools, ground_truth, category, calls, **kwargs):
    return [
        1.0 if valid else (0.0 if pred is None else 0.5 * partial_credit(pred, gold))
        for (pred, valid), gold in rows(completion_ids, tools, ground_truth, category, calls)
    ]


REWARDS = {"checker": checker_reward, "partial": partial_reward}


def demo():
    tools = [
        {
            "name": "get_weather",
            "description": "Current weather in a city.",
            "parameters": {
                "type": "dict",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "default": "celsius"}},
                "required": ["city"],
            },
        }
    ]
    call = "<tool_call>\n<function={}>\n{}</function>\n</tool_call>"
    param = "<parameter={}>\n{}\n</parameter>\n"
    paris = call.format("get_weather", param.format("city", "Paris"))
    cases = [
        ("correct call", paris, [{"get_weather": {"city": "Paris"}}], 1.0, 1.0),
        ("wrong optional value", call.format("get_weather", param.format("city", "Paris") + param.format("unit", "fahrenheit")), [{"get_weather": {"city": "Paris"}}], 0.0, 0.375),
        ("wrong function name", call.format("get_forecast", param.format("city", "Paris")), [{"get_weather": {"city": "Paris"}}], 0.0, 0.0),
        ("extra parallel call", paris + "\n" + call.format("get_weather", param.format("city", "London")), [{"get_weather": {"city": "Paris"}}], 0.0, 0.25),
        ("missing call", "It is sunny in Paris.", [{"get_weather": {"city": "Paris"}}], 0.0, 0.0),
        ("malformed markup", paris.replace("</tool_call>", ""), [{"get_weather": {"city": "Paris"}}], 0.0, 0.0),
        ("no call when none needed", "I cannot book flights with these tools.", [], 1.0, 1.0),
        ("call when none needed", paris, [], 0.0, 0.0),
    ]
    import prepare_data

    for title, text, gold, expected_checker, expected_partial in cases:
        ids = TOKENIZER(text + "<|im_end|>", add_special_tokens=False)["input_ids"]
        args = (
            [ids],
            [json.dumps(tools)],
            [json.dumps(prepare_data.ground_truth(gold, tools))],
            [prepare_data.category(gold, tools)],
            [json.dumps(gold)],
        )
        checker, partial = checker_reward(*args)[0], partial_reward(*args)[0]
        print(f"| {title} | {checker:.2f} | {partial:.3f} |")
        assert (checker, round(partial, 3)) == (expected_checker, expected_partial), title


if __name__ == "__main__":
    demo()
