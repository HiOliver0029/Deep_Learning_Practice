# convert_kaggle_to_mc.py
import json
import sys
from pathlib import Path

def load_json_maybe_plain(path):
    txt = Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(txt)
    except Exception:
        # fallback: split by blank line, return list paragraphs
        paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
        if len(paras) <= 1:
            paras = [p.strip() for p in txt.splitlines() if p.strip()]
        return paras

def build_context_map(context_obj):
    """
    Return dict: id (as str) -> paragraph text
    Accepts:
      - dict (keys may be str or int)
      - list (index -> text) -> keys will be stringified indices
      - list of tuples -> try to convert
    """
    ctx = {}
    if isinstance(context_obj, dict):
        for k, v in context_obj.items():
            ctx[str(k)] = v
    elif isinstance(context_obj, list):
        for i, v in enumerate(context_obj):
            ctx[str(i)] = v
    else:
        raise ValueError("unsupported context format")
    return ctx

def convert_mc(input_json_path, context_path, out_json_path):
    data = load_json_maybe_plain(input_json_path)
    context_obj = load_json_maybe_plain(context_path)
    ctx = build_context_map(context_obj)

    fout = open(out_json_path, 'w', encoding='utf-8')
    # data could be a list or a dict with "data"
    if isinstance(data, dict) and "data" in data:
        examples = data["data"]
    else:
        examples = data

    for ex in examples:
        q = ex.get("question", "")
        paras_ids = ex.get("paragraphs", [])
        # map ids to texts; accept int ids too
        choices = []
        for pid in paras_ids:
            pid_s = str(pid)
            text = ctx.get(pid_s, None)
            if text is None:
                # if missing, put empty string and warn
                print(f"WARNING: paragraph id {pid} not found in context", file=sys.stderr)
                text = ""
            choices.append(text)
        rec = {"question": q, "choices": choices}
        # label: index of relevant paragraph within paragraphs list (if available)
        relevant = ex.get("relevant", None)
        if relevant is not None:
            # relevant is an id, so find its index in paras_ids
            try:
                label = paras_ids.index(relevant)
            except ValueError:
                # maybe types differ (str vs int) — try string compare
                try:
                    label = [str(x) for x in paras_ids].index(str(relevant))
                except ValueError:
                    print(f"WARNING: relevant id {relevant} not in paragraphs for example {ex.get('id')}", file=sys.stderr)
                    label = None
            if label is not None:
                rec["label"] = int(label)
        # keep id for traceability
        if "id" in ex:
            rec["id"] = ex["id"]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.close()
    print("written", out_json_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python convert_kaggle_to_mc.py <kaggle_train_or_dev.json> <context.json> <out.json>")
        sys.exit(1)
    convert_mc(sys.argv[1], sys.argv[2], sys.argv[3])
