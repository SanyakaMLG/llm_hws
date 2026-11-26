import sys

import nbformat

SETUP = [4, 5, 7, 10, 11, 14, 15, 16, 18, 22, 27, 31, 34, 35]  # ячейки только с определениями
BASELINE, TEACHER_LOAD, TEACHER_EVAL, TEACHER_ANSWERS = 20, 28, 29, 32

nb = nbformat.read("homework_compression_student.ipynb", as_version=4)
ns = {}


def run(index):
    exec(compile(nb.cells[index].source, f"cell{index}", "exec"), ns)


for index in SETUP:
    run(index)
kind, _, name = sys.argv[1].partition(":")
if kind == "baseline":
    run(BASELINE)
elif kind == "teacher":
    for index in (BASELINE, TEACHER_LOAD, TEACHER_EVAL, TEACHER_ANSWERS):
        run(index)
elif kind == "quant":
    ns["run_quantization"](next(c for c in ns["QUANT_CONFIGS"] if c.name == name))
elif kind == "distill":
    cfg = next(c for c in ns["DISTILL_CONFIGS"] if c.name == name)
    ns["evaluate_cached"](f"student_{name}", ns["train_student"](cfg), ns["dev_data"])
else:
    raise SystemExit(f"unknown task {sys.argv[1]}")
