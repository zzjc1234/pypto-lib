# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the tracked W2 implementation before publishing generated artifacts.

No old binary is loaded. The original complete generated implementation and ABI
must match before the tracked C++ is selected and compiled by the normal native
builder. Changing the generated implementation requires an explicit review.
"""

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import re

NATIVE_SOURCE = Path(__file__).resolve().parent / "native/experts_w2_prefetch.cpp"
GENERATED_DIGEST = "296fcbd93e7a343de3d8b11651776316d7be0299edfbdb19fa133e7722970c2f"
KERNEL_PATTERN = re.compile(
    r"\b(?:l2_s6_cycles_only_|l2_full_cycles_)?exp_routed_balanced"
    r"(?:_[0-9]+)?(?P<core>_aic|_aiv)?\b"
)


def fingerprint():
    """Include this digest in the JIT entry's constexpr cache-key arguments."""
    return hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest()


def normalized_cpp(text):
    """Retain operators, types, addresses and constants; normalize only names."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = KERNEL_PATTERN.sub(lambda m: "dspark_experts" + (m["core"] or ""), text)
    names = {}

    def rename(match):
        name = match[0]
        if re.search(r"(__ssa_v\d|_inline\d|__rv_v\d|__idx_v\d|_tensor$|_conv$)", name):
            return names.setdefault(name, "local_" + str(len(names)))
        return name

    return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", rename, text))


def select_sources(files, expected_fingerprint):
    """Return a new generated-file map, preserving the original source for audit."""
    source = NATIVE_SOURCE.read_text()
    actual_fingerprint = hashlib.sha256(source.encode()).hexdigest()
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("Native W2 source differs from the requested build fingerprint")
    result = dict(files)
    selected = []
    for name, text in files.items():
        path = Path(name)
        if path.suffix != ".cpp" or path.parent.name != "aic":
            continue
        if not KERNEL_PATTERN.fullmatch(path.stem) or not path.stem.endswith("_aic"):
            continue
        digest = hashlib.sha256(normalized_cpp(text).encode()).hexdigest()
        if digest != GENERATED_DIGEST:
            raise ValueError(f"Unrecognized complete generated W2 implementation: {name}: {digest}")
        result[str(Path("report/original-native-source") / path)] = text
        result[name] = source
        selected.append(dict(source=name, original_normalized_sha256=digest))
    if len(selected) != 10:
        raise ValueError(f"Expected five W2 sites in each of two roles, found {len(selected)}")
    result["report/native-w2-source-selection.json"] = json.dumps(
        dict(implementation="tracked_w2_prefetch", source_sha256=actual_fingerprint,
             selected=selected, old_binaries_loaded=False, native_build="REQUIRED"),
        indent=2,
    ) + "\n"
    return result


@contextmanager
def use_native_w2(expected_fingerprint):
    """Scope source selection to one JIT generation; restore the backend on exit."""
    module = importlib.import_module("pypto.ir.compile")
    original = module.generate
    calls = []

    def generate(*args, **kwargs):
        files = original(*args, **kwargs)
        result = select_sources(files, expected_fingerprint)
        calls.append(True)
        return result

    module.generate = generate
    try:
        yield calls
    finally:
        module.generate = original
