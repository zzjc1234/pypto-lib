# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise lease file ownership with synthetic metadata and temporary locks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.dspark_bs1 import resources


class TestResourceLeaseOwnership(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.lease = self.root / "lease.json"
        # No HAL, topology, Worker or device API is reached in these unit tests.
        snapshot = dict(
            hostname="unit-test",
            boot_id="unit-test",
            epoch=1.0,
            devices={
                rank: dict(
                    npu_id=rank // 2,
                    chip_id=rank % 2,
                    logical_device_id=rank,
                    smi_physical_id=rank,
                    idle=True,
                    health="OK",
                    free_bytes=64 * resources.GiB,
                )
                for rank in range(8)
            },
        )
        for target, options in [
            ("snapshot", dict(return_value=snapshot)),
            ("topology", dict(return_value="synthetic-symmetric-SIO")),
            ("owned_children", dict(return_value=[])),
        ]:
            item = patch.object(resources, target, **options)
            result = item.start()
            self.addCleanup(item.stop)
            if target == "owned_children":
                self.children = result
        marker = patch.dict(resources.os.environ, DSPARK_EXPERIMENT_TOKEN="unit-test-only")
        marker.start()
        self.addCleanup(marker.stop)

    def claim(self):
        return resources.claim(
            [1] * 8, self.lease, wait_seconds=0, execution_lock=self.root / "execution.lock"
        )

    def test_removes_own_lease_with_integer_snapshot_keys(self):
        with self.claim() as record:
            self.assertIn(0, record["capacity"]["devices"])
            self.assertIn("0", json.loads(self.lease.read_text())["capacity"]["devices"])
        self.assertFalse(self.lease.exists())

    def test_preserves_replaced_foreign_lease(self):
        with self.claim():
            foreign = json.loads(self.lease.read_text())
            foreign["token"] = "different-owner-token"
            self.lease.write_text(json.dumps(foreign))
        self.assertEqual(json.loads(self.lease.read_text())["token"], "different-owner-token")

    def test_defers_release_while_owned_processes_remain(self):
        self.children.return_value = [dict(pid=123, starttime=456)]
        with self.claim():
            pass
        self.assertTrue(self.lease.exists())
        pending = json.loads((self.root / "release-pending.json").read_text())
        self.assertEqual(pending["remaining"], self.children.return_value)
        self.assertFalse(pending["lease_removed"])
