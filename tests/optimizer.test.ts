import assert from "node:assert/strict";
import test from "node:test";
import { optimalGroups, parseAmounts, sum, tightGroups } from "../app/optimizer.ts";

test("parses Polish currency without losing grosze", () => {
  assert.deepEqual(parseAmounts("150,25 zł\n49.99 PLN").values, [15025, 4999]);
});

test("tight strategy chooses the closest available sum above the limit", () => {
  const groups = tightGroups([12900, 1199, 3799, 1999, 2000, 20000], 15000);
  assert.equal(sum(groups[0]), 16098);
});

test("tight strategy adds the incomplete remainder only to the last package", () => {
  const values = [10000, 5000, 9000, 6000, 1000];
  const groups = tightGroups(values, 15000);
  assert.deepEqual(groups.map(sum), [15000, 16000]);
  assert.deepEqual(groups.flat().sort((a, b) => a - b), values.slice().sort((a, b) => a - b));
});

test("optimal strategy still assigns the remainder when valid packages exist", () => {
  const values = Array.from({ length: 19 }, (_, index) => index === 0 ? 15000 : 100);
  const groups = optimalGroups(values, 15000);
  assert.equal(groups.length, 1);
  assert.equal(sum(groups[0]), sum(values));
});
