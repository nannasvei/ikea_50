import assert from "node:assert/strict";
import test from "node:test";
import { optimalGroups, parseAmounts, remainingValues, sum, tightGroups } from "../app/optimizer.ts";

test("parses Polish currency without losing grosze", () => {
  assert.deepEqual(parseAmounts("150,25 zł\n49.99 PLN").values, [15025, 4999]);
});

test("tight strategy chooses the closest available sum above the limit", () => {
  const groups = tightGroups([12900, 1199, 3799, 1999, 2000], 15000);
  assert.equal(sum(groups[0]), 16098);
});

test("tight strategy does not add an incomplete remainder to a finished package", () => {
  const values = [24500, 1000, 2000];
  const groups = tightGroups(values, 15000);
  assert.deepEqual(groups, [[24500]]);
  assert.deepEqual(remainingValues(values, groups), [1000, 2000]);
});

test("optimal strategy still assigns the remainder when valid packages exist", () => {
  const values = Array.from({ length: 19 }, (_, index) => index === 0 ? 15000 : 100);
  const groups = optimalGroups(values, 15000);
  assert.equal(groups.length, 1);
  assert.equal(sum(groups[0]), sum(values));
});
