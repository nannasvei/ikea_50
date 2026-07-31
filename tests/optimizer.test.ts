import assert from "node:assert/strict";
import test from "node:test";
import { optimalGroups, parseAmounts, sum, tightGroups } from "../app/optimizer.ts";

test("parses Polish currency without losing grosze", () => {
  assert.deepEqual(parseAmounts("150,25 zł\n49.99 PLN").values, [15025, 4999]);
});

test("tight strategy keeps the optimal number of packages", () => {
  const values = [9000, 8000, 7000, 6000, 4000, 3000, 2000, 1000];
  const limit = 15000;
  const tight = tightGroups(values, limit);
  const optimal = optimalGroups(values, limit);

  assert.equal(tight.length, optimal.length);
  assert.ok(tight.every(group => sum(group) >= limit));
  assert.deepEqual(
    tight.flat().sort((a, b) => a - b),
    values.slice().sort((a, b) => a - b)
  );
});

test("tight strategy balances surplus instead of bloating one package", () => {
  const groups = tightGroups([12000, 11000, 9000, 8000, 7000, 6000, 5000, 4000], 15000);
  const totals = groups.map(sum);
  assert.ok(Math.max(...totals) - Math.min(...totals) <= 4000);
});
