export type Group = number[];

export function parseAmounts(text: string): { values: number[]; invalid: number } {
  const values: number[] = [];
  let invalid = 0;

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const normalized = line
      .toLowerCase()
      .replace(/\s/g, "")
      .replace(/zł|pln/g, "")
      .replace(",", ".");
    const value = Number(normalized);
    if (!Number.isFinite(value) || value <= 0) {
      invalid++;
      continue;
    }
    values.push(Math.round(value * 100));
  }

  return { values, invalid };
}

function finalize(values: number[], core: Group[]): Group[] {
  if (!core.length) return values.length ? [values.slice()] : [];
  const remaining = values.slice();
  for (const group of core) {
    for (const value of group) {
      const index = remaining.indexOf(value);
      if (index >= 0) remaining.splice(index, 1);
    }
  }
  for (const value of remaining) {
    core.reduce((best, group) => sum(group) < sum(best) ? group : best).push(value);
  }
  return core;
}

export const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);

export function greedyGroups(values: number[], limit: number): Group[] {
  const remaining = values.slice().sort((a, b) => b - a);
  const groups: Group[] = [];

  while (sum(remaining) >= limit) {
    const group: Group = [];
    let total = 0;
    for (let i = 0; i < remaining.length;) {
      const value = remaining[i];
      if (total < limit) {
        group.push(value);
        total += value;
        remaining.splice(i, 1);
      } else {
        i++;
      }
    }
    if (total < limit) break;
    groups.push(group);
  }
  return finalize(values, groups);
}

function closestSubsetIndexes(values: number[], limit: number): number[] | null {
  const states = new Map<number, number[]>([[0, []]]);
  let bestSum = Number.POSITIVE_INFINITY;
  let best: number[] | null = null;

  values.forEach((value, index) => {
    const snapshot = [...states.entries()];
    for (const [currentSum, indexes] of snapshot) {
      const nextSum = currentSum + value;
      if (nextSum >= bestSum) continue;
      const nextIndexes = [...indexes, index];
      if (nextSum >= limit) {
        bestSum = nextSum;
        best = nextIndexes;
      } else if (!states.has(nextSum)) {
        states.set(nextSum, nextIndexes);
      }
    }
  });
  return best;
}

export function tightGroups(values: number[], limit: number): Group[] {
  let remaining = values.slice();
  const groups: Group[] = [];
  while (sum(remaining) >= limit) {
    const indexes = closestSubsetIndexes(remaining, limit);
    if (!indexes) break;
    const selected = new Set(indexes);
    groups.push(remaining.filter((_, index) => selected.has(index)));
    remaining = remaining.filter((_, index) => !selected.has(index));
  }
  return finalize(values, groups);
}

export function optimalGroups(values: number[], limit: number): Group[] {
  const n = values.length;
  if (!n) return [];
  if (n > 18) return tightGroups(values, limit);

  const maxMask = 1 << n;
  const subsetSums = new Int32Array(maxMask);
  const minimalCandidates: number[] = [];

  for (let mask = 1; mask < maxMask; mask++) {
    const bit = mask & -mask;
    const index = 31 - Math.clz32(bit);
    subsetSums[mask] = subsetSums[mask ^ bit] + values[index];
    if (subsetSums[mask] < limit) continue;
    let minimal = true;
    for (let bits = mask; bits; bits &= bits - 1) {
      const one = bits & -bits;
      if (subsetSums[mask] - values[31 - Math.clz32(one)] >= limit) {
        minimal = false;
        break;
      }
    }
    if (minimal) minimalCandidates.push(mask);
  }

  const byItem = Array.from({ length: n }, () => [] as number[]);
  for (const candidate of minimalCandidates) {
    for (let index = 0; index < n; index++) {
      if (candidate & (1 << index)) byItem[index].push(candidate);
    }
  }

  const memo = new Map<number, { count: number; choice: number }>();
  const solve = (mask: number): { count: number; choice: number } => {
    if (!mask) return { count: 0, choice: 0 };
    const cached = memo.get(mask);
    if (cached) return cached;
    const firstBit = mask & -mask;
    const firstIndex = 31 - Math.clz32(firstBit);
    let best = solve(mask ^ firstBit);
    best = { count: best.count, choice: 0 };
    for (const candidate of byItem[firstIndex]) {
      if ((candidate & mask) !== candidate) continue;
      const next = solve(mask ^ candidate);
      if (next.count + 1 > best.count) best = { count: next.count + 1, choice: candidate };
    }
    memo.set(mask, best);
    return best;
  };

  const core: Group[] = [];
  let mask = maxMask - 1;
  while (mask) {
    const result = solve(mask);
    if (result.choice) {
      const group: Group = [];
      for (let index = 0; index < n; index++) {
        if (result.choice & (1 << index)) group.push(values[index]);
      }
      core.push(group);
      mask ^= result.choice;
    } else {
      mask ^= mask & -mask;
    }
  }
  return finalize(values, core);
}

export function findMinimumExtra(values: number[], limit: number, maxExtra: number) {
  const base = optimalGroups(values, limit).length;
  const target = base + 1;
  if (sum(values) + maxExtra < target * limit) return null;

  let low = 1;
  let high = maxExtra;
  let answer: number | null = null;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    const count = optimalGroups([...values, middle], limit).length;
    if (count >= target) {
      answer = middle;
      high = middle - 1;
    } else {
      low = middle + 1;
    }
  }
  return answer;
}
