import streamlit as st
from itertools import combinations, combinations_with_replacement
from functools import lru_cache
import math
import time

# ==========================================================
#  PASTELOWE KOLORY
# ==========================================================

def generate_color_map(nums):
    unique = sorted(set(nums))
    color_map = {}
    for i, value in enumerate(unique):
        hue = (i * 137.508) % 360
        color_map[value] = f"hsl({hue}, 55%, 65%)"
    return color_map


# ==========================================================
#  POMOCNICZE
# ==========================================================

def parse_input(text):
    result = []
    for line in text.splitlines():
        line = line.strip().lower()
        if not line:
            continue

        clean = (
            line.replace("zł", "")
                .replace("pln", "")
                .replace(" ", "")
                .replace(",", ".")
                .strip()
        )
        try:
            value = float(clean)
            result.append(int(round(value)))
        except ValueError:
            pass
    return result


def assign_remainders_to_groups(groups, remainders):
    if not groups:
        return []
    for r in remainders:
        target = min(groups, key=lambda g: sum(g))
        target.append(r)
    return groups


def finalize_groups(nums, groups, limit):
    nums_all = nums[:]
    valid = [g[:] for g in groups if sum(g) >= limit]

    if not nums_all:
        return []

    if not valid:
        return [nums_all]

    remaining = nums_all[:]
    for g in valid:
        for x in g:
            if x in remaining:
                remaining.remove(x)

    assign_remainders_to_groups(valid, remaining)
    return valid


# ==========================================================
#  ALGORYTMY „NORMALNE”
# ==========================================================

def alg_largest_smallest(nums, limit):
    nums_sorted = sorted(nums, reverse=True)
    unused = nums_sorted[:]
    groups = []

    while unused:
        largest = unused[0]
        others = unused[1:]
        group = [largest]
        added = False

        smalls = sorted(others)
        for s in smalls:
            if largest + s >= limit:
                group.append(s)
                added = True
                break

        if not added:
            for a, b in combinations(smalls, 2):
                if largest + a + b >= limit:
                    group.extend([a, b])
                    added = True
                    break

        if not added:
            break

        for x in group:
            unused.remove(x)

        groups.append(group)

    return finalize_groups(nums, groups, limit)


def alg_best_fit_increasing(nums, limit):
    nums_sorted = sorted(nums)
    groups = []

    for num in nums_sorted:
        best_group = None
        best_over = 10**9

        for g in groups:
            if sum(g) < limit:
                new_sum = sum(g) + num
                if new_sum >= limit:
                    over = new_sum - limit
                    if over < best_over:
                        best_over = over
                        best_group = g

        if best_group is not None:
            best_group.append(num)
        else:
            groups.append([num])

    return finalize_groups(nums, groups, limit)


def alg_greedy_largest(nums, limit):
    nums_sorted = sorted(enumerate(nums), key=lambda x: x[1], reverse=True)
    used = set()
    groups = []

    for idx, val in nums_sorted:
        if idx in used:
            continue

        group = [val]
        total = val
        used.add(idx)

        for j, v in nums_sorted:
            if j in used:
                continue
            if total + v <= limit + 1:
                total += v
                group.append(v)
                used.add(j)

        if total >= limit:
            groups.append(group)

    return finalize_groups(nums, groups, limit)


# ==========================================================
#  CORE DP – OPTYMALNE GRUPOWANIE
# ==========================================================

def dp_optimal_groups(nums, limit):
    n = len(nums)
    if n == 0:
        return []

    def subset_sum(mask):
        return sum(nums[i] for i in range(n) if (mask >> i) & 1)

    subsets = [m for m in range(1, 1 << n) if subset_sum(m) >= limit]

    @lru_cache(None)
    def solve(mask):
        best = 0
        for s in subsets:
            if (mask & s) == s:
                best = max(best, 1 + solve(mask ^ s))
        return best

    groups = []
    mask = (1 << n) - 1
    while mask:
        chosen = None
        for s in subsets:
            if (mask & s) == s and solve(mask) == 1 + solve(mask ^ s):
                chosen = s
                break
        if chosen is None:
            break
        group = [nums[i] for i in range(n) if (chosen >> i) & 1]
        groups.append(group)
        mask ^= chosen

    return groups


# ==========================================================
#  BACKTRACKING — Z *ŻYWYM PODGLĄDEM*
# ==========================================================

def alg_backtracking(nums, limit):
    """
    Backtracking z żywym podglądem:
    - czas działania
    - liczba kroków
    - najlepszy wynik do tej pory
    """

    nums_sorted = sorted(nums, reverse=True)
    best = []
    n = len(nums_sorted)

    # miejsca UI na dynamiczne aktualizacje
    status_time = st.empty()
    status_steps = st.empty()
    status_best = st.empty()

    start_time = time.time()
    step = 0

    def dfs(remaining, current):
        nonlocal best, step

        step += 1

        # aktualizacja co 200 kroków (mniej spowalnia)
        if step % 200 == 0:
            elapsed = time.time() - start_time
            status_time.write(f"⏳ Czas działania: **{elapsed:.1f} s**")
            status_steps.write(f"🔢 Kroki DFS: **{step:,}**")

            if best:
                status_best.write(
                    f"⭐ Najlepszy wynik: **{len(best)} pudełek**\n"
                    f"`{best}`"
                )
            else:
                status_best.write("⏳ Szukam pierwszego lepszego podziału...")

        # normalna logika DFS — nie zmieniamy jej
        if not remaining:
            if len(current) > len(best):
                best = list(current)
            return

        for r in range(1, len(remaining) + 1):
            for combo in combinations(remaining, r):
                if sum(combo) >= limit:
                    new_rem = remaining[:]
                    for c in combo:
                        new_rem.remove(c)
                    dfs(new_rem, current + [list(combo)])

    dfs(nums_sorted, [])

    # zakończenie
    elapsed = time.time() - start_time
    status_time.write(f"✅ Zakończono w **{elapsed:.2f} s**")
    status_steps.write(f"🔢 Łączna liczba kroków: **{step:,}**")
    status_best.write(f"🏁 OSTATECZNY wynik: **{len(best)} pudełek**\n`{best}`")

    return finalize_groups(nums, best, limit)


def alg_dp_bitmask(nums, limit):
    return finalize_groups(nums, dp_optimal_groups(nums, limit), limit)


# ==========================================================
#  AUGMENTACJA
# ==========================================================

def optimal_box_count(nums, limit):
    return len(dp_optimal_groups(nums, limit))


def find_min_extra_for_new_box(nums, limit, max_x=100):
    progress = st.progress(0)
    status = st.empty()
    start = time.time()

    base = optimal_box_count(nums, limit)
    target = base + 1 if base > 0 else 1

    for x in range(1, max_x + 1):
        pct = x / max_x
        progress.progress(pct)

        est = time.time() - start
        eta = est * (1/pct - 1) if pct > 0 else 0

        status.write(f"Testuję X={x}  | {pct*100:.1f}% | ETA {eta:.1f}s")

        boxes = optimal_box_count(nums + [x], limit)
        if boxes >= target:
            progress.progress(1.0)
            status.write(f"Znaleziono X={x}")
            return x, base, boxes

    return None, base, base


# ==========================================================
#  HTML RENDERING
# ==========================================================

def draw_box(num, colors):
    return (
        f"<span style='background-color:{colors[num]};"
        f"padding:6px 10px;border-radius:8px;margin-right:6px;"
        f"color:black;font-weight:bold;display:inline-block;'>"
        f"{num}</span>"
    )


def show_groups(title, groups, colors):
    st.markdown(f"### {title}")
    if not groups:
        st.info("Brak pudełek.")
        st.markdown("<hr>", unsafe_allow_html=True)
        return

    for i, g in enumerate(groups, start=1):
        tiles = "".join(draw_box(x, colors) for x in g)
        st.markdown(
            f"<b>Pudełko #{i}</b> | {tiles} | <b>suma = {sum(g)}</b>",
            unsafe_allow_html=True
        )
    st.markdown("<hr>", unsafe_allow_html=True)


# ==========================================================
#  UI
# ==========================================================

st.title("📦 Grupowanie kwot na pudełka ≥ limit")

col1, col2 = st.columns(2)

with col1:
    raw = st.text_area(
        "Lista kwot:",
        "35,99 zł\n35,00 zł\n35,99 zł\n21,99 zł\n39,99 zł\n44,99 zł\n25,99 zł\n4,00 zł\n3,99 zł\n29,99 zł\n24,99 zł\n12,99 zł",
    )
with col2:
    limit = st.number_input("Limit pudełka:", value=50, min_value=1)

nums = parse_input(raw)

if not nums:
    st.warning("Wprowadź poprawne dane.")

tab1, tab2 = st.tabs(["🧮 Grupowanie", "💸 Minimalna dopłata"])

with tab1:
    if st.button("Oblicz", key="btn_group"):
        colors = generate_color_map(nums)

        show_groups("1. Largest + Smallest", alg_largest_smallest(nums, limit), colors)
        show_groups("2. Best-Fit Increasing", alg_best_fit_increasing(nums, limit), colors)
        show_groups("3. Greedy Largest First", alg_greedy_largest(nums, limit), colors)
        show_groups("4. Backtracking (LIVE!)", alg_backtracking(nums, limit), colors)
        show_groups("5. Bitmask DP (optymalne)", alg_dp_bitmask(nums, limit), colors)

with tab2:
    max_x = st.number_input("Maks. X", min_value=1, value=50)

    if st.button("Policz dopłatę"):
        x, before, after = find_min_extra_for_new_box(nums, limit, max_x)

        st.write(f"Pudełek przed: **{before}**")
        if x is None:
            st.error("Nie znaleziono X")
        else:
            st.success(f"Minimalne X = **{x}** → {after} pudełek")
            new_nums = nums + [x]
            show_groups("Po dopłacie", alg_dp_bitmask(new_nums, limit), generate_color_map(new_nums))
