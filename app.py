import streamlit as st
from itertools import combinations, combinations_with_replacement
from functools import lru_cache
import time

def generate_color_map(nums):
    unique = sorted(set(nums))
    color_map = {}
    for i, value in enumerate(unique):
        hue = (i * 137.508) % 360
        color_map[value] = f"hsl({hue}, 55%, 65%)"
    return color_map

# ==========================================================
#  PARSE INPUT — JEDYNA ZMIANA W CAŁYM PLIKU
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
            result.append(value)   # ← ← TU JEDYNA ZMIANA
        except ValueError:
            pass

    return result

# ==========================================================
#  RESZTA KODU — IDENTYCZNA Z TWOJĄ WERSJĄ
# ==========================================================

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

# ----------------------------------------------------------
# BACKTRACKING
# ----------------------------------------------------------

def backtracking_core(nums, limit, label_prefix="Backtracking"):
    nums_sorted = sorted(nums, reverse=True)
    best = []
    n = len(nums_sorted)

    status_title = st.markdown(f"#### {label_prefix}")
    status_time = st.empty()
    status_steps = st.empty()
    status_best = st.empty()

    start_time = time.time()
    step = 0

    def dfs(remaining, current):
        nonlocal best, step
        step += 1

        if step % 200 == 0:
            elapsed = time.time() - start_time
            status_time.write(f"⏳ Czas działania: **{elapsed:.1f} s**")
            status_steps.write(f"🔢 Kroki DFS: **{step:,}**")

            if best:
                status_best.write(
                    f"⭐ Najlepszy wynik: **{len(best)} pudełek**  \n"
                    f"`{best}`"
                )
            else:
                status_best.write("⏳ Szukam pierwszego pełnego podziału...")

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

    elapsed = time.time() - start_time
    status_time.write(f"✅ Zakończono w **{elapsed:.2f} s**")
    status_steps.write(f"🔢 Łączna liczba kroków: **{step:,}**")
    status_best.write(
        f"🏁 Ostateczny wynik ({label_prefix}): **{len(best)} pudełek**  \n"
        f"`{best}`"
    )

    return best

def alg_backtracking(nums, limit):
    best_groups = backtracking_core(nums, limit, label_prefix="5️⃣ Pełny Backtracking (LIVE)")
    return finalize_groups(nums, best_groups, limit)

# ----------------------------------------------------------
# ALGORYTM #4 – PRE-GROUPS
# ----------------------------------------------------------

def best_subset_closest_at_least(nums, limit):
    dp = {0: []}
    for idx, val in enumerate(nums):
        current = list(dp.items())
        for s, idx_list in current:
            new_s = s + val
            if new_s not in dp:
                dp[new_s] = idx_list + [idx]

    candidates = [s for s in dp.keys() if s >= limit]
    if not candidates:
        return None
    best_sum = min(candidates)
    return dp[best_sum]

def alg_prepack_backtracking(nums, limit, pre_groups_count):
    if pre_groups_count <= 0:
        st.info("Alg.4: pre_groups_count = 0, używam pełnego backtrackingu.")
        return alg_backtracking(nums, limit)

    remaining = nums[:]
    pre_groups = []

    for i in range(pre_groups_count):
        if not remaining:
            break

        idxs = best_subset_closest_at_least(remaining, limit)
        if idxs is None:
            st.warning(
                f"Alg.4: nie udało się stworzyć kolejnej grupy ≥ {limit} "
                f"przy próbie #{i+1}. Przerywam heurystykę."
            )
            break

        group = [remaining[j] for j in idxs]
        pre_groups.append(group)

        idx_set = set(idxs)
        remaining = [v for j, v in enumerate(remaining) if j not in idx_set]

    st.write(
        f"Alg.4: utworzono heurystycznie **{len(pre_groups)}** grup, "
        f"pozostało **{len(remaining)}** elementów dla backtrackingu."
    )

    if remaining:
        rest_groups_core = backtracking_core(
            remaining,
            limit,
            label_prefix="4️⃣ Backtracking na pozostałych (po wstępnych grupach)"
        )
    else:
        rest_groups_core = []

    all_core_groups = pre_groups + rest_groups_core
    return finalize_groups(nums, all_core_groups, limit)

# ----------------------------------------------------------
# DP BITMASK
# ----------------------------------------------------------

def alg_dp_bitmask(nums, limit):
    groups_core = dp_optimal_groups(nums, limit)
    return finalize_groups(nums, groups_core, limit)

# ----------------------------------------------------------
# AUGMENTATION
# ----------------------------------------------------------

def optimal_box_count(nums, limit):
    groups = dp_optimal_groups(nums, limit)
    return len(groups)

def find_min_extra_for_new_box(nums, limit, max_x=100):
    progress = st.progress(0, text="⏳ Szukam minimalnej dopłaty X…")
    status = st.empty()
    start = time.time()

    base = optimal_box_count(nums, limit)
    target = base + 1 if base > 0 or sum(nums) >= limit else 1

    for x in range(1, max_x + 1):
        pct = x / max_x
        progress.progress(pct)

        elapsed = time.time() - start
        eta = elapsed * (1/pct - 1) if pct > 0 else 0

        status.write(
            f"Testuję X = {x} | cel: ≥ {target} pudełek | "
            f"{pct*100:.1f}% | ETA: {eta:.1f} s"
        )

        boxes = optimal_box_count(nums + [x], limit)
        if boxes >= target:
            progress.progress(1.0)
            status.write(f"✅ Znaleziono X = {x}")
            return x, base, boxes

    progress.progress(1.0)
    status.write("❌ Nie znaleziono X w zadanym zakresie")
    return None, base, base

# ----------------------------------------------------------
# HTML RENDERING
# ----------------------------------------------------------

def draw_box(num, colors):
    return (
        f"<span style='background-color:{colors[num]};"
        f"padding:6px 10px;border-radius:8px;"
        f"margin-right:6px;color:black;font-weight:bold;"
        f"display:inline-block;'>{num}</span>"
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

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.title("📦 Grupowanie kwot na pudełka ≥ limit")

col1, col2 = st.columns(2)
with col1:
    raw = st.text_area(
        "Lista kwot:",
        "35,99 zł\n35,00 zł\n35,99 zł\n21,99 zł\n39,99 zł\n44,99 zł\n25,99 zł\n4,00 zł\n3,99 zł\n29,99 zł\n24,99 zł\n12,99 zł",
        key="lista_kwot",
    )
with col2:
    limit = st.number_input(
        "Limit pudełka:", value=50, min_value=1, step=1, key="limit_pudelka"
    )
    pre_groups_count = st.number_input(
        "Alg.4: ile wstępnych grup (x)?",
        value=2, min_value=0, step=1, key="pre_groups_count"
    )

nums = parse_input(raw)

if not nums:
    st.warning("Wpisz poprawne dane wejściowe (co najmniej jedna liczba).")

tab1, tab2 = st.tabs([
    "🧮 Grupowanie pudełek",
    "💸 Minimalna dopłata na nową paczkę"
])

with tab1:
    if st.button("Oblicz grupowanie", key="btn_oblicz_grupowanie"):
        if not nums:
            st.error("Błędne dane wejściowe.")
        else:
            colors = generate_color_map(nums)
            total = sum(nums)
            theoretical = total // limit

            st.markdown(f"### Całkowita suma: **{total}**")
            st.markdown(f"### Teoretyczna liczba pudełek: **{theoretical}**")
            st.markdown("---")

            show_groups("1. Largest + Smallest Fit", alg_largest_smallest(nums, limit), colors)
            show_groups("2. Best-Fit Increasing", alg_best_fit_increasing(nums, limit), colors)
            show_groups("3. Greedy Largest First", alg_greedy_largest(nums, limit), colors)

            show_groups(
                "4. Wstępne grupy (x) + Backtracking na reszcie",
                alg_prepack_backtracking(nums, limit, int(pre_groups_count)),
                colors
            )

            show_groups("5. Pełny Backtracking (LIVE)", alg_backtracking(nums, limit), colors)

            show_groups("6. Bitmask DP (optymalne)", alg_dp_bitmask(nums, limit), colors)

            st.markdown("## 🔍 Analiza optymalnej liczby pudełek")

            opt_boxes = optimal_box_count(nums, limit)
            st.write(f"**Maksymalna liczba pudełek (optymalnie, DP):** {opt_boxes}")

            if opt_boxes >= theoretical:
                st.success("Już osiągasz teoretyczną liczbę pudełek – augmentacja niepotrzebna.")
            else:
                st.warning(
                    f"Aktualnie da się ułożyć maksymalnie {opt_boxes} pudełek, "
                    f"a teoretycznie możliwe byłoby {theoretical}."
                )
                st.info("Przejdź do zakładki **Minimalna dopłata**, aby znaleźć brakującą kwotę.")

with tab2:
    max_x = st.number_input(
        "Maksymalna dopłata X:",
        value=min(limit, 100),
        min_value=1,
        step=1,
        key="max_x_doplata"
    )

    if st.button("Policz minimalną dopłatę X", key="btn_min_doplata"):
        if not nums:
            st.error("Najpierw wprowadź listę kwot.")
        else:
            colors_before = generate_color_map(nums)
            groups_before = alg_dp_bitmask(nums, limit)
            base_boxes = len(groups_before)

            st.markdown("#### 📦 Stan wyjściowy")
            st.write(f"Pudełek przed dopłatą: **{base_boxes}**")
            show_groups("Pudełka przed dopłatą", groups_before, colors_before)

            x, base_boxes_calc, boxes_after = find_min_extra_for_new_box(
                nums, limit, max_x=int(max_x)
            )

            st.markdown("---")
            st.markdown("#### 🔍 Wynik dopłaty")

            if x is None:
                st.error(
                    f"Nie znaleziono wartości X w zakresie 1..{int(max_x)} "
                    f"zwiększającej liczbę pudełek."
                )
            else:
                st.success(
                    f"Minimalne X = **{x} zł**  \n"
                    f"Pudełek przed: **{base_boxes_calc}**  \n"
                    f"Pudełek po: **{boxes_after}**"
                )
                new_nums = nums + [x]
                colors_after = generate_color_map(new_nums)
                groups_after = alg_dp_bitmask(new_nums, limit)
                show_groups("Pudełka po dodaniu X", groups_after, colors_after)
