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
    """
    Obsługuje dane typu:
    35,99 zł
    44,00
    12 pln
    i zwraca LISTĘ int (zł) – zaokrąglone do najbliższej złotówki.
    """
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
            result.append(int(round(value)))   # pełne złotówki
        except ValueError:
            # ignorujemy linie niebędące liczbami
            pass

    return result


def assign_remainders_to_groups(groups, remainders):
    """Dokładamy resztki do tych pudełek, które mają najmniejszą sumę."""
    if not groups:
        return []
    for r in remainders:
        target = min(groups, key=lambda g: sum(g))
        target.append(r)
    return groups


def finalize_groups(nums, groups, limit):
    """
    nums   – pełna lista wejściowa (int)
    groups – grupy wygenerowane przez algorytm
    limit  – próg (np. 50)

    1. Zostawiamy tylko grupy z sumą >= limit.
    2. Pozostałe liczby (nieużyte + z grup < limit) traktujemy jako resztki.
    3. Doklejamy resztki do pudełek tam, gdzie suma najmniejsza.
    """
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
#  CORE DP – OPTYMALNE GRUPOWANIE (bez UI)
# ==========================================================

def dp_optimal_groups(nums, limit):
    """Zwraca listę optymalnych grup (bez finalize) – używa bitmask DP."""
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


def alg_backtracking(nums, limit):
    nums_sorted = sorted(nums, reverse=True)
    best = []
    n = len(nums_sorted)
    total_est = 2 ** n
    step = 0

    progress = st.progress(0, text="⏳ Backtracking…")
    status = st.empty()
    start_time = time.time()

    def dfs(remaining, current):
        nonlocal best, step
        step += 1

        if step % 500 == 0:
            pct = min(1.0, step / total_est)
            progress.progress(pct)
            elapsed = time.time() - start_time
            eta = elapsed * (1 / pct - 1) if pct > 0 else 0
            status.write(f"⏳ Backtracking… {pct*100:.1f}% | ETA: {eta:.1f} s")

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
    progress.progress(1.0)
    status.write("✅ Backtracking zakończony")

    return finalize_groups(nums, best, limit)


def alg_dp_bitmask(nums, limit):
    groups_core = dp_optimal_groups(nums, limit)
    return finalize_groups(nums, groups_core, limit)


# ==========================================================
#  AUGMENTACJA – SZUKANIE DODATKOWYCH LICZB
# ==========================================================

def optimal_box_count(nums, limit):
    groups = dp_optimal_groups(nums, limit)
    return len(groups)


def find_min_single_addition(nums, limit, target_boxes, max_x=100):
    """
    A: jedna dodatkowa liczba X (1..max_x), minimalna,
    dla której liczba pudełek (optymalnie) >= target_boxes.
    """
    progress = st.progress(0, text="⏳ Szukam minimalnej pojedynczej liczby X…")
    status = st.empty()
    start_time = time.time()

    for x in range(1, max_x + 1):
        pct = x / max_x
        progress.progress(pct)
        elapsed = time.time() - start_time
        eta = elapsed * (1 / pct - 1) if pct > 0 else 0
        status.write(f"Testuję X = {x} | {pct*100:.1f}% | ETA: {eta:.1f} s")

        boxes = optimal_box_count(nums + [x], limit)
        if boxes >= target_boxes:
            progress.progress(1.0)
            status.write(f"✅ Znaleziono X = {x}")
            return x

    progress.progress(1.0)
    status.write("❌ Nie znaleziono X w zadanym zakresie")
    return None


def find_min_augmentation_set(nums, limit, target_boxes, max_k=3, max_val=None):
    """
    B: najmniej możliwych liczb (1..max_k) o możliwie najniższej sumie,
    które pozwalają osiągnąć target_boxes pudełek.
    Wartości z przedziału [1..max_val] (domyślnie = limit).
    """
    if max_val is None:
        max_val = limit

    progress = st.progress(0, text="⏳ Szukam minimalnego zestawu liczb…")
    status = st.empty()
    start_time = time.time()

    for k in range(1, max_k + 1):
        combos = list(combinations_with_replacement(range(1, max_val + 1), k))
        combos.sort(key=sum)  # rosnąco po sumie

        total = len(combos)
        for i, combo in enumerate(combos, start=1):
            if i % 100 == 0 or i == total:
                pct = i / total
                progress.progress(pct)
                elapsed = time.time() - start_time
                eta = elapsed * (1 / pct - 1) if pct > 0 else 0
                status.write(
                    f"k={k}, sprawdzam #{i}/{total} {combo} | {pct*100:.1f}% | ETA: {eta:.1f} s"
                )

            boxes = optimal_box_count(nums + list(combo), limit)
            if boxes >= target_boxes:
                progress.progress(1.0)
                status.write(f"✅ Zestaw znaleziony dla k={k}")
                return list(combo)

    progress.progress(1.0)
    status.write("❌ Nie znaleziono zestawu w zadanym zakresie")
    return None


# ==========================================================
#  RENDERING HTML – numer | kafelki | suma
# ==========================================================

def draw_box(num, colors):
    return (
        f"<span style='"
        f"background-color:{colors[num]};"
        f"padding:6px 10px;"
        f"border-radius:8px;"
        f"margin-right:6px;"
        f"color:black;"
        f"font-weight:bold;"
        f"display:inline-block;'>"
        f"{num}</span>"
    )


def show_groups(title, groups, colors):
    st.markdown(f"### {title}")
    if not groups:
        st.info("Brak pudełek.")
        st.markdown("<hr>", unsafe_allow_html=True)
        return

    for i, g in enumerate(groups, start=1):
        boxes = "".join(draw_box(x, colors) for x in g)
        s = sum(g)
        html = (
            f"<div style='margin:4px 0;'>"
            f"<b>Pudełko #{i}</b>"
            f"&nbsp;|&nbsp;"
            f"{boxes}"
            f"&nbsp;|&nbsp;"
            f"<b>suma = {s}</b>"
            f"</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

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
        key="lista_kwot",
    )
with col2:
    limit = st.number_input("Limit pudełka:", value=50, min_value=1, step=1, key="limit_pudelka")

nums = parse_input(raw)

if st.button("Oblicz", key="btn_oblicz"):
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
        show_groups("4. Full Backtracking (z paskiem postępu)", alg_backtracking(nums, limit), colors)
        show_groups("5. Bitmask DP (optymalne)", alg_dp_bitmask(nums, limit), colors)

        # --- Analiza optymalnej liczby pudełek ---

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

            st.markdown("### A) Jedna minimalna dodatkowa liczba X (1 zł, 2 zł, …)")

            x = find_min_single_addition(nums, limit, theoretical, max_x=100)
            if x is not None:
                new_nums = nums + [x]
                new_colors = generate_color_map(new_nums)
                st.write(f"**Minimalne X:** {x} zł")
                show_groups("Pudełka po dodaniu X", alg_dp_bitmask(new_nums, limit), new_colors)

            st.markdown("### B) Najtańszy zestaw dodatkowych liczb")

            combo = find_min_augmentation_set(nums, limit, theoretical, max_k=3, max_val=limit)
            if combo is not None:
                new_nums2 = nums + combo
                new_colors2 = generate_color_map(new_nums2)
                st.write(f"**Najtańszy zestaw:** {combo} (suma = {sum(combo)} zł)")
                show_groups("Pudełka po dodaniu zestawu", alg_dp_bitmask(new_nums2, limit), new_colors2)
