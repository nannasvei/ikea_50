"use client";

import { useMemo, useState } from "react";
import { findMinimumExtra, greedyGroups, optimalGroups, parseAmounts, sum, tightGroups, type Group } from "./optimizer";

const SAMPLE = `35,99 zł
35,00 zł
35,99 zł
21,99 zł
39,99 zł
44,99 zł
25,99 zł
4,00 zł
3,99 zł
29,99 zł
24,99 zł
12,99 zł`;

const money = new Intl.NumberFormat("pl-PL", { style: "currency", currency: "PLN" });
const formatMoney = (cents: number) => money.format(cents / 100);

function PackageCard({ group, index, limit }: { group: Group; index: number; limit: number }) {
  const total = sum(group);
  return (
    <article className="package-card">
      <div className="package-head">
        <span className="package-number">Paczka {String(index + 1).padStart(2, "0")}</span>
        <strong>{formatMoney(total)}</strong>
      </div>
      <div className="amount-pills">
        {group.map((value, itemIndex) => (
          <span className="amount-pill" key={`${value}-${itemIndex}`}>{formatMoney(value)}</span>
        ))}
      </div>
      <div className="meter"><span style={{ width: `${Math.min(100, (total / limit) * 100)}%` }} /></div>
      <small>{total >= limit ? `${formatMoney(total - limit)} ponad limit` : `${formatMoney(limit - total)} do limitu`}</small>
    </article>
  );
}

function Results({ groups, limit }: { groups: Group[]; limit: number }) {
  return (
    <div className="package-grid">
      {groups.map((group, index) => <PackageCard key={index} group={group} index={index} limit={limit} />)}
    </div>
  );
}

export default function Home() {
  const [raw, setRaw] = useState(SAMPLE);
  const [limitText, setLimitText] = useState("50,00");
  const [activeTab, setActiveTab] = useState<"groups" | "extra">("groups");
  const [strategy, setStrategy] = useState<"optimal" | "tight" | "fast">("optimal");
  const [calculated, setCalculated] = useState(true);
  const [maxExtraText, setMaxExtraText] = useState("100,00");

  const parsed = useMemo(() => parseAmounts(raw), [raw]);
  const limit = Math.max(1, Math.round(Number(limitText.replace(",", ".")) * 100) || 5000);
  const total = sum(parsed.values);
  const theoretical = Math.floor(total / limit);
  const groups = useMemo(() => {
    if (!calculated || !parsed.values.length) return [];
    if (strategy === "fast") return greedyGroups(parsed.values, limit);
    if (strategy === "tight") return tightGroups(parsed.values, limit);
    return optimalGroups(parsed.values, limit);
  }, [calculated, parsed.values, limit, strategy]);
  const maxExtra = Math.max(1, Math.round(Number(maxExtraText.replace(",", ".")) * 100) || 10000);
  const extra = useMemo(
    () => activeTab === "extra" && calculated && parsed.values.length
      ? findMinimumExtra(parsed.values, limit, maxExtra)
      : null,
    [activeTab, calculated, parsed.values, limit, maxExtra]
  );
  const groupsAfter = extra ? optimalGroups([...parsed.values, extra], limit) : [];

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#" aria-label="Paczki 50 — strona główna">
          <span className="brand-mark">50</span>
          <span>Paczki</span>
        </a>
        <div className="header-status"><span /> Obliczenia lokalne</div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">GRUPOWANIE KWOT</p>
          <h1>Więcej pełnych paczek.<br /><em>Mniej zgadywania.</em></h1>
          <p className="hero-copy">Wklej kwoty, ustaw próg i od razu zobacz najlepszy podział. Wszystko liczy się w Twojej przeglądarce.</p>
        </div>
        <div className="hero-badge">
          <span>Aktualny potencjał</span>
          <strong>{theoretical}</strong>
          <small>paczek z całej sumy</small>
        </div>
      </section>

      <nav className="tabs" aria-label="Widok kalkulatora">
        <button className={activeTab === "groups" ? "active" : ""} onClick={() => setActiveTab("groups")}>Podział na paczki</button>
        <button className={activeTab === "extra" ? "active" : ""} onClick={() => setActiveTab("extra")}>Minimalna dopłata</button>
      </nav>

      <section className="workspace">
        <aside className="control-panel">
          <div className="section-label"><span>01</span> Dane wejściowe</div>
          <label htmlFor="amounts">Kwoty <small>jedna w wierszu</small></label>
          <textarea id="amounts" value={raw} onChange={(event) => { setRaw(event.target.value); setCalculated(false); }} spellCheck={false} />
          <div className="input-meta">
            <span>{parsed.values.length} pozycji</span>
            <span>{formatMoney(total)} łącznie</span>
          </div>
          {parsed.invalid > 0 && <p className="warning">Pominięto niepoprawne wiersze: {parsed.invalid}</p>}

          <div className="field-row">
            <label>Limit paczki
              <div className="money-input"><input value={limitText} onChange={e => { setLimitText(e.target.value); setCalculated(false); }} inputMode="decimal" /><span>PLN</span></div>
            </label>
            {activeTab === "extra" && (
              <label>Maks. dopłata
                <div className="money-input"><input value={maxExtraText} onChange={e => { setMaxExtraText(e.target.value); setCalculated(false); }} inputMode="decimal" /><span>PLN</span></div>
              </label>
            )}
          </div>

          {activeTab === "groups" && (
            <>
              <div className="section-label second"><span>02</span> Strategia</div>
              <div className="strategies">
                <button className={strategy === "optimal" ? "selected" : ""} onClick={() => setStrategy("optimal")}><strong>Najlepszy wynik</strong><small>Dokładnie do 18 pozycji</small></button>
                <button className={strategy === "tight" ? "selected" : ""} onClick={() => setStrategy("tight")}><strong>Ciasne paczki</strong><small>Najmniejsze nadwyżki</small></button>
                <button className={strategy === "fast" ? "selected" : ""} onClick={() => setStrategy("fast")}><strong>Szybki podział</strong><small>Dla długich list</small></button>
              </div>
            </>
          )}

          <button className="primary" onClick={() => setCalculated(true)} disabled={!parsed.values.length}>
            {activeTab === "groups" ? "Ułóż paczki" : "Znajdź dopłatę"} <span>→</span>
          </button>
          <p className="privacy">Dane nie opuszczają tego urządzenia.</p>
        </aside>

        <section className="results-panel">
          <div className="result-heading">
            <div>
              <p className="eyebrow">WYNIK</p>
              <h2>{activeTab === "groups" ? "Gotowy podział" : "Brakująca kwota"}</h2>
            </div>
            {activeTab === "groups" && <span className="result-count">{groups.length} {groups.length === 1 ? "paczka" : groups.length < 5 ? "paczki" : "paczek"}</span>}
          </div>

          {!calculated ? (
            <div className="empty"><strong>Dane zostały zmienione</strong><span>Uruchom obliczenia, aby odświeżyć wynik.</span></div>
          ) : activeTab === "groups" ? (
            <>
              <div className="stats">
                <div><span>Wartość</span><strong>{formatMoney(total)}</strong></div>
                <div><span>Wykorzystany potencjał</span><strong>{theoretical ? Math.round((groups.length / theoretical) * 100) : 0}%</strong></div>
                <div><span>Średnio w paczce</span><strong>{groups.length ? formatMoney(total / groups.length) : "—"}</strong></div>
              </div>
              {parsed.values.length > 18 && strategy === "optimal" && <p className="notice">Dla ponad 18 pozycji użyliśmy szybkiej, bezpiecznej strategii.</p>}
              <Results groups={groups} limit={limit} />
            </>
          ) : extra !== null ? (
            <>
              <div className="extra-result">
                <span>Minimalna dopłata</span>
                <strong>+ {formatMoney(extra)}</strong>
                <p>Ta kwota pozwala utworzyć jeszcze jedną pełną paczkę.</p>
                <div><span>{optimalGroups(parsed.values, limit).length} paczek</span><b>→</b><span>{groupsAfter.length} paczek</span></div>
              </div>
              <Results groups={groupsAfter} limit={limit} />
            </>
          ) : (
            <div className="empty"><strong>Za mały zakres dopłaty</strong><span>Zwiększ maksymalną dopłatę i spróbuj ponownie.</span></div>
          )}
        </section>
      </section>

      <footer><span>Paczki 50</span><span>Precyzyjne obliczenia co do grosza</span></footer>
    </main>
  );
}
