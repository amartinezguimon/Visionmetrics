import { useEffect, useMemo, useState } from "react";
import { useDashboardCtx } from "./DashboardLayout";
import HeroChart from "../components/charts/HeroChart";
import MonthlyChart from "../components/charts/MonthlyChart";
import MiniCalendar from "../components/MiniCalendar";
import { MONTHS } from "../lib/format";

const ZEROS12 = Array(12).fill(0);

export default function HomePage() {
  const { data, storeName } = useDashboardCtx();
  const { ensureMonth, getDaily, getHourly, loading, error } = data;

  const now = useMemo(() => new Date(), []);
  const [viewYear, setViewYear] = useState(now.getFullYear());
  const [viewMonth0, setViewMonth0] = useState(now.getMonth());
  const [selectedDay, setSelectedDay] = useState(now.getDate());

  useEffect(() => {
    ensureMonth(viewYear, viewMonth0);
  }, [ensureMonth, viewYear, viewMonth0]);

  const daily = getDaily(viewYear, viewMonth0);
  const hourly = getHourly(viewYear, viewMonth0);

  const isCurrentMonth = viewYear === now.getFullYear() && viewMonth0 === now.getMonth();
  const todayDay = isCurrentMonth ? now.getDate() : null;

  const days = useMemo(
    () => (daily ? Object.keys(daily).map(Number).sort((a, b) => a - b) : []),
    [daily],
  );
  const hasMonthData = days.length > 0;

  // Monthly averages (right-panel "media mensual").
  const allTotals = days.map((d) => daily![String(d)].total);
  const allRates = days.map((d) => daily![String(d)].rate);
  const avgTot = allTotals.length ? Math.round(allTotals.reduce((a, b) => a + b, 0) / allTotals.length) : 0;
  const avgRate = allRates.length ? Math.round(allRates.reduce((a, b) => a + b, 0) / allRates.length) : 0;

  const dayStat = daily?.[String(selectedDay)];
  const dayTotal = dayStat?.total ?? 0;
  const dayRate = dayStat?.rate ?? 0;

  const hb = hourly?.[String(selectedDay)];
  const passing = hb?.passing ?? ZEROS12;
  const looking = hb?.looking ?? ZEROS12;

  const heroLabel =
    todayDay === selectedDay
      ? "Actividad de hoy"
      : `Actividad del ${selectedDay} de ${MONTHS[viewMonth0].toLowerCase()}`;

  function navMonth(dir: number) {
    let m = viewMonth0 + dir;
    let y = viewYear;
    if (m > 11) { m = 0; y++; }
    if (m < 0) { m = 11; y--; }
    setViewYear(y);
    setViewMonth0(m);
  }

  return (
    <div>
      <div className="mb-[18px] text-[20px] font-semibold text-dark">
        Bienvenido, <span className="text-gray4">{storeName}</span>
      </div>

      <div className="mb-[14px] grid grid-cols-[6fr_2fr] gap-[14px]">
        {/* HERO CHART */}
        <div className="flex flex-col rounded-card bg-card p-[22px]">
          <div className="mb-[18px] flex shrink-0 items-start justify-between">
            <div className="text-[17px] font-semibold -tracking-[0.3px]">{heroLabel}</div>
            <div className="flex gap-4">
              <Legend color="#D8D8D8">Pasaron</Legend>
              <Legend color="var(--purple)">Miraron +2s</Legend>
            </div>
          </div>
          <div className="relative min-h-[260px] flex-1">
            {hasMonthData ? (
              <HeroChart passing={passing} looking={looking} />
            ) : (
              <EmptyHint loading={loading} error={error} month={MONTHS[viewMonth0]} year={viewYear} />
            )}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="flex flex-col overflow-hidden rounded-card bg-card p-[22px]">
          <MiniCalendar
            year={viewYear}
            monthIndex0={viewMonth0}
            selectedDay={selectedDay}
            todayDay={todayDay}
            daily={daily}
            onSelectDay={setSelectedDay}
            onPrev={() => navMonth(-1)}
            onNext={() => navMonth(1)}
          />

          <Divider />
          <Stat
            label="Personas pasando"
            value={dayTotal}
            valueColor={dayTotal > avgTot ? "#22863A" : "#C4604A"}
            sub={<>media mensual: <b>{avgTot}</b></>}
          />
          <Divider />
          <Stat
            label="Tasa de atención"
            value={`${dayRate}%`}
            valueColor={dayRate > avgRate ? "#22863A" : "#C4604A"}
            sub={<>media mensual: <b>{avgRate}%</b></>}
          />
        </div>
      </div>

      {/* MONTHLY CHART */}
      <div className="rounded-card bg-card p-[22px]">
        <div className="mb-1.5 flex items-start justify-between">
          <div className="text-[16px] font-semibold -tracking-[0.2px]">
            {MONTHS[viewMonth0]} {viewYear}
          </div>
          <div className="flex gap-[18px]">
            <LineLegend color="#1C1C1C">Media pasando</LineLegend>
            <LineLegend color="#F0DC6A">Media mirando</LineLegend>
          </div>
        </div>
        <div className="relative mt-2.5 h-[220px]">
          {hasMonthData ? (
            <MonthlyChart
              days={days}
              totals={allTotals}
              looks={days.map((d) => daily![String(d)].looking)}
              rates={allRates}
              onSelectDay={setSelectedDay}
            />
          ) : (
            <EmptyHint loading={loading} error={error} month={MONTHS[viewMonth0]} year={viewYear} />
          )}
        </div>
      </div>
    </div>
  );
}

function Legend({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-[7px] text-[12px] text-gray3">
      <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: color }} />
      {children}
    </div>
  );
}

function LineLegend({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-[7px] text-[11px] text-gray3">
      <span className="h-[2px] w-[22px] rounded-[2px]" style={{ background: color }} />
      {children}
    </div>
  );
}

function Divider() {
  return <div className="mb-[14px] h-px bg-gray5" />;
}

function Stat({
  label,
  value,
  valueColor,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  valueColor: string;
  sub: React.ReactNode;
}) {
  return (
    <div className="mb-[14px] text-center">
      <div className="mb-1.5 text-[9.5px] font-semibold uppercase tracking-[0.08em] text-gray3">{label}</div>
      <div className="text-[44px] font-extrabold leading-none -tracking-[2.5px]" style={{ color: valueColor }}>
        {value}
      </div>
      <div className="mt-1 text-[10px] text-gray3">{sub}</div>
    </div>
  );
}

function EmptyHint({
  loading,
  error,
  month,
  year,
}: {
  loading: boolean;
  error: string | null;
  month: string;
  year: number;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      {loading ? (
        <>
          <div className="spinner h-7 w-7 rounded-full border-[3px] border-gray5 border-t-purple" />
          <div className="text-[12px] text-gray3">Cargando…</div>
        </>
      ) : error ? (
        <div className="text-[12px] text-danger">{error}</div>
      ) : (
        <>
          <div className="text-[13px] font-medium text-gray3">
            Aún no hay datos de {month} {year}
          </div>
        </>
      )}
    </div>
  );
}
