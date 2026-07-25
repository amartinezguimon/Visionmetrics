import { NavLink } from "react-router-dom";
import { initials } from "../lib/format";
import type { StoreInfo } from "../lib/types";

interface Props {
  stores: StoreInfo[];
  selectedStoreId: string;
  onSelectStore: (id: string) => void;
  storeName: string;
  live: boolean;
  onLogout: () => void;
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  [
    "rounded-[24px] px-[18px] py-[7px] text-[13px] cursor-pointer",
    isActive
      ? "bg-purple font-medium text-white shadow-[0_2px_12px_rgba(61,26,110,0.28)]"
      : "text-gray2 hover:text-dark hover:bg-gray6",
  ].join(" ");

export default function TopBar({
  stores,
  selectedStoreId,
  onSelectStore,
  storeName,
  live,
  onLogout,
}: Props) {
  return (
    <div className="mb-6 flex items-center justify-between">
      <div className="rounded-[30px] bg-purple px-5 py-2 text-[15px] font-bold text-white shadow-[0_4px_16px_rgba(61,26,110,0.28)] transition-transform hover:-translate-y-0.5">
        VisionMetrics
      </div>

      <nav className="flex gap-[3px] rounded-[30px] bg-card p-1">
        <NavLink to="/" end className={navClass}>
          Home
        </NavLink>
        <NavLink to="/analysis" className={navClass}>
          Analysis
        </NavLink>
      </nav>

      <div className="flex items-center gap-2.5">
        {stores.length > 1 && (
          <select
            value={selectedStoreId}
            onChange={(e) => onSelectStore(e.target.value)}
            className="cursor-pointer rounded-[30px] bg-card px-4 py-[7px] text-[13px] text-gray2 outline-none"
            aria-label="Seleccionar tienda"
          >
            {stores.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        )}

        <div className="flex items-center gap-[7px] rounded-[30px] bg-card px-4 py-[7px] text-[13px] text-gray2">
          <span
            className={[
              "h-[7px] w-[7px] rounded-full",
              live ? "live-dot bg-live" : "bg-gray4",
            ].join(" ")}
          />
          {live ? "En directo" : "Sin conexión"}
        </div>

        <button
          onClick={onLogout}
          title={`${storeName} — cerrar sesión`}
          className="flex h-[34px] w-[34px] items-center justify-center rounded-full bg-purple text-[12px] font-semibold text-white transition hover:-translate-y-0.5 hover:shadow-[0_6px_16px_rgba(61,26,110,0.30)]"
        >
          {initials(storeName)}
        </button>
      </div>
    </div>
  );
}
