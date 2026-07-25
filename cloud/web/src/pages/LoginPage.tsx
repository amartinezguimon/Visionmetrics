import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await login(email.trim(), password);
      navigate("/", { replace: true });
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? "Correo o contraseña incorrectos."
          : "No se pudo iniciar sesión. Inténtalo de nuevo.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg px-6">
      <div className="w-full max-w-[380px]">
        <div className="mb-6 flex justify-center">
          <div className="rounded-[30px] bg-purple px-5 py-2 text-[15px] font-bold text-white">
            VisionMetrics
          </div>
        </div>

        <form onSubmit={onSubmit} className="rounded-card bg-card p-7 shadow-sm">
          <h1 className="mb-5 text-[18px] font-semibold text-dark">Inicia sesión</h1>

          <label className="mb-1 block text-[12px] font-medium text-slate">
            Correo
          </label>
          <input
            type="email"
            autoComplete="username"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mb-4 w-full rounded-lg border border-gray5 px-3 py-2.5 text-[14px] outline-none focus:border-purple"
            placeholder="tu@correo.com"
          />

          <label className="mb-1 block text-[12px] font-medium text-slate">
            Contraseña
          </label>
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mb-5 w-full rounded-lg border border-gray5 px-3 py-2.5 text-[14px] outline-none focus:border-purple"
            placeholder="••••••••"
          />

          {error && (
            <div className="mb-4 rounded-lg bg-danger/10 px-3 py-2 text-[12px] font-medium text-danger">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-purple py-2.5 text-[14px] font-semibold text-white transition hover:opacity-90 disabled:opacity-60"
          >
            {busy ? "Entrando…" : "Entrar"}
          </button>
        </form>

        <p className="mt-4 text-center text-[11px]">
          <a href="/staff/login" className="text-gray4 hover:text-purple">
            Acceso del equipo
          </a>
        </p>
      </div>
    </div>
  );
}
