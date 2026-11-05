export type RunStatus = "queued" | "running" | "completed" | "failed";
export interface RunConfig {
  n_patients: number;
  seed: number;
  threshold_profile: string;
  [key: string]: unknown;
}
export interface Run {
  id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  progress: number;
  config: RunConfig;
  error: string | null;
  summary: Record<string, unknown> | null;
}
export async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const error = await response.json();
      if (typeof error.detail === "string") {
        message = error.detail;
      } else if (Array.isArray(error.detail)) {
        message = error.detail
          .map((item: { loc?: string[]; msg?: string }) => {
            const field = item.loc?.slice(1).join(" ").replaceAll("_", " ");
            return `${field ? `${field}: ` : ""}${item.msg || "Invalid input"}`;
          })
          .join(". ");
      }
    } catch {
      /* no JSON */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const api = {
  health: () => request<{ status: string }>("/health"),
  runs: () => request<{ runs: Run[] }>("/runs"),
  run: (id: string) => request<Run>(`/runs/${encodeURIComponent(id)}`),
  create: (config: RunConfig) =>
    request<Run>("/runs", { method: "POST", body: JSON.stringify(config) }),
  results: (id: string) =>
    request<Record<string, unknown>>(`/runs/${encodeURIComponent(id)}/results`),
};
