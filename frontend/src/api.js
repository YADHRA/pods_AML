const BASE = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'
).replace(/\/$/, '');

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { Accept: 'application/json' },
    ...options,
  });

  if (!response.ok) {
    let detail = '';
    try {
      const j = await response.json();
      if (typeof j.detail === 'string') detail = j.detail;
      else if (j.message) detail = j.message;
    } catch {
      detail = await response.text().catch(() => '');
    }

    throw new Error(
      detail || `${response.status} ${response.statusText}`
    );
  }

  return response.json();
}

export const api = {
  base: BASE,

  // Health
  health: () => request('/api/health'),

  // Dashboard / Overview
  overview: () => request('/api/dashboard/summary'),
  daily: () => request('/api/data/daily'),
  splits: () => request('/api/data/splits'),
  quality: () => request('/api/data/quality'),

  // Alerts
  alerts: (params = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') q.append(k, String(v));
    });
    return request(`/api/alerts?${q.toString()}`);
  },
  accountAlerts: (params = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') q.append(k, String(v));
    });
    return request(`/api/alerts/accounts?${q.toString()}`);
  },

  // Accounts
  accountLookup: (q) => request(`/api/accounts/lookup?q=${encodeURIComponent(q)}`),
  account: (id, date = '') =>
    request(`/api/accounts/${encodeURIComponent(id)}${date ? `?date=${encodeURIComponent(date)}` : ''}`),

  // Transactions & SHAP explanations
  transaction: (id) => request(`/api/transactions/${encodeURIComponent(id)}`),
  transactionExplanation: (id, model = 'm2') =>
    request(`/api/transactions/${encodeURIComponent(id)}/explanation?model=${encodeURIComponent(model)}`),

  // Models evaluation & curves
  evaluation: () => request('/api/models/evaluation'),
  curves: (model = 'm2', split = 'test') =>
    request(`/api/models/curves?model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}`),
  shap: (model = 'm2') =>
    request(`/api/models/shap?model=${encodeURIComponent(model)}`),
  comparison: () => request('/api/models/comparison'),
  prediction: (id) => request(`/api/models/predictions/${encodeURIComponent(id)}`),
};
