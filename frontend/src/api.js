
const BASE = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'
).replace(/\/$/, '');

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { Accept: 'application/json' },
    ...options,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');

    throw new Error(
      `${response.status} ${response.statusText}${
        detail ? ` — ${detail.slice(0, 180)}` : ''
      }`
    );
  }

  return response.json();
}

export const api = {
  base: BASE,

  // Check whether the backend is running
  health: () => request('/api/health'),

  // Dashboard summary
  overview: () => request('/api/dashboard/summary'),

  // Model information
  models: () => request('/api/models'),
  evaluation: () => request('/api/models/evaluation'),

  // Transaction graph
  

graph: ({
  node_id = 0,
  date = '2022-09-18',
  depth = 2,
  max_nodes = 60,
} = {}) => {
  const query = new URLSearchParams({
    date,
    depth: String(depth),
    max_nodes: String(max_nodes),
  });

  return request(
    `/api/graph/${encodeURIComponent(node_id)}?${query.toString()}`
  );
},


  // Transaction investigation
  transaction: (id) =>
    request(`/api/transactions/${encodeURIComponent(id)}`),

  // Account investigation
  account: (id) =>
    request(`/api/accounts/${encodeURIComponent(id)}`),

  // Prediction for a transaction
  prediction: (id) =>
    request(`/api/models/predictions/${encodeURIComponent(id)}`),
};

