
import React, { useCallback, useEffect, useState } from 'react';
import { RefreshCw, Network as NetworkIcon } from 'lucide-react';
import { api } from '../api.js';
import GraphView from '../components/GraphView.jsx';
import { Loading, Status } from '../components/Status.jsx';

export default function Network() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);

  const [nodeId, setNodeId] = useState('0');
  const [snapshot, setSnapshot] = useState('2022-09-18');
  const [depth, setDepth] = useState(2);
  const [maxNodes, setMaxNodes] = useState(60);

  const load = useCallback(() => {
    setLoading(true);
    setError('');
    setSelected(null);

    api
      .graph({
        node_id: nodeId,
        date: snapshot,
        depth,
        max_nodes: maxNodes,
      })
      .then((result) => {
        setData(result);
      })
      .catch((err) => {
        setError(err.message || 'Unable to load the transaction graph.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [nodeId, snapshot, depth, maxNodes]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="page-stack">
      {/* Graph controls */}
      <div className="toolbar card">
        <div className="toolbar-title">
          <NetworkIcon size={18} />
          <span>Transaction Network</span>
        </div>

        <label>
          Account node ID
          <input
            type="number"
            min="0"
            value={nodeId}
            onChange={(event) => setNodeId(event.target.value)}
          />
        </label>

        <label>
          Snapshot date
          <input
            type="date"
            value={snapshot}
            onChange={(event) => setSnapshot(event.target.value)}
          />
        </label>

        <label>
          Link depth
          <select
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value))}
          >
            <option value={1}>1 - Direct connections</option>
            <option value={2}>2 - Two levels</option>
            <option value={3}>3 - Three levels</option>
          </select>
        </label>

        <label>
          Node limit
          <select
            value={maxNodes}
            onChange={(event) => setMaxNodes(Number(event.target.value))}
          >
            <option value={30}>30 nodes</option>
            <option value={60}>60 nodes</option>
            <option value={100}>100 nodes</option>
          </select>
        </label>

        <button
          type="button"
          className="button"
          onClick={load}
          disabled={loading}
        >
          <RefreshCw size={15} />
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <Status kind="warning">
          Could not load graph data: {error}. Check that the backend is
          running at {api.base}, and verify the account node ID and snapshot
          date.
        </Status>
      )}

      {/* Loading state */}
      {loading && (
        <Loading label="Loading real transaction graph..." />
      )}

      {/* Graph results */}
      {!loading && data && (
        <>
          <div className="card">
            <div className="toolbar-title">
              Transaction Network for Account Node {data.node_id}
            </div>

            <p className="muted">
              Snapshot: {data.date} · Depth: {data.depth} ·
              {' '}{data.nodes?.length ?? 0} displayed nodes ·
              {' '}{data.edges?.length ?? 0} links
            </p>

            {data.truncated && (
              <Status kind="info">
                Showing {data.nodes?.length ?? 0} of{' '}
                {data.total_nodes_found ?? 'more'} reachable accounts.
                Increase the node limit or change the depth to explore
                the network.
              </Status>
            )}

            {!data.truncated && (
              <p className="muted">
                All accounts returned by the backend for these graph
                settings are displayed.
              </p>
            )}
          </div>

          {/* Interactive graph */}
          <GraphView
            data={data}
            onSelect={setSelected}
          />

          {/* Selected account details */}
          {selected && (
            <div className="card">
              <div className="toolbar-title">
                Selected Account Details
              </div>

              <p>
                <strong>Node ID:</strong> {selected.id}
              </p>

              <p>
                <strong>Bank:</strong> {selected.bank}
              </p>

              <p>
                <strong>Account:</strong> {selected.account}
              </p>

              {selected.depth !== undefined && (
                <p>
                  <strong>Depth from center:</strong> {selected.depth}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

