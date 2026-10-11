import React, { useMemo, useState, useEffect } from 'react';
import { EmptyState } from './Status.jsx';

const COLORS = {
center: '#60a5fa',
selected: '#e2e8f0',
link: '#64748b',
linked: '#60a5fa',
text: '#e2e8f0',
muted: '#94a3b8',
High: '#f87171',
Medium: '#fb923c',
Low: '#93a3b8',
'Not scored': '#64748b',
};

const formatAmount = (value) =>
value != null && Number.isFinite(Number(value))
? Number(value).toLocaleString('en-IN', {
minimumFractionDigits: 2,
maximumFractionDigits: 2,
})
: 'Unavailable';

const formatNumber = (value) =>
Number(value ?? 0).toLocaleString('en-IN');

const formatScore = (value) => {
if (
value == null ||
!Number.isFinite(Number(value))
) {
return 'Not scored';
}

return `${(Number(value) * 100).toFixed(4)}%`;
};

const getRiskLevel = (node) => {
const level = String(node?.risk_level ?? '').toLowerCase();

if (level === 'high') return 'High';
if (level === 'medium') return 'Medium';
if (level === 'low') return 'Low';

if (
node?.risk_score == null ||
!Number.isFinite(Number(node.risk_score))
) {
return 'Not scored';
}

return 'Not scored';
};

export default function GraphView({ data, onSelect }) {
const [selectedNodeId, setSelectedNodeId] = useState('');
const [selectedEdgeId, setSelectedEdgeId] = useState('');
const [depth, setDepth] = useState('all');

useEffect(() => {
setSelectedNodeId('');
setSelectedEdgeId('');
}, [data]);

const graph = useMemo(() => {
if (
!data ||
!Array.isArray(data.nodes) ||
!Array.isArray(data.edges)
) {
return null;
}


const W = 760;
const H = 470;
const cx = W / 2;
const cy = H / 2;

const centerNode =
  data.nodes.find((n) => n.is_center) ?? data.nodes[0];

const otherNodes = data.nodes.filter((n) => n !== centerNode);
const positioned = [];

if (centerNode) {
  positioned.push({
    ...centerNode,
    id: String(centerNode.id),
    x: cx,
    y: cy,
  });
}

// Place accounts in rings using the actual BFS depth.
const grouped = new Map();

for (const node of otherNodes) {
  const ring = Math.max(1, Number(node.depth ?? 1));

  if (!grouped.has(ring)) grouped.set(ring, []);
  grouped.get(ring).push(node);
}

for (const [ring, ringNodes] of [...grouped.entries()].sort(
  ([a], [b]) => a - b
)) {
  const radius = Math.min(205, 125 + (ring - 1) * 65);

  ringNodes.forEach((node, index) => {
    const angle =
      (2 * Math.PI * index) / Math.max(ringNodes.length, 1) -
      Math.PI / 2;

    positioned.push({
      ...node,
      id: String(node.id),
      x: cx + radius * Math.cos(angle),
      y: cy + radius * 0.76 * Math.sin(angle),
    });
  });
}

const byId = new Map(positioned.map((node) => [node.id, node]));

const edges = data.edges
  .map((edge, index) => ({
    ...edge,
    id: `${edge.source}-${edge.target}-${index}`,
    source: String(edge.source),
    target: String(edge.target),
  }))
  .filter(
    (edge) => byId.has(edge.source) && byId.has(edge.target)
  );

return {
  nodes: positioned,
  edges,
  byId,
  centerId: centerNode ? String(centerNode.id) : '',
};


}, [data]);

if (!graph || graph.nodes.length === 0) {
return ( <EmptyState
     title="Graph data not available"
     detail="No graph nodes were returned. Check the backend and try again."
   />
);
}

const shownEdges =
depth === 'all'
? graph.edges
: graph.edges.filter((edge) => {
const source = graph.byId.get(edge.source);
const target = graph.byId.get(edge.target);


      return (
        Number(source?.depth ?? 0) <= Number(depth) &&
        Number(target?.depth ?? 0) <= Number(depth)
      );
    });


const activeNode = graph.byId.get(selectedNodeId);
const activeEdge = shownEdges.find((edge) => edge.id === selectedEdgeId);

const selectNode = (node) => {
if (!node) return;

```
setSelectedNodeId(node.id);
setSelectedEdgeId('');
onSelect?.(node);
```

};

const selectEdge = (edge) => {
setSelectedEdgeId(edge.id);
setSelectedNodeId('');
};

const totalFound = Number(
data.total_nodes_found ?? data.nodes.length
);

const truncated = Boolean(data.truncated);
const scoredCount = graph.nodes.filter(
(node) => getRiskLevel(node) !== 'Not scored'
).length;

const highCount = graph.nodes.filter(
(node) => getRiskLevel(node) === 'High'
).length;

const mediumCount = graph.nodes.filter(
(node) => getRiskLevel(node) === 'Medium'
).length;

return ( <div className="graph-layout"> <div className="graph-canvas"> <div className="graph-toolbar"> <label>
Link depth
<select
value={depth}
onChange={(event) => setDepth(event.target.value)}
> <option value="all">All returned links</option> <option value="1">Through depth 1</option> <option value="2">Through depth 2</option> <option value="3">Through depth 3</option> </select> </label>

```
      <span>
        {formatNumber(graph.nodes.length)} accounts ·{' '}
        {formatNumber(shownEdges.length)} links
      </span>
    </div>

    <div className="graph-summary">
      <div className="graph-summary-item">
        <span>Accounts displayed</span>
        <strong>{formatNumber(graph.nodes.length)}</strong>
      </div>

      <div className="graph-summary-item">
        <span>Reachable accounts found</span>
        <strong>{formatNumber(totalFound)}</strong>
      </div>

      <div className="graph-summary-item">
        <span>Scored accounts</span>
        <strong>{formatNumber(scoredCount)}</strong>
      </div>

      <div className="graph-summary-item">
        <span>High / Medium</span>
        <strong>
          {formatNumber(highCount)} / {formatNumber(mediumCount)}
        </strong>
      </div>
    </div>

    {truncated && (
      <div className="graph-warning" role="status">
        <strong>Graph limited by node cap.</strong>{' '}
        Showing {formatNumber(graph.nodes.length)} accounts out of{' '}
        {formatNumber(totalFound)} found by the backend. Increase the
        maximum node limit or reduce the search depth to explore a
        manageable network.
      </div>
    )}

    <svg
      viewBox="0 0 760 470"
      role="img"
      aria-label="Account transaction network. Node colours represent backend risk levels."
    >
      <defs>
        <filter
          id="center-glow"
          x="-100%"
          y="-100%"
          width="300%"
          height="300%"
        >
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {shownEdges.map((edge) => {
        const source = graph.byId.get(edge.source);
        const target = graph.byId.get(edge.target);
        const isSelected = edge.id === selectedEdgeId;

        const isConnectedToSelected =
          edge.source === selectedNodeId ||
          edge.target === selectedNodeId;

        const transactionCount = Number(edge.txn_count ?? 0);

        const width = Math.min(
          4,
          1 + Math.log2(Math.max(transactionCount, 1)) * 0.5
        );

        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;

        return (
          <g
            key={edge.id}
            role="button"
            tabIndex={0}
            aria-label={`Link ${edge.source} to ${edge.target}, ${transactionCount} transactions`}
            onClick={() => selectEdge(edge)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                selectEdge(edge);
              }
            }}
            style={{ cursor: 'pointer', outline: 'none' }}
          >
            <line
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="transparent"
              strokeWidth="14"
            />

            <line
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={
                isSelected || isConnectedToSelected
                  ? COLORS.linked
                  : COLORS.link
              }
              strokeWidth={isSelected ? 3.5 : width}
              strokeDasharray={isSelected ? '6 4' : undefined}
              opacity={
                isSelected || isConnectedToSelected ? 1 : 0.5
              }
              pointerEvents="none"
            />

            {transactionCount > 1 && (
              <text
                x={midX}
                y={midY - 5}
                textAnchor="middle"
                fontSize="10"
                fill={COLORS.text}
                pointerEvents="none"
              >
                ×{formatNumber(transactionCount)}
              </text>
            )}

            <title>
              {`Node ${edge.source} → Node ${edge.target}. Transactions: ${transactionCount}. Total amount: ${formatAmount(edge.total_amount)}. Last active: ${edge.last_date ?? 'Unavailable'}`}
            </title>
          </g>
        );
      })}

      {graph.nodes.map((node) => {
        const isCenter = node.id === graph.centerId;
        const isSelected = node.id === selectedNodeId;
        const riskLevel = getRiskLevel(node);
        const nodeColor = COLORS[riskLevel];

        const radius = isSelected ? 25 : isCenter ? 23 : 18;

        return (
          <g
            key={node.id}
            transform={`translate(${node.x} ${node.y})`}
            role="button"
            tabIndex={0}
            aria-label={`Account ${node.account ?? node.id}, ${riskLevel} risk`}
            onClick={() => selectNode(node)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                selectNode(node);
              }
            }}
            className="graph-node"
            style={{ cursor: 'pointer', outline: 'none' }}
          >
            <circle
              r={radius}
              fill={nodeColor}
              fillOpacity={riskLevel === 'Not scored' ? 0.12 : 0.4}
              stroke={
                isSelected
                  ? COLORS.selected
                  : isCenter
                    ? COLORS.center
                    : nodeColor
              }
              strokeWidth={isSelected || isCenter ? 3 : 1.8}
              strokeDasharray={
                riskLevel === 'Not scored' ? '4 3' : undefined
              }
              filter={isCenter ? 'url(#center-glow)' : undefined}
            />

            <text
              textAnchor="middle"
              dy="4"
              fontSize="10"
              fontWeight="600"
              fill={COLORS.text}
              pointerEvents="none"
            >
              {String(node.bank ?? 'Account').slice(0, 6)}
            </text>

            <text
              textAnchor="middle"
              dy="34"
              fontSize="10"
              fill={COLORS.text}
              pointerEvents="none"
            >
              {node.id}
            </text>

            <title>
              {`Node ID: ${node.id} | Bank: ${node.bank ?? 'Unavailable'} | Account: ${node.account ?? 'Unavailable'} | Depth: ${node.depth ?? 0} | Risk level: ${riskLevel} | Model score: ${formatScore(node.risk_score)}${isCenter ? ' | Center account' : ''}`}
            </title>
          </g>
        );
      })}
    </svg>

    <div className="legend">
      {['High', 'Medium', 'Low', 'Not scored'].map((level) => (
        <span key={level}>
          <i
            style={{
              background: COLORS[level],
              opacity: level === 'Not scored' ? 0.65 : 1,
            }}
          />
          {level}
        </span>
      ))}

      <span>
        <i style={{ background: COLORS.center }} />
        Center account outline
      </span>

      <span>
        <i style={{ background: COLORS.linked }} />
        Selected connection
      </span>
    </div>

    <p className="muted">
      Node colours use risk levels supplied by the backend. Model scores
      are not necessarily calibrated probabilities. Not scored means no
      usable historical score was supplied. Lines represent account
      connections; line thickness reflects transaction count. Node
      positions are visual and do not represent geography.
    </p>
  </div>

  <aside className="card graph-details">
    {activeEdge ? (
      <>
        <h3>Transaction link</h3>

        <div className="detail-id">
          Node {activeEdge.source} → Node {activeEdge.target}
        </div>

        <dl className="detail-list">
          <dt>Source node</dt>
          <dd>{activeEdge.source}</dd>

          <dt>Source account</dt>
          <dd>
            {graph.byId.get(activeEdge.source)?.account ?? 'Unavailable'}
          </dd>

          <dt>Destination node</dt>
          <dd>{activeEdge.target}</dd>

          <dt>Destination account</dt>
          <dd>
            {graph.byId.get(activeEdge.target)?.account ?? 'Unavailable'}
          </dd>

          <dt>Transaction count</dt>
          <dd>{formatNumber(activeEdge.txn_count)}</dd>

          <dt>Total amount paid</dt>
          <dd>{formatAmount(activeEdge.total_amount)}</dd>

          <dt>Last active date</dt>
          <dd>{activeEdge.last_date ?? 'Unavailable'}</dd>

          <dt>Snapshot date</dt>
          <dd>{data.date ?? 'Unavailable'}</dd>
        </dl>

        <button
          type="button"
          className="button"
          onClick={() => {
            setSelectedEdgeId('');
            selectNode(graph.byId.get(activeEdge.source));
          }}
        >
          Inspect source account
        </button>
      </>
    ) : activeNode ? (
      <>
        <h3>Selected account</h3>

        <div className="detail-id">
          {activeNode.account ?? `Node ${activeNode.id}`}
        </div>

        <div
          className="risk-badge"
          style={{
            color: COLORS[getRiskLevel(activeNode)],
            borderColor: COLORS[getRiskLevel(activeNode)],
          }}
        >
          {getRiskLevel(activeNode)} risk
        </div>

        <dl className="detail-list">
          <dt>Node ID</dt>
          <dd>{activeNode.id}</dd>

          <dt>Bank</dt>
          <dd>{activeNode.bank ?? 'Unavailable'}</dd>

          <dt>Account</dt>
          <dd>{activeNode.account ?? 'Unavailable'}</dd>

          <dt>Center account</dt>
          <dd>{activeNode.is_center ? 'Yes' : 'No'}</dd>

          <dt>Depth from center</dt>
          <dd>{activeNode.depth ?? 0}</dd>

          <dt>Historical model score</dt>
          <dd>{formatScore(activeNode.risk_score)}</dd>

          <dt>Snapshot date</dt>
          <dd>{data.date ?? 'Unavailable'}</dd>
        </dl>

        <p className="muted">
          This score summarises the historical model-score value returned
          by the backend. It should not be interpreted as proof of
          laundering.
        </p>
      </>
    ) : (
      <>
        <h3>Explore the network</h3>

        <p className="muted">
          Select an account to inspect its bank, model score and risk
          level. Click a connection to view its transaction count, total
          amount and last active date.
        </p>

        <div className="detail-hint">
          <strong>Investigation tip</strong>
          <p>
            Start with the centre account, then explore connected
            accounts and compare their historical model scores.
          </p>
        </div>
      </>
    )}
  </aside>
</div>


);
}
