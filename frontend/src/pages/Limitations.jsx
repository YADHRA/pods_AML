import React from 'react';
import {ShieldAlert, Database, GitBranch, Gauge} from 'lucide-react';
const items=[
 ['Real data only','The UI does not manufacture transactions, risk scores, account nodes, or model metrics. Empty states mean the corresponding API output is not available.','Database'],
 ['Time-based leakage prevention','Historical account and graph features for a transaction on date D must use only records strictly before D.','GitBranch'],
 ['Risk is not proof','A model alert indicates a transaction for review; it is not proof of money laundering.','ShieldAlert'],
 ['Evaluation discipline','Use the frozen validation threshold on the held-out test set. Report PR-AUC prominently for the imbalanced dataset and include precision, recall, F1, ROC-AUC and confusion matrix.','Gauge']
];
const icons={Database,GitBranch,ShieldAlert,Gauge};
export default function Limitations(){return <div className="page-stack"><div className="notice info">Project safeguards and interpretation notes.</div><div className="two-col">{items.map(([title,body,icon])=>{const Icon=icons[icon];return <article className="card safeguard" key={title}><div className="safeguard-icon"><Icon size={20}/></div><div><h3>{title}</h3><p>{body}</p></div></article>})}</div></div>}