// Maps technical feature names to human-readable plain English labels
// and contextual explanation sentences. NO jargon (PageRank, WCC, SHAP).

export const FEATURE_INFO = {
  // --- Pure Transaction Features (M1 & M2) ---
  t_amount_z_in_currency: {
    label: 'Unusual transaction amount',
    explain: (val, shap) =>
      shap > 0
        ? 'Transaction amount is significantly higher than typical for this currency.'
        : 'Transaction amount is standard or modest for this payment type.',
  },
  t_log_amount_paid: {
    label: 'Overall payment size',
    explain: (val, shap) =>
      shap > 0
        ? 'High total value transfer.'
        : 'Low to moderate transaction size.',
  },
  t_payment_format: {
    label: 'Payment channel',
    explain: (val) =>
      val ? `Transferred via ${val}, a channel with elevated monitoring weight.` : 'Specific payment format flagged under monitoring thresholds.',
  },
  t_cur_paid: {
    label: 'Currency profile',
    explain: (val) =>
      val ? `Currency denominated in ${val}.` : 'Currency flow pattern.',
  },
  t_is_fx: {
    label: 'Currency exchange (FX)',
    explain: (val) =>
      val && val !== '0'
        ? 'Cross-currency conversion involved between sender and receiver.'
        : 'Domestic single-currency transaction.',
  },
  t_is_self_loop: {
    label: 'Self-transfer loop',
    explain: () => 'Funds transferred back to the originating account entity.',
  },
  t_same_bank: {
    label: 'Inter-bank vs intra-bank transfer',
    explain: (val) =>
      val && val !== '0'
        ? 'Transfer remained within the same financial institution.'
        : 'Cross-institution transfer between different banks.',
  },
  t_hour: {
    label: 'Time of execution',
    explain: (val) =>
      val != null ? `Executed at ${val}:00 hours.` : 'Executed at an unusual hour of activity.',
  },

  // --- Pair / Relationship Features (M2) ---
  g_pair_seen_before: {
    label: 'New counterparty relationship',
    explain: (val) =>
      val === 0 || val === '0'
        ? 'First time this sender has sent money to this receiver.'
        : 'Sender and receiver have transacted prior to this transfer.',
  },
  g_pair_prior_count_log: {
    label: 'Relationship transaction volume',
    explain: (val, shap) =>
      shap > 0
        ? 'Rapid or high-frequency transfers between this pair.'
        : 'Infrequent prior transfers between this sender and receiver.',
  },
  g_reverse_pair_seen: {
    label: 'Two-way transfer pattern',
    explain: (val) =>
      val && val !== '0'
        ? 'Receiver previously sent money back to this sender (circular flow).'
        : 'One-directional payment flow with no reciprocal history.',
  },

  // --- Network / Account Activity Features (M2) ---
  g_src_n_unique_out_log: {
    label: 'Sender counterparty fan-out',
    explain: () => 'Sender is distributing funds outward to many distinct accounts.',
  },
  g_dst_n_unique_in_log: {
    label: 'Receiver counterparty fan-in',
    explain: () => 'Receiver is rapidly collecting funds from many separate senders.',
  },
  g_src_pagerank_log: {
    label: 'Sender network centrality',
    explain: (val, shap) =>
      shap > 0
        ? 'Sender is positioned centrally in connected financial transaction clusters.'
        : 'Sender is in a peripheral network cluster.',
  },
  g_dst_pagerank_log: {
    label: 'Receiver network centrality',
    explain: (val, shap) =>
      shap > 0
        ? 'Receiver is a high-volume nexus account in the transaction network.'
        : 'Receiver has standard network positioning.',
  },
  g_src_wcc_size_log: {
    label: 'Sender transaction cluster size',
    explain: () => 'Sender belongs to a large, interconnected network of transacting parties.',
  },
  g_dst_wcc_size_log: {
    label: 'Receiver transaction cluster size',
    explain: () => 'Receiver belongs to a large interconnected account cluster.',
  },
  g_src_has_received_before: {
    label: 'Sender funding history',
    explain: (val) =>
      val && val !== '0'
        ? 'Sender account previously received inbound deposits.'
        : 'Sender initiated outbound transfer without previous inward deposits.',
  },
  g_dst_has_sent_before: {
    label: 'Receiver forwarding history',
    explain: (val) =>
      val && val !== '0'
        ? 'Receiver has history of relaying funds onwards.'
        : 'Receiver account has solely accumulated inward transfers.',
  },

  // --- Account History Features (M2) ---
  a_src_n_out_rate: {
    label: 'Sender outbound transfer rate',
    explain: () => 'Elevated frequency of outgoing transfers from the sender.',
  },
  a_src_n_in_rate: {
    label: 'Sender inbound transfer rate',
    explain: () => 'Frequent incoming deposits preceding this transaction.',
  },
  a_src_max_z_out: {
    label: 'Sender peak outbound amount',
    explain: () => 'Spike in sender withdrawal or payment volume.',
  },
  a_src_mean_z_out: {
    label: 'Sender average outbound volume',
    explain: () => 'Higher than usual average expenditure from sender.',
  },
  a_src_mean_z_in: {
    label: 'Sender average inbound volume',
    explain: () => 'Higher than usual average deposit size into sender account.',
  },
  a_src_days_since_last: {
    label: 'Sender dormancy burst',
    explain: (val) =>
      val ? `Sudden transaction activity after ${val} days of inactivity.` : 'Sudden burst after dormancy.',
  },
  a_src_days_active_frac: {
    label: 'Sender account age & activity',
    explain: () => 'Intermittent or newly opened sender account activity.',
  },
  a_src_fmt_diversity: {
    label: 'Sender payment channel switching',
    explain: () => 'Sender utilizes multiple disparate payment rails.',
  },
  a_dst_n_in_rate: {
    label: 'Receiver incoming velocity',
    explain: () => 'Receiver account is absorbing a high velocity of inbound payments.',
  },
  a_dst_n_out_rate: {
    label: 'Receiver pass-through velocity',
    explain: () => 'Receiver rapidly passes funds through after receipt.',
  },
  a_dst_max_z_out: {
    label: 'Receiver peak outbound amount',
    explain: () => 'Substantial outward withdrawal spikes by the recipient entity.',
  },
  a_dst_mean_z_out: {
    label: 'Receiver average outbound amount',
    explain: () => 'Higher than expected average outward transaction volume.',
  },
  a_dst_mean_z_in: {
    label: 'Receiver average inbound amount',
    explain: () => 'Significant average inward deposit size into receiver account.',
  },
  a_dst_days_since_last: {
    label: 'Receiver dormancy burst',
    explain: (val) =>
      val ? `Receiver reactivated after ${val} days of silence.` : 'Sudden reactivation after period of dormancy.',
  },
  a_dst_days_active_frac: {
    label: 'Receiver account activity consistency',
    explain: () => 'Irregular transaction schedule on receiver account.',
  },
  a_dst_fmt_diversity: {
    label: 'Receiver payment channel variety',
    explain: () => 'Receiver accepts payments across varied settlement methods.',
  },
};

/** Prettifies unknown feature names */
export function prettifyFeatureName(feature) {
  if (!feature) return 'Unknown attribute';
  if (FEATURE_INFO[feature]?.label) return FEATURE_INFO[feature].label;

  let clean = feature
    .replace(/^[tag]_/, '')
    .replace(/_log$/, '')
    .replace(/_z_in_currency$/, ' (relative size)')
    .replace(/_/g, ' ');
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

/** Generates plain-language explanation sentence */
export function getFeatureExplanation(feature, value, shapValue) {
  const info = FEATURE_INFO[feature];
  if (info?.explain) {
    return info.explain(value, shapValue);
  }
  const pretty = prettifyFeatureName(feature);
  if (shapValue > 0) {
    return `${pretty} contributed toward the elevated risk score.`;
  }
  return `${pretty} reduced overall suspicion score.`;
}

/** Determines badge impact from SHAP magnitude */
export function getImpactLevel(shapValue) {
  const abs = Math.abs(shapValue || 0);
  if (abs >= 0.5) return 'High';
  if (abs >= 0.1) return 'Medium';
  return 'Low';
}
