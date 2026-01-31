'use client';

import type { ResultSetResponse, ResultItem, EvidenceRef } from '@/types/api';

interface ResultsListProps {
  resultSet: ResultSetResponse | null;
  onEvidenceClick: (evidence: EvidenceRef) => void;
}

export function ResultsList({ resultSet, onEvidenceClick }: ResultsListProps) {
  if (!resultSet) {
    return (
      <div className="empty-state">
        <p>No results yet</p>
        <p className="text-sm">Execute a plan to see results</p>
      </div>
    );
  }

  return (
    <div>
      {/* Summary */}
      <div className="card">
        <div className="card-title mb-sm">{resultSet.name}</div>
        <div className="text-sm">
          <span>{resultSet.summary.item_count} items</span>
          <span className="text-muted"> · </span>
          <span>{resultSet.summary.document_count} documents</span>
          {resultSet.summary.entity_count !== undefined && (
            <>
              <span className="text-muted"> · </span>
              <span>{resultSet.summary.entity_count} entities</span>
            </>
          )}
        </div>
      </div>

      {/* Results */}
      {resultSet.items.length > 0 ? (
        resultSet.items.map((item) => (
          <ResultItemCard
            key={item.id}
            item={item}
            onEvidenceClick={onEvidenceClick}
          />
        ))
      ) : (
        <div className="empty-state">
          <p>No items in result set</p>
        </div>
      )}
    </div>
  );
}

function ResultItemCard({
  item,
  onEvidenceClick,
}: {
  item: ResultItem;
  onEvidenceClick: (evidence: EvidenceRef) => void;
}) {
  return (
    <div className="result-item">
      {/* Rank and kind */}
      <div className="result-rank">
        #{item.rank}
        {item.kind && <span className="text-muted"> · {item.kind}</span>}
        {item.scores?.hybrid !== undefined && (
          <span className="text-muted"> · score: {item.scores.hybrid.toFixed(3)}</span>
        )}
      </div>

      {/* Text / highlight */}
      <div
        className="result-text"
        dangerouslySetInnerHTML={{
          __html: item.highlight || escapeHtml(item.text),
        }}
      />

      {/* Matched terms */}
      {item.matched_terms && item.matched_terms.length > 0 && (
        <div className="text-sm text-muted mb-sm">
          Matched: {item.matched_terms.join(', ')}
        </div>
      )}

      {/* Citations */}
      {item.evidence_refs.length > 0 && (
        <div className="flex gap-sm" style={{ flexWrap: 'wrap' }}>
          {item.evidence_refs.map((ref, idx) => (
            <CitationLink
              key={idx}
              evidence={ref}
              onClick={() => onEvidenceClick(ref)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function CitationLink({
  evidence,
  onClick,
}: {
  evidence: EvidenceRef;
  onClick: () => void;
}) {
  return (
    <button
      className="citation-link"
      onClick={onClick}
      title={evidence.quote || `Document ${evidence.document_id}, page ${evidence.pdf_page}`}
    >
      <svg
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
      </svg>
      Doc {evidence.document_id}, p.{evidence.pdf_page}
    </button>
  );
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
