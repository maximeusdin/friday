'use client';

import type { SearchPageHitItem } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';

interface SearchResultsListProps {
  items: SearchPageHitItem[];
  totalHits: number;
  onOpenPage: (evidence: EvidenceRef, resultSetId: string) => void;
  resultSetId: string;
  isLoading?: boolean;
}

export function SearchResultsList({
  items,
  totalHits,
  onOpenPage,
  resultSetId,
  isLoading,
}: SearchResultsListProps) {
  if (isLoading) {
    return (
      <div className="search-results-loading">
        <div className="search-progress-spinner" />
        <p>Loading results…</p>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="search-results-empty">
        <p>No matching pages found.</p>
      </div>
    );
  }

  const handleOpen = (item: SearchPageHitItem) => {
    onOpenPage(
      {
        document_id: item.evidence_ref.document_id,
        pdf_page: item.evidence_ref.pdf_page,
        chunk_id: item.evidence_ref.chunk_id,
        quote: item.evidence_ref.quote,
      },
      resultSetId
    );
  };

  const collectionName = (item: SearchPageHitItem) =>
    item.collection.title || item.collection.slug || 'Unknown';

  return (
    <div className="search-results-list">
      {items.map((item, idx) => (
        <div
          key={`${item.document.id}-${item.page.id}-${idx}`}
          className="search-result-row"
        >
          <div className="search-result-header">
            <span className="search-result-num">{idx + 1}.</span>
            <button
              type="button"
              className="search-result-collection-link"
              onClick={(e) => {
                e.stopPropagation();
                handleOpen(item);
              }}
              title={`Open page ${item.page.pdf_page} in ${collectionName(item)}`}
            >
              {collectionName(item)}
            </button>
            <span className="search-result-meta">
              <span className="search-result-page">p. {item.page.pdf_page}</span>
            </span>
          </div>
          {item.snippet ? (
            <div
              className="search-result-snippet"
              role="button"
              tabIndex={0}
              onClick={() => handleOpen(item)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleOpen(item);
                }
              }}
            >
              {item.snippet}
            </div>
          ) : (
            <div
              className="search-result-snippet search-result-snippet-empty"
              role="button"
              tabIndex={0}
              onClick={() => handleOpen(item)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleOpen(item);
                }
              }}
            >
              View page {item.page.pdf_page}
            </div>
          )}
        </div>
      ))}
      <div className="search-results-footer">
        Showing {items.length} of {totalHits} page hits
      </div>
    </div>
  );
}
