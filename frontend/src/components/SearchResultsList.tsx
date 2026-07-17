'use client';

import type { SearchPageHitItem } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';

interface SearchResultsListProps {
  items: SearchPageHitItem[];
  totalHits: number;
  onOpenPage: (evidence: EvidenceRef, resultSetId: string) => void;
  resultSetId: string;
  isLoading?: boolean;
  /** Render hidden (removed) rows grayed out with a Restore control. */
  showHidden?: boolean;
  /** When provided, each row shows a remove (✕) / restore control. */
  onSetItemHidden?: (item: SearchPageHitItem, hidden: boolean) => void;
}

export function SearchResultsList({
  items,
  totalHits,
  onOpenPage,
  resultSetId,
  isLoading,
  showHidden,
  onSetItemHidden,
}: SearchResultsListProps) {
  if (isLoading) {
    return (
      <div className="search-results-loading">
        <div className="search-progress-spinner" />
        <p>Loading results…</p>
      </div>
    );
  }

  const visibleItems = items.filter((it) => !it.hidden);
  const hiddenCount = items.length - visibleItems.length;
  // Numbering skips hidden rows so visible numbers stay sequential (1, 2, 3, …)
  // and a researcher can use them as bookmarks; hidden rows show no number.
  const displayItems = showHidden ? items : visibleItems;
  let visibleNum = 0;

  if (displayItems.length === 0) {
    return (
      <div className="search-results-empty">
        <p>{items.length === 0 ? 'No matching pages found.' : 'All results in this search have been removed.'}</p>
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
      {displayItems.map((item, idx) => {
        if (!item.hidden) visibleNum += 1;
        const num = item.hidden ? null : visibleNum;
        return (
          <div
            key={`${item.document.id}-${item.page.id}-${idx}`}
            className={`search-result-row${item.hidden ? ' search-result-row-hidden' : ''}`}
          >
            <div className="search-result-header">
              <span className="search-result-num">{num != null ? `${num}.` : '—'}</span>
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
                {item.hidden && <span className="search-result-removed-label">removed</span>}
                {onSetItemHidden && (
                  item.hidden ? (
                    <button
                      type="button"
                      className="search-result-restore"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSetItemHidden(item, false);
                      }}
                      title="Restore this result"
                      aria-label="Restore result"
                    >
                      Restore
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="search-result-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSetItemHidden(item, true);
                      }}
                      title="Remove this result from the search (restorable)"
                      aria-label={`Remove result ${num}`}
                    >
                      ✕
                    </button>
                  )
                )}
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
        );
      })}
      <div className="search-results-footer">
        Showing {visibleItems.length} of {totalHits} page hits
        {hiddenCount > 0 && ` · ${hiddenCount} removed`}
      </div>
    </div>
  );
}
