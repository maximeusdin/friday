'use client';

import type { SearchPageHitItem } from '@/lib/api';
import type { EvidenceRef } from '@/types/api';
import { Icon } from './ui/Icon';

interface SearchResultsListProps {
  items: SearchPageHitItem[];
  totalHits: number;
  onOpenPage: (evidence: EvidenceRef, resultSetId: string) => void;
  resultSetId: string;
  isLoading?: boolean;
  /** Render hidden (removed) rows greyed out with a Restore control. */
  showHidden?: boolean;
  /** When provided, each row gets a remove / restore control. */
  onSetItemHidden?: (item: SearchPageHitItem, hidden: boolean) => void;
}

export function SearchResultsList({
  items, totalHits, onOpenPage, resultSetId, isLoading, showHidden, onSetItemHidden,
}: SearchResultsListProps) {
  if (isLoading) {
    return <div className="loading"><span className="spinner" /> Loading results…</div>;
  }

  const visibleItems = items.filter((it) => !it.hidden);
  const hiddenCount = items.length - visibleItems.length;
  // Numbering skips hidden rows so visible numbers stay sequential (1, 2, 3, …)
  // and a researcher can use them as bookmarks; hidden rows show no number.
  const displayItems = showHidden ? items : visibleItems;
  let visibleNum = 0;

  if (displayItems.length === 0) {
    return (
      <div className="empty-state">
        {items.length === 0
          ? 'No matching pages found.'
          : 'Every result in this search has been removed.'}
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
      resultSetId,
    );
  };

  const collectionName = (item: SearchPageHitItem) =>
    item.collection.title || item.collection.slug || 'Unknown';

  return (
    <div className="hit-list">
      {displayItems.map((item, idx) => {
        if (!item.hidden) visibleNum += 1;
        const num = item.hidden ? null : visibleNum;
        return (
          <div
            key={`${item.document.id}-${item.page.id}-${idx}`}
            className={`hit${item.hidden ? ' is-hidden' : ''}`}
          >
            <span className="hit-num">{num != null ? num : '·'}</span>

            <div
              className="hit-main"
              role="button"
              tabIndex={0}
              onClick={() => handleOpen(item)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleOpen(item);
                }
              }}
              title={`Open page ${item.page.pdf_page} in ${collectionName(item)}`}
            >
              <span className="hit-title">
                {collectionName(item)}
                <span className="count">page {item.page.pdf_page}</span>
                {item.hidden && <span className="chip">removed</span>}
              </span>
              {/* Which file inside the collection: some collections hold dozens,
                  and the collection name alone doesn't say where you have landed. */}
              {item.document?.title && (
                <span className="hit-doc" title={item.document.title}>{item.document.title}</span>
              )}
              <span className="hit-snippet">
                {item.snippet || `View page ${item.page.pdf_page}`}
              </span>
            </div>

            {onSetItemHidden && (
              <span className="hit-actions">
                <button
                  type="button"
                  className="icon-btn icon-btn-sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSetItemHidden(item, !item.hidden);
                  }}
                  title={item.hidden ? 'Restore this result' : 'Remove this result from the search (reversible)'}
                  aria-label={item.hidden ? 'Restore result' : `Remove result ${num}`}
                >
                  <Icon name={item.hidden ? 'restore' : 'close'} size={15} />
                </button>
              </span>
            )}
          </div>
        );
      })}

      <div className="hit-foot">
        <span>
          Showing {visibleItems.length.toLocaleString()} of {totalHits.toLocaleString()} page hits
          {hiddenCount > 0 && ` · ${hiddenCount} removed`}
        </span>
      </div>
    </div>
  );
}
