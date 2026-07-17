'use client';

import { useState } from 'react';
import type { ClarificationPlan, ClarificationAnswer } from '@/types/api';

interface Props {
  clarification: ClarificationPlan;
  onSubmit: (answers: ClarificationAnswer[]) => void;
  disabled?: boolean;
}

interface LocalAnswer {
  option_ids: string[];
  free_text: string;
}

const OTHER = '__other__';

/**
 * Renders the v12 follow-up questions (single/multi choice + free text) and
 * collects answers. On submit it emits ClarificationAnswer[] which the caller
 * forwards as carry_context to re-run the query with resolved intent.
 */
export function ClarificationCard({ clarification, onSubmit, disabled }: Props) {
  const questions = clarification.questions || [];
  const [answers, setAnswers] = useState<Record<string, LocalAnswer>>(() =>
    Object.fromEntries(questions.map((q) => [q.id, { option_ids: [], free_text: '' }]))
  );

  const update = (qid: string, patch: Partial<LocalAnswer>) =>
    setAnswers((prev) => ({ ...prev, [qid]: { ...prev[qid], ...patch } }));

  const toggleSingle = (qid: string, oid: string) => update(qid, { option_ids: [oid] });
  const toggleMulti = (qid: string, oid: string) =>
    setAnswers((prev) => {
      const cur = prev[qid].option_ids;
      const next = cur.includes(oid) ? cur.filter((x) => x !== oid) : [...cur, oid];
      return { ...prev, [qid]: { ...prev[qid], option_ids: next } };
    });

  // At least one question must have a usable answer.
  const ready = questions.some((q) => {
    const a = answers[q.id];
    if (!a) return false;
    const picksReal = a.option_ids.some((id) => id !== OTHER);
    const hasFree = (a.option_ids.includes(OTHER) || q.kind === 'free_text') && a.free_text.trim().length > 0;
    return picksReal || hasFree;
  });

  const handleSubmit = () => {
    const out: ClarificationAnswer[] = questions.map((q) => {
      const a = answers[q.id] || { option_ids: [], free_text: '' };
      const realOpts = a.option_ids.filter((id) => id !== OTHER);
      const wantsFree = a.option_ids.includes(OTHER) || q.kind === 'free_text';
      return {
        question_id: q.id,
        option_ids: realOpts,
        free_text: wantsFree && a.free_text.trim() ? a.free_text.trim() : undefined,
      };
    }).filter((a) => (a.option_ids && a.option_ids.length) || a.free_text);
    onSubmit(out);
  };

  return (
    <div className="card" style={{ borderColor: 'var(--color-accent)', margin: 'var(--spacing-md)' }}>
      <div className="text-sm" style={{ fontWeight: 600, marginBottom: 'var(--spacing-sm)' }}>
        A couple of quick questions to get this right:
      </div>

      {questions.map((q) => {
        const a = answers[q.id] || { option_ids: [], free_text: '' };
        return (
          <div key={q.id} style={{ marginBottom: 'var(--spacing-md)' }}>
            <div className="text-sm" style={{ marginBottom: '6px' }}>{q.question}</div>

            {q.kind === 'free_text' ? (
              <textarea
                className="input"
                rows={2}
                style={{ width: '100%' }}
                placeholder="Type your answer…"
                value={a.free_text}
                disabled={disabled}
                onChange={(e) => update(q.id, { free_text: e.target.value })}
              />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {q.options.map((o) => {
                  const checked = a.option_ids.includes(o.id);
                  return (
                    <label key={o.id} className="text-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                      <input
                        type={q.kind === 'multi_choice' ? 'checkbox' : 'radio'}
                        name={q.id}
                        checked={checked}
                        disabled={disabled}
                        onChange={() => (q.kind === 'multi_choice' ? toggleMulti(q.id, o.id) : toggleSingle(q.id, o.id))}
                      />
                      <span>{o.label}{o.hint ? <span className="text-muted"> — {o.hint}</span> : null}</span>
                    </label>
                  );
                })}
                {q.allow_free_text && (
                  <label className="text-sm" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                    <input
                      type={q.kind === 'multi_choice' ? 'checkbox' : 'radio'}
                      name={q.id}
                      checked={a.option_ids.includes(OTHER)}
                      disabled={disabled}
                      onChange={() => (q.kind === 'multi_choice'
                        ? toggleMulti(q.id, OTHER)
                        : toggleSingle(q.id, OTHER))}
                    />
                    <span>Something else:</span>
                    <input
                      className="input"
                      style={{ flex: 1 }}
                      placeholder="describe…"
                      value={a.free_text}
                      disabled={disabled}
                      onChange={(e) => update(q.id, { free_text: e.target.value })}
                    />
                  </label>
                )}
              </div>
            )}
          </div>
        );
      })}

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <button className="btn-primary" onClick={handleSubmit} disabled={disabled || !ready}>
          Continue
        </button>
        <button className="btn-secondary" onClick={() => onSubmit([])} disabled={disabled}>
          Skip — just answer
        </button>
      </div>
    </div>
  );
}
