'use client';

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { Plan, ResultSetResponse } from '@/types/api';

interface PlanCardProps {
  plan: Plan | null;
  onPlanIdUpdate: (planId: number | null) => void;
  onResultSetIdUpdate: (resultSetId: number | null) => void;
}

export function PlanCard({ plan, onPlanIdUpdate, onResultSetIdUpdate }: PlanCardProps) {
  const queryClient = useQueryClient();

  const approveMutation = useMutation({
    mutationFn: (planId: number) => api.approvePlan(planId),
    onSuccess: (updatedPlan) => {
      onPlanIdUpdate(updatedPlan.id);
    },
  });

  const executeMutation = useMutation({
    mutationFn: (planId: number) => api.executePlan(planId),
    onSuccess: (response) => {
      onPlanIdUpdate(response.plan.id);
      onResultSetIdUpdate(response.result_set.id);
    },
  });

  const clarifyMutation = useMutation({
    mutationFn: ({ planId, choiceId }: { planId: number; choiceId: number }) =>
      api.clarifyPlan(planId, { choice_id: choiceId }),
    onSuccess: (newPlan) => {
      // Refresh conversation (system message recorded) and swap active plan
      queryClient.invalidateQueries({ queryKey: ['messages', newPlan.session_id] });
      onPlanIdUpdate(newPlan.id);
      onResultSetIdUpdate(newPlan.result_set_id ?? null);
    },
  });

  if (!plan) {
    return (
      <div className="empty-state">
        <p>No plan yet</p>
        <p className="text-sm">Send a message to generate a research plan</p>
      </div>
    );
  }

  const needsClarification =
    (plan.plan_json as any)?.needs_clarification === true;

  const primitives: Array<Record<string, unknown>> =
    (plan.plan_json as any)?.primitives ??
    (plan.plan_json as any)?.query?.primitives ??
    [];

  const canApprove = plan.status === 'proposed';
  const canExecute = plan.status === 'approved';
  const isExecuted = plan.status === 'executed';

  return (
    <div>
      {/* Plan header */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Plan #{plan.id}</span>
          <span className={`badge badge-${plan.status}`}>{plan.status}</span>
        </div>

        {/* User query */}
        <div className="text-sm text-muted mb-sm">Query:</div>
        <div className="mb-md">{plan.user_utterance}</div>

        {/* Plan summary */}
        {plan.plan_summary && (
          <>
            <div className="text-sm text-muted mb-sm">Summary:</div>
            <div className="mb-md">{plan.plan_summary}</div>
          </>
        )}

        {/* Clarification choices */}
        {needsClarification && Array.isArray((plan.plan_json as any)?.choices) && (
          <div className="card" style={{ background: 'var(--color-highlight)' }}>
            <div className="card-title mb-sm">Clarification needed</div>
            <div className="text-sm text-muted mb-sm">
              Temporary UI affordance: click a choice below (or reply in chat with 1/2/3).
            </div>
            <ol style={{ paddingLeft: '1.25rem' }}>
              {((plan.plan_json as any).choices as string[]).map((c, idx) => {
                const choiceId = idx + 1; // 1-based
                return (
                  <li key={idx} style={{ marginBottom: 'var(--spacing-xs)' }}>
                    <div style={{ marginBottom: 'var(--spacing-xs)' }}>{c}</div>
                    <button
                      className="btn-primary"
                      onClick={() => clarifyMutation.mutate({ planId: plan.id, choiceId })}
                      disabled={clarifyMutation.isPending}
                    >
                      {clarifyMutation.isPending ? 'Applying...' : `Choose ${choiceId}`}
                    </button>
                  </li>
                );
              })}
            </ol>
            {clarifyMutation.error && (
              <div className="mt-md" style={{ color: 'var(--color-danger)' }}>
                {String(clarifyMutation.error)}
              </div>
            )}
          </div>
        )}

        {/* Primitives */}
        <div className="text-sm text-muted mb-sm">
          Steps ({primitives.length || 0}):
        </div>
        <div style={{ marginBottom: 'var(--spacing-md)' }}>
          {primitives.map((primitive, idx) => (
            <PrimitiveStep key={idx} index={idx + 1} primitive={primitive} />
          ))}
        </div>

        {/* Action buttons */}
        <div className="flex gap-sm">
          {canApprove && !needsClarification && (
            <button
              className="btn-success flex-1"
              onClick={() => approveMutation.mutate(plan.id)}
              disabled={approveMutation.isPending}
            >
              {approveMutation.isPending ? 'Approving...' : 'Approve Plan'}
            </button>
          )}
          {canExecute && !needsClarification && (
            <button
              className="btn-primary flex-1"
              onClick={() => executeMutation.mutate(plan.id)}
              disabled={executeMutation.isPending}
            >
              {executeMutation.isPending ? 'Executing...' : 'Execute Plan'}
            </button>
          )}
          {isExecuted && plan.result_set_id && (
            <div className="text-sm text-muted">
              ✓ Executed · Result Set #{plan.result_set_id}
            </div>
          )}
        </div>

        {/* Error display */}
        {(approveMutation.error || executeMutation.error) && (
          <div className="mt-md" style={{ color: 'var(--color-danger)' }}>
            {String(approveMutation.error || executeMutation.error)}
          </div>
        )}
      </div>

      {/* Execution envelope (if present) */}
      {plan.plan_json.execution_envelope && (
        <div className="card">
          <div className="card-title mb-sm">Execution Settings</div>
          <div className="text-sm">
            {plan.plan_json.execution_envelope.top_k && (
              <div>Top K: {plan.plan_json.execution_envelope.top_k}</div>
            )}
            {plan.plan_json.execution_envelope.search_type && (
              <div>Search Type: {plan.plan_json.execution_envelope.search_type}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function PrimitiveStep({
  index,
  primitive,
}: {
  index: number;
  primitive: Record<string, unknown>;
}) {
  const { type, ...params } = primitive;

  // Format primitive for display
  const formatValue = (value: unknown): string => {
    if (typeof value === 'string') return `"${value}"`;
    if (Array.isArray(value)) return value.map(formatValue).join(', ');
    return String(value);
  };

  const paramEntries = Object.entries(params).filter(
    ([key]) => !key.startsWith('_')
  );

  return (
    <div
      style={{
        padding: 'var(--spacing-sm)',
        background: 'var(--color-bg-secondary)',
        borderRadius: 'var(--radius-sm)',
        marginBottom: 'var(--spacing-xs)',
        fontFamily: 'var(--font-mono)',
        fontSize: '12px',
      }}
    >
      <span className="text-muted">{index}.</span>{' '}
      <strong>{String(type)}</strong>
      {paramEntries.length > 0 && (
        <span className="text-muted">
          {' '}
          ({paramEntries.map(([k, v]) => `${k}=${formatValue(v)}`).join(', ')})
        </span>
      )}
    </div>
  );
}
