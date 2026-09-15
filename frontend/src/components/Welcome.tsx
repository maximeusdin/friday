'use client';

import { useState } from 'react';
import { api } from '@/lib/api';
import { AddToClaudeButton } from './ConnectChatbot';
import { ConcordanceModal } from './ConcordanceIndex';
import { HelpModal, type HelpSection } from './HelpModal';
import { Icon } from './ui/Icon';

const PROMPTS = [
  'Who was involved in the Rosenberg case?',
  'Which agents had contact with Klaus Fuchs?',
  'How did Soviet intelligence recruit in Washington?',
  'What do the files say about Harry Dexter White?',
];

/**
 * Welcome — what fills the thread before the first question.
 *
 * The old splash was a landing page stacked inside the chat pane: hero, four
 * example cards, a connector pitch, five documentation buttons and a
 * concordance card, all competing at the same weight. This keeps the one
 * primary action (ask something) at the top with the composer directly below
 * it, and demotes the rest to three quiet tiles.
 */
export function Welcome({ onAsk }: { onAsk: (question: string) => void }) {
  const [help, setHelp] = useState<HelpSection | null>(null);
  const [concordance, setConcordance] = useState(false);

  return (
    <div className="welcome">
      <div className="welcome-hero">
        <h1 className="welcome-title">Search the Cold War archives</h1>
        <p className="welcome-lede">
          Ask a question in plain language. Friday reads the declassified files and links
          every answer back to the scanned page.
        </p>
      </div>

      <section className="welcome-section">
        <h2 className="eyebrow">Try one of these</h2>
        <div className="prompt-grid">
          {PROMPTS.map((q) => (
            <button key={q} type="button" className="prompt-card" onClick={() => onAsk(q)}>
              <Icon name="search" size={16} />
              <span>{q}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="welcome-section">
        <h2 className="eyebrow">Also worth knowing</h2>
        <div className="tile-row">
          <div className="tile">
            <span className="tile-title">
              <Icon name="book" size={16} />
              Concordance index
            </span>
            <p className="tile-body">
              Proper names, cover names and organizations across twenty-one volumes of
              Vassiliev notebooks and Venona cables, cross-indexed to the real names behind
              the codenames.
            </p>
            <div className="flex gap-sm">
              <button type="button" className="btn-secondary btn-sm" onClick={() => setConcordance(true)}>
                Browse
              </button>
              <button
                type="button"
                className="btn-ghost btn-sm"
                onClick={() => window.open(api.getConcordancePdfUrl(), '_blank')}
              >
                <Icon name="download" size={14} />
                PDF
              </button>
            </div>
          </div>

          <div className="tile">
            <span className="tile-title">
              <Icon name="plug" size={16} />
              Use Friday in your chatbot
            </span>
            <p className="tile-body">
              Add the connector to Claude, ChatGPT or any MCP-capable assistant and research
              the archive straight from your own chats.
            </p>
            <AddToClaudeButton onShowInstructions={() => setHelp('connect')} />
          </div>

          <div className="tile">
            <span className="tile-title">
              <Icon name="library" size={16} />
              Collections
            </span>
            <p className="tile-body">
              Browse every collection and download single files, or take a whole
              collection as a zip.
            </p>
            <button type="button" className="btn-secondary btn-sm" onClick={() => setHelp('collections')}>
              Browse &amp; download
            </button>
          </div>
        </div>
      </section>

      {help && <HelpModal section={help} onClose={() => setHelp(null)} />}
      {concordance && <ConcordanceModal onClose={() => setConcordance(false)} />}
    </div>
  );
}
