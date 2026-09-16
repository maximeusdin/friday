'use client';

import { useState } from 'react';
import { ChatbotConnectBody } from './ConnectChatbot';
import { CollectionsDownloadsBody } from './CollectionsDownloads';
import { Modal } from './ui/Modal';

export type HelpSection = 'about' | 'collections' | 'howto' | 'chatsearch' | 'connect';

export const HELP_SECTIONS: { key: HelpSection; label: string }[] = [
  { key: 'howto', label: 'How to use Friday' },
  { key: 'chatsearch', label: 'Chat vs. Search' },
  { key: 'collections', label: 'Collections & downloads' },
  { key: 'connect', label: 'Use in chatbots' },
  { key: 'about', label: 'About & funding' },
];

const TITLES: Record<HelpSection, string> = {
  about: 'About Friday',
  collections: 'Collections & downloads',
  howto: 'How to use Friday',
  chatsearch: 'Chat vs. Search',
  connect: 'Use Friday in your chatbot',
};

/**
 * HelpModal — every explanatory surface in one dialog with a section rail.
 *
 * These five documents used to be five separate header buttons opening five
 * separate modals, which meant reading one and closing it to find the next. The
 * rail keeps them one click apart, and collapses to a scrolling tab strip on
 * narrow screens.
 */
export function HelpModal({
  section,
  onClose,
  initialCollectionId,
}: {
  section: HelpSection;
  onClose: () => void;
  /** For the collections section: expand this collection on open. */
  initialCollectionId?: number;
}) {
  const [active, setActive] = useState<HelpSection>(section);

  return (
    <Modal
      title={TITLES[active]}
      onClose={onClose}
      wide
      bare
    >
      <div className="modal-split">
        <nav className="modal-nav" aria-label="Help sections">
          {HELP_SECTIONS.map((s) => (
            <button
              key={s.key}
              type="button"
              className="modal-nav-item"
              aria-current={active === s.key}
              onClick={() => setActive(s.key)}
            >
              {s.label}
            </button>
          ))}
        </nav>
        <div className="help-body">
          {active === 'about' && <AboutBody />}
          {active === 'collections' && (
            <CollectionsDownloadsBody initialCollectionId={initialCollectionId} />
          )}
          {active === 'howto' && <HowToBody />}
          {active === 'chatsearch' && <ChatVsSearchBody />}
          {active === 'connect' && <ChatbotConnectBody />}
        </div>
      </div>
    </Modal>
  );
}

function HowToBody() {
  return (
    <div className="prose">
      <p className="help-lede">
        Friday searches scanned, OCR-processed declassified archives and answers with
        citations to the actual pages. It only reports what is in the documents. It adds no
        outside knowledge, and it tells you when the files don&apos;t answer your question.
      </p>

      <h3>Asking questions that maximize returns</h3>
      <ul>
        <li><strong>Name names.</strong> Include the people, organizations, and places you care
          about. Friday automatically expands cover names and aliases (asking about Golos also
          finds &ldquo;Sound&rdquo;; asking &ldquo;Who is Jurist?&rdquo; resolves the codename), but it can only
          do that when a name is in the question.</li>
        <li><strong>Include distinctive details.</strong> Rare numbers, dates, and quoted phrases
          are gold: &ldquo;500-600 meetings&rdquo; or a distinctive phrase from a document pins the search
          to the exact passage.</li>
        <li><strong>Direct questions are fine.</strong> Friday rewrites questions like &ldquo;When was
          X recruited?&rdquo; into the archive&apos;s own record language (&ldquo;initial contact&rdquo;,
          &ldquo;memorandum&rdquo;, &ldquo;informant&rdquo;) behind the scenes, because FBI files rarely use everyday
          words. Phrasing a question around records (&ldquo;what records exist about&hellip;&rdquo;) works well too.</li>
        <li><strong>Lists and counts search the whole archive.</strong> Questions like
          &ldquo;Which journalists were recruited?&rdquo; or &ldquo;How many engineers&hellip;?&rdquo; make Friday sweep
          every collection and read the full pool of pages, so people spread across many files
          aren&apos;t missed.</li>
        <li><strong>Scope narrows the hunt.</strong> The scope chip beside the message box limits
          a session to the collections or files you pick, and applies to both Chat and Search.
          You can also just say it in the question (&ldquo;&hellip;in the Vassiliev notebooks&rdquo;).</li>
        <li><strong>Click the citations.</strong> Evidence links open the document at the right
          page with the supporting passage highlighted. The quoted passage shown is taken
          verbatim from the document, never paraphrased.</li>
      </ul>

      <h3>Reading the answers</h3>
      <p>
        Findings with a source are grounded in the page they cite. Anything under an{' '}
        <em>unverified</em> heading did not pass citation checks, so treat it as a lead and follow it
        through to the sources. When Friday says it found no evidence, that is a statement about
        the search, not proof the fact is absent from the archive. Try again with more specific
        names or details, or use <em>Think deeper</em> to push the investigation further.
      </p>

      <h3>Defaults</h3>
      <p>
        Friday searches the entire archive, puts deep effort into every query, expands aliases
        and codenames, and rewrites questions into the archive&apos;s own record language. Think
        deeper goes further still, reusing the evidence it has already gathered. In the Search
        tab, exact matching is the default; Fuzzy (for OCR errors and typos) and Aliases are the
        toggles beside the search box.
      </p>

      <h3>What Friday doesn&apos;t do</h3>
      <p>
        It can&apos;t read text the OCR mangled beyond recognition, though Fuzzy in Search often
        catches near-miss spellings. It won&apos;t speculate beyond the documents, and it never
        searches outside the indexed collections.
      </p>
    </div>
  );
}

function ChatVsSearchBody() {
  return (
    <div className="prose">
      <p className="help-lede">
        A rule of thumb: <em>Search finds pages; Chat answers questions.</em> Researchers often
        begin in Chat, then switch to Search to exhaustively walk the pages behind an answer.
      </p>

      <h3>Search</h3>
      <p>
        A concordance, plain and literal. It matches your terms against every page and returns
        numbered page hits you can open, prune and export as CSV. Terms can be exact, boolean
        (<code>AND</code>, <code>OR</code>, <code>NOT</code>, quoted phrases) or fuzzy. It never
        interprets your query. Use Search when you know words that actually appear on the page,
        when you want <em>every</em> occurrence rather than a summary, or when you&apos;re
        building a citation list. Each search becomes a tab in your session, and the numbering
        lets you stop at hit 40 and pick it up next week.
      </p>

      <h3>Chat</h3>
      <p>
        An investigative assistant: it plans the question, runs many searches (semantic and
        exact, in your words and the archive&apos;s), resolves codenames, reads the retrieved
        pages, and writes an answer with citations. Use Chat when you have a question rather
        than a term, when your wording may not match the documents&apos; vocabulary, or when the
        answer must be assembled from several documents.
      </p>

      <h3>They meet in the middle</h3>
      <p>
        Both share the same session and the same scope. Chat runs the same boolean search engine
        you do, and every search it runs is saved into your session under{' '}
        <strong>Chat&apos;s searches</strong> in the Search tab. Open one to see exactly what Chat
        looked at, prune its results, or continue the investigation yourself where it left off.
      </p>
    </div>
  );
}

function AboutBody() {
  return (
    <div className="prose">
      <h3>What is Friday?</h3>
      <p>
        Friday is a research assistant for declassified Cold War archives. It holds scanned,
        OCR-processed collections: FBI files, HUAC hearings, the Vassiliev notebooks, the Venona
        decrypts and more. There are two ways to work across them. Search is literal and matches
        your terms against every page. Chat plans a question, reads the pages it retrieves,
        resolves cover names against a concordance, and answers with citations that open the
        scanned original at the cited page.
      </p>

      {/* Editorial placeholder — replace with the real origin story: who conceived and
          built Friday (its architects) and the institutions involved. Deliberately
          rendered as an obvious to-do rather than filler prose. */}
      <h3>Origins of Friday</h3>
      <p className="text-muted">
        [To be written: who conceived and built Friday, and the institutions behind it.]
      </p>

      <h3>Funding</h3>
      <p>
        Friday has been funded by a National Endowment for the Humanities Chairman&apos;s Grant,
        FEJ-310584-26, and an Emory University Heilbrun Distinguished Emeritus Fellowship.
      </p>

      <h3>Collections</h3>
      <p>
        Every collection, and every file inside it, can be browsed and downloaded under{' '}
        <strong>Collections &amp; downloads</strong>, one file at a time or in bulk.
      </p>
    </div>
  );
}
