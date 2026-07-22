'use client';

import { useState } from 'react';

/** Public MCP endpoint for the Friday connector (mounted by the API at /mcp). */
export const MCP_URL = process.env.NEXT_PUBLIC_MCP_URL || 'https://api.fridayarchive.org/mcp';

const CLAUDE_CONNECTORS_URL = 'https://claude.ai/settings/connectors';

async function copyMcpUrl(): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(MCP_URL);
    return true;
  } catch {
    return false;
  }
}

/**
 * One-click "Add to Claude" CTA.
 *
 * Claude has no deep link that installs a connector directly, so one click does
 * the next best thing: copy the connector URL to the clipboard and open Claude's
 * connector settings in a new tab. `onShowInstructions` (optional) opens the
 * instructions so they're waiting when the user returns to this tab.
 */
export function AddToClaudeButton({ onShowInstructions }: { onShowInstructions?: () => void }) {
  const [copied, setCopied] = useState(false);

  const handleClick = async () => {
    const ok = await copyMcpUrl();
    setCopied(ok);
    window.open(CLAUDE_CONNECTORS_URL, '_blank', 'noopener');
    onShowInstructions?.();
  };

  return (
    <div className="claude-cta-wrap">
      <button type="button" className="claude-cta" onClick={handleClick}>
        Add to Claude
      </button>
      {copied && (
        <span className="claude-cta-copied">
          Connector URL copied — in the Claude tab: <strong>Add custom connector</strong>, paste, <strong>Add</strong>.
        </span>
      )}
    </div>
  );
}

/** The connector URL in a copyable box. */
function McpUrlBox() {
  const [copied, setCopied] = useState(false);
  return (
    <div className="claude-url-box">
      <code>{MCP_URL}</code>
      <button
        type="button"
        className="claude-copy-btn"
        onClick={async () => setCopied(await copyMcpUrl())}
      >
        {copied ? 'Copied ✓' : 'Copy'}
      </button>
    </div>
  );
}

/**
 * Full "use Friday in Claude" instructions. Rendered in the "Use in Claude"
 * info modal and inside How to Use.
 */
export function ClaudeConnectBody() {
  return (
    <>
      <p>
        Friday is available as a <strong>connector</strong> for Claude: add it once, and
        Claude can search the archives, resolve codenames, and read documents directly in
        your Claude conversations — citing pages with links that open Friday&apos;s document
        viewer. Free, no Friday account needed.
      </p>

      <AddToClaudeButton />

      <p><strong>Connect in claude.ai (Free, Pro, or Max):</strong></p>
      <ol className="claude-steps">
        <li>
          Copy the connector URL:
          <McpUrlBox />
        </li>
        <li>
          Open{' '}
          <a href={CLAUDE_CONNECTORS_URL} target="_blank" rel="noopener noreferrer">
            claude.ai &rarr; Settings &rarr; Connectors
          </a>{' '}
          (the Add to Claude button above does both of these steps for you).
        </li>
        <li>Click <strong>Add custom connector</strong>, paste the URL, and click <strong>Add</strong>. No login or API key is required.</li>
      </ol>

      <p><strong>Then, in any Claude chat:</strong></p>
      <ul>
        <li>Open the <strong>+</strong> (tools) menu near the message box and make sure the
          Friday connector is enabled for the conversation.</li>
        <li>Ask questions in plain language — &ldquo;Who was ALES in the Venona decrypts?&rdquo;
          Mentioning Friday or the archives (&ldquo;check the Friday archive&rdquo;) nudges Claude to
          search it. Approve the tool calls when prompted, or choose &ldquo;Allow always.&rdquo;</li>
        <li>For deep dives, use Claude&apos;s <strong>Research</strong> mode with Friday enabled as a
          source — Claude will run many searches and cite documents throughout its report.</li>
      </ul>

      <p><strong>Other Claude apps:</strong></p>
      <ul>
        <li><strong>Claude Desktop &amp; mobile:</strong> same flow — Settings &rarr; Connectors &rarr;
          Add custom connector, paste the URL.</li>
        <li>
          <strong>Claude Code:</strong> run
          <span className="claude-url-box claude-url-box-inline"><code>claude mcp add --transport http friday {MCP_URL}</code></span>
        </li>
        <li><strong>Team / Enterprise plans:</strong> an admin must add the connector first
          (Admin settings &rarr; Connectors), after which members enable it under Settings &rarr;
          Connectors.</li>
      </ul>

      <p>
        Claude uses Friday&apos;s search the way a researcher uses the Search tab: it looks up
        cover names in the concordance, runs keyword searches, and reads the OCR text of the
        pages it finds. Every citation links back to the scanned original here on Friday.
      </p>
    </>
  );
}
