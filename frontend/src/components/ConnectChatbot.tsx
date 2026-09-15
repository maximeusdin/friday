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
 * Full "use Friday in your chatbot" instructions. Rendered in the
 * "Use in Chatbots" info modal and inside How to Use.
 */
export function ChatbotConnectBody() {
  return (
    <>
      <p>
        Friday is available as a <strong>connector</strong> for AI chatbots: add it once, and
        your chatbot can search the archives, resolve codenames, and read documents directly
        in your conversations — citing pages with links that open Friday&apos;s document
        viewer. Free, no Friday account needed.
      </p>
      <p>
        The connector uses <strong>MCP</strong> (Model Context Protocol), the open standard
        for connecting AI assistants to external tools — it works with Claude, ChatGPT, and
        any other chatbot or AI tool that accepts custom MCP connectors. The connector URL
        is the same everywhere:
      </p>
      <McpUrlBox />

      <p><strong>Claude (claude.ai — Free, Pro, or Max):</strong></p>
      <AddToClaudeButton />
      <ol className="claude-steps">
        <li>
          Open{' '}
          <a href={CLAUDE_CONNECTORS_URL} target="_blank" rel="noopener noreferrer">
            claude.ai &rarr; Settings &rarr; Connectors
          </a>{' '}
          (the Add to Claude button above copies the URL and opens that page for you).
        </li>
        <li>Click <strong>Add custom connector</strong>, paste the URL, and click <strong>Add</strong>. No login or API key is required.</li>
        <li>In any chat, open the <strong>+</strong> (tools) menu near the message box and make
          sure the Friday connector is enabled. For deep dives, use <strong>Research</strong> mode
          with Friday enabled as a source.</li>
        <li><strong>Claude Desktop &amp; mobile:</strong> same flow — Settings &rarr; Connectors &rarr;
          Add custom connector. <strong>Team / Enterprise:</strong> an admin adds the connector
          first (Admin settings &rarr; Connectors); members then enable it under Settings &rarr;
          Connectors.</li>
      </ol>

      <p><strong>ChatGPT (Plus, Pro, Business, or Enterprise):</strong></p>
      <ol className="claude-steps">
        <li>Open <strong>Settings &rarr; Connectors</strong>. If there is no option to add a
          custom connector, enable <strong>Developer mode</strong> first (under
          Settings &rarr; Connectors &rarr; Advanced).</li>
        <li>Choose <strong>Create</strong> / <strong>Add custom connector</strong>, give it a name
          (e.g. &ldquo;Friday&rdquo;), paste the URL as the MCP server URL, and select
          <strong> No authentication</strong>.</li>
        <li>In a chat, enable the Friday connector from the tools/connectors menu, then ask
          your question. ChatGPT&apos;s <strong>Deep research</strong> can also use Friday as a source.</li>
      </ol>

      <p><strong>Any other chatbot or AI tool:</strong></p>
      <ul>
        <li>Wherever the app asks for a <strong>remote MCP server</strong> or
          <strong> custom connector</strong>, paste the URL above (transport: HTTP; no
          authentication). This covers Microsoft Copilot Studio, Cursor, VS Code,
          LM Studio, and most other MCP-capable clients.</li>
        <li>
          <strong>Claude Code:</strong> run
          <span className="claude-url-box claude-url-box-inline"><code>claude mcp add --transport http friday {MCP_URL}</code></span>
        </li>
      </ul>

      <p><strong>Then, in any chat:</strong> ask questions in plain language —
        &ldquo;Who was ALES in the Venona decrypts?&rdquo; Mentioning Friday or the archives
        (&ldquo;check the Friday archive&rdquo;) nudges the chatbot to search it. Approve the tool
        calls when prompted, or choose &ldquo;Allow always.&rdquo;</p>

      <p>
        The chatbot uses Friday&apos;s search the way a researcher uses the Search tab: it looks up
        cover names in the concordance, runs keyword searches, and reads the OCR text of the
        pages it finds. Every citation links back to the scanned original here on Friday.
      </p>
    </>
  );
}
